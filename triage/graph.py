"""Assemble the heterogeneous graph from report-time bug records."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable

import torch
from torch_geometric.data import HeteroData

from .raw import RawBug, bug_sort_key, clean_value, component_key, load_raw_bugs, split_bug_reference_ids
from .splits import Masks, filter_to_causally_observed_labels, write_split_csv
from .text import TextEncoderConfig, encode_texts
from .util import write_json

EdgeType = tuple[str, str, str]
EdgeGroups = dict[EdgeType, list[tuple[int, int]]]


@dataclass(frozen=True)
class GraphConfig:
    history_ratio: float = 0.50
    train_ratio: float = 0.20
    val_ratio: float = 0.15
    min_history_dev_bugs: int = 10
    min_train_dev_bugs: int = 2
    max_developers: int = 100
    # The five research relation families.
    recency_expertise: bool = True
    shortcut_candidates: bool = True
    developer_collaboration: bool = True
    typed_history_links: bool = True
    relation_aware_similarity: bool = True
    similarity_top_k: int = 5
    similarity_batch_size: int = 256
    candidate_top_k: int = 10
    recency_days: tuple[int, ...] = (30, 180, 365)
    similarity_medium_quantile: float = 0.33
    similarity_high_quantile: float = 0.67


def add_edges(graph: HeteroData, edge_type: EdgeType, edges: Iterable[tuple[int, int]], unique: bool = True) -> int:
    edge_list = sorted(set(edges)) if unique else list(edges)
    if edge_list:
        graph[edge_type].edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        graph[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
    return len(edge_list)


def add_edge_groups(graph: HeteroData, edge_counts: dict[str, int], edge_groups: EdgeGroups) -> None:
    for edge_type in sorted(edge_groups):
        edge_counts[f"{edge_type[0]}_{edge_type[1]}"] = add_edges(graph, edge_type, edge_groups[edge_type])


def reversed_edges(edges: list[tuple[int, int]]) -> Iterable[tuple[int, int]]:
    return ((dst, src) for src, dst in edges)


def add_node_ids(graph: HeteroData) -> None:
    for node_type in graph.node_types:
        graph[node_type].node_id = torch.arange(graph[node_type].num_nodes, dtype=torch.long)


def values_to_ids(values: Iterable[str], skip_unknown: bool = False) -> dict[str, int]:
    cleaned = sorted({clean_value(value) for value in values if not (skip_unknown and clean_value(value) == "unknown")})
    return {value: idx for idx, value in enumerate(cleaned)}


def normalized_numeric_features(bugs: list[RawBug]) -> torch.Tensor:
    values = torch.tensor(
        [[math.log1p(float(len(bug.initial_text.split()))), math.log1p(float(len(bug.summary.split())))] for bug in bugs],
        dtype=torch.float32,
    )
    return (values - values.mean(dim=0, keepdim=True)) / values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)


def recency_bucket(event_time: datetime | None, reference_time: datetime, cutoffs: tuple[int, ...]) -> str | None:
    if event_time is None or event_time > reference_time:
        return None
    age_days = max(0, (reference_time - event_time).days)
    for cutoff in cutoffs:
        if age_days <= cutoff:
            return f"recent_{cutoff}d"
    return "older"


def add_recency_expertise_edges(
    edge_groups: EdgeGroups, relation_base: str, developer_id: int, component_id: int, product_id: int, bucket: str | None
) -> None:
    if bucket is None:
        return
    component_relation = f"{relation_base}_component_{bucket}"
    product_relation = f"{relation_base}_product_{bucket}"
    edge_groups[("developer", component_relation, "component")].append((developer_id, component_id))
    edge_groups[("component", f"rev_{component_relation}", "developer")].append((component_id, developer_id))
    edge_groups[("developer", product_relation, "product")].append((developer_id, product_id))
    edge_groups[("product", f"rev_{product_relation}", "developer")].append((product_id, developer_id))


def top_counter_keys(counter: Counter[int], limit: int) -> list[int]:
    return [key for key, _count in (counter.most_common(limit) if limit > 0 else counter.most_common())]


def build_developer_candidate_edges(
    bugs: list[RawBug],
    history_mask: torch.Tensor,
    label_to_id: dict[str, int],
    bug_component_ids: list[int],
    bug_product_ids: list[int],
    bug_reporter_ids: list[int | None],
    top_k: int,
) -> EdgeGroups:
    """Developer-to-bug shortcut candidates from the history seen before each bug."""
    component_history: defaultdict[int, Counter[int]] = defaultdict(Counter)
    product_history: defaultdict[int, Counter[int]] = defaultdict(Counter)
    reporter_history: defaultdict[int, Counter[int]] = defaultdict(Counter)
    edge_groups: EdgeGroups = defaultdict(list)
    for bug_idx, bug in enumerate(bugs):
        component_id, product_id, reporter_id = bug_component_ids[bug_idx], bug_product_ids[bug_idx], bug_reporter_ids[bug_idx]
        for developer_id in top_counter_keys(component_history[component_id], top_k):
            edge_groups[("developer", "candidate_by_component", "bug")].append((developer_id, bug_idx))
        for developer_id in top_counter_keys(product_history[product_id], top_k):
            edge_groups[("developer", "candidate_by_product", "bug")].append((developer_id, bug_idx))
        if reporter_id is not None:
            for developer_id in top_counter_keys(reporter_history[reporter_id], top_k):
                edge_groups[("developer", "candidate_by_reporter_history", "bug")].append((developer_id, bug_idx))
        if bool(history_mask[bug_idx]):
            developer_id = label_to_id[bug.assignee]
            component_history[component_id][developer_id] += 1
            product_history[product_id][developer_id] += 1
            if reporter_id is not None:
                reporter_history[reporter_id][developer_id] += 1
    return edge_groups


def similarity_score_bucket(score: float, medium_threshold: float, high_threshold: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def calibrated_similarity_thresholds(
    similarity_edges: list[tuple[int, int, float]],
    history_mask: torch.Tensor,
    medium_quantile: float,
    high_quantile: float,
    fallback: tuple[float, float] = (0.75, 0.90),
) -> tuple[float, float, int]:
    """Fit the score buckets on edges into history bugs only, never validation or test."""
    historical_scores = [score for _src, dst, score in similarity_edges if bool(history_mask[dst])]
    if len(historical_scores) < 3:
        return fallback[0], fallback[1], len(historical_scores)
    values = torch.tensor(historical_scores, dtype=torch.float32)
    medium = float(torch.quantile(values, medium_quantile).item())
    high = float(torch.quantile(values, high_quantile).item())
    return medium, max(medium, high), len(historical_scores)


def temporal_similarity_edges(
    features: torch.Tensor,
    created_at: torch.Tensor,
    top_k: int,
    batch_size: int,
    device: torch.device,
    progress_every: int = 10000,
) -> list[tuple[int, int, float]]:
    """Each bug's ``top_k`` most cosine-similar strictly older bugs, as (past, current, score)."""
    if top_k <= 0:
        return []
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Building temporal text similarity edges: nodes={features.size(0)}, top_k={top_k}, device={device}, dtype={dtype}", flush=True)
    normalized = features.float()
    normalized = (normalized / normalized.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(device=device, dtype=dtype)
    created_at = created_at.double().cpu()
    created_at_device = created_at.to(device)

    edges: list[tuple[int, int, float]] = []
    n = normalized.size(0)
    next_progress = progress_every
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        candidate_stop = int(torch.searchsorted(created_at, created_at[start:stop].max(), right=False).item())
        if candidate_stop == 0:
            continue
        scores = normalized[start:stop] @ normalized[:candidate_stop].t()
        previous_mask = created_at_device[:candidate_stop].view(1, -1) < created_at_device[start:stop].view(-1, 1)
        scores = scores.masked_fill(~previous_mask, float("-inf"))
        top_scores, top_indices = torch.topk(scores, k=min(top_k, candidate_stop), dim=1)
        for row, (row_scores, row_indices) in enumerate(zip(top_scores.float().cpu().tolist(), top_indices.cpu().tolist())):
            for score, past_idx in zip(row_scores, row_indices):
                if math.isfinite(score):
                    edges.append((int(past_idx), start + row, float(score)))
        if progress_every > 0 and stop >= next_progress:
            print(f"Similarity search processed {stop}/{n} bugs; edges={len(edges)}", flush=True)
            while stop >= next_progress:
                next_progress += progress_every
    print(f"Built {len(edges)} temporal text similarity edges.", flush=True)
    return edges


def candidate_edge_diagnostics(graph: HeteroData, labels: torch.Tensor, masks: dict[str, torch.Tensor]) -> dict[str, dict[str, float | int]]:
    """How often the true developer is among a bug's candidate shortcuts, per split."""
    candidates: defaultdict[int, set[int]] = defaultdict(set)
    for edge_type in graph.edge_types:
        source, relation, target = edge_type
        if source == "developer" and target == "bug" and relation.startswith("candidate_"):
            for developer_id, bug_idx in graph[edge_type].edge_index.t().tolist():
                candidates[int(bug_idx)].add(int(developer_id))
    result: dict[str, dict[str, float | int]] = {}
    for split_name, mask in masks.items():
        indices = mask.nonzero(as_tuple=False).view(-1).tolist()
        if not indices:
            result[split_name] = {"bugs": 0, "average_candidates": 0.0, "oracle_recall": 0.0}
            continue
        hit_count = sum(int(labels[idx]) in candidates.get(idx, set()) for idx in indices)
        result[split_name] = {
            "bugs": len(indices),
            "average_candidates": sum(len(candidates.get(idx, set())) for idx in indices) / len(indices),
            "oracle_recall": hit_count / len(indices),
        }
    return result


def assemble_graph(
    bugs: list[RawBug],
    label_to_id: dict[str, int],
    masks: Masks,
    boundaries: dict[str, str],
    text_features: torch.Tensor,
    encoder_name: str,
    config: GraphConfig,
    device: torch.device,
) -> HeteroData:
    """Build the HeteroData for bugs already sorted by creation time and filtered to eligible labels."""
    history_mask, train_mask, val_mask, test_mask = masks
    history_end = datetime.fromisoformat(boundaries["history_end_created_at"])
    y = torch.tensor([label_to_id[bug.assignee] for bug in bugs], dtype=torch.long)

    products = values_to_ids(bug.initial_product for bug in bugs)
    components = values_to_ids(component_key(bug.initial_product, bug.initial_component) for bug in bugs)
    reporters = values_to_ids((bug.reporter for bug in bugs), skip_unknown=True)
    priorities = values_to_ids((bug.initial_priority for bug in bugs), skip_unknown=True)
    severities = values_to_ids((bug.initial_severity for bug in bugs), skip_unknown=True)
    milestones = values_to_ids((bug.initial_milestone for bug in bugs), skip_unknown=True)

    graph = HeteroData()
    graph["bug"].x = torch.cat([text_features, normalized_numeric_features(bugs)], dim=1)
    graph["bug"].y = y
    graph["bug"].history_mask = history_mask
    graph["bug"].train_mask = train_mask
    graph["bug"].val_mask = val_mask
    graph["bug"].test_mask = test_mask
    graph["developer"].num_nodes = len(label_to_id)
    graph["product"].num_nodes = len(products)
    graph["component"].num_nodes = len(components)
    if reporters:
        graph["reporter"].num_nodes = len(reporters)

    bug_product_ids: list[int] = []
    bug_component_ids: list[int] = []
    bug_reporter_ids: list[int | None] = []
    bug_product: list[tuple[int, int]] = []
    bug_component: list[tuple[int, int]] = []
    component_product: list[tuple[int, int]] = []
    bug_reporter: list[tuple[int, int]] = []
    bug_priority: list[tuple[int, int]] = []
    bug_severity: list[tuple[int, int]] = []
    bug_milestone: list[tuple[int, int]] = []
    dev_fixed_component: list[tuple[int, int]] = []
    dev_fixed_product: list[tuple[int, int]] = []
    dev_touched_component: list[tuple[int, int]] = []
    dev_touched_product: list[tuple[int, int]] = []
    dev_cc_component: list[tuple[int, int]] = []
    dev_cc_product: list[tuple[int, int]] = []
    typed_history_edges: EdgeGroups = defaultdict(list)
    recency_edges: EdgeGroups = defaultdict(list)
    collaboration_edges: EdgeGroups = defaultdict(list)

    bug_id_to_idx = {bug.bug_id: idx for idx, bug in enumerate(bugs)}
    for bug_idx, bug in enumerate(bugs):
        product_id = products[bug.initial_product]
        comp_id = components[component_key(bug.initial_product, bug.initial_component)]
        bug_product_ids.append(product_id)
        bug_component_ids.append(comp_id)
        bug_product.append((bug_idx, product_id))
        bug_component.append((bug_idx, comp_id))
        component_product.append((comp_id, product_id))

        reporter_id = reporters.get(bug.reporter)
        bug_reporter_ids.append(reporter_id)
        if reporter_id is not None:
            bug_reporter.append((bug_idx, reporter_id))
        for mapping, value, edges in (
            (priorities, bug.initial_priority, bug_priority),
            (severities, bug.initial_severity, bug_severity),
            (milestones, bug.initial_milestone, bug_milestone),
        ):
            node_id = mapping.get(value)
            if node_id is not None:
                edges.append((bug_idx, node_id))

        if not bool(history_mask[bug_idx]):
            continue

        # Everything below is label-derived and therefore uses history-observed bugs only.
        developer_id = label_to_id[bug.assignee]
        dev_fixed_component.append((developer_id, comp_id))
        dev_fixed_product.append((developer_id, product_id))
        if config.recency_expertise:
            add_recency_expertise_edges(
                recency_edges, "fixed", developer_id, comp_id, product_id, recency_bucket(bug.fixed_at, history_end, config.recency_days)
            )
        participants: set[int] = {developer_id}

        for event in bug.history:
            observed = event.when is not None and event.when <= history_end
            actor_id = label_to_id.get(event.who)
            if actor_id is not None and observed:
                dev_touched_component.append((actor_id, comp_id))
                dev_touched_product.append((actor_id, product_id))
                if config.developer_collaboration:
                    participants.add(actor_id)
                if config.recency_expertise:
                    add_recency_expertise_edges(
                        recency_edges, "touched", actor_id, comp_id, product_id, recency_bucket(event.when, history_end, config.recency_days)
                    )
            field_name = event.what.lower()
            if field_name == "cc":
                cc_id = label_to_id.get(event.added)
                if cc_id is not None and observed:
                    dev_cc_component.append((cc_id, comp_id))
                    dev_cc_product.append((cc_id, product_id))
                    if config.developer_collaboration:
                        participants.add(cc_id)
                        if cc_id != developer_id:
                            collaboration_edges[("developer", "cc_with", "developer")].append((cc_id, developer_id))
                            collaboration_edges[("developer", "rev_cc_with", "developer")].append((developer_id, cc_id))
                    if config.recency_expertise:
                        add_recency_expertise_edges(
                            recency_edges, "cc", cc_id, comp_id, product_id, recency_bucket(event.when, history_end, config.recency_days)
                        )
            if config.developer_collaboration and field_name == "assignee" and observed:
                previous_id = label_to_id.get(event.removed)
                next_id = label_to_id.get(event.added)
                if previous_id is not None:
                    participants.add(previous_id)
                if next_id is not None:
                    participants.add(next_id)
                if previous_id is not None and next_id is not None and previous_id != next_id:
                    collaboration_edges[("developer", "reassigned_to", "developer")].append((previous_id, next_id))
                    collaboration_edges[("developer", "rev_reassigned_to", "developer")].append((next_id, previous_id))
            if config.typed_history_links and field_name in {"depends on", "blocks", "duplicate", "duplicates", "duplicate of", "see also", "related"}:
                allow_plain = field_name in {"depends on", "blocks", "duplicate", "duplicates", "duplicate of"}
                observed_at_creation = event.when is not None and event.when <= bug.created_at
                for linked_bug_id in split_bug_reference_ids(event.added, allow_plain_numbers=allow_plain):
                    linked_idx = bug_id_to_idx.get(linked_bug_id)
                    if not observed_at_creation or linked_idx is None or bugs[linked_idx].created_at >= bug.created_at:
                        continue
                    if field_name == "depends on":
                        relation = "depends_on_past_bug"
                    elif field_name == "blocks":
                        relation = "blocks_past_bug"
                    elif field_name in {"duplicate", "duplicates", "duplicate of"}:
                        relation = "duplicate_of_past_bug"
                    else:
                        relation = "related_to_past_bug"
                    typed_history_edges[("bug", relation, "bug")].append((linked_idx, bug_idx))

        if config.developer_collaboration:
            for left_id, right_id in combinations(sorted(participants), 2):
                collaboration_edges[("developer", "co_touched_with", "developer")].append((left_id, right_id))
                collaboration_edges[("developer", "rev_co_touched_with", "developer")].append((right_id, left_id))

    edge_counts: dict[str, int] = {
        "product_rev_has_product": add_edges(graph, ("product", "rev_has_product", "bug"), reversed_edges(bug_product)),
        "component_rev_has_component": add_edges(graph, ("component", "rev_has_component", "bug"), reversed_edges(bug_component)),
        "component_part_of_product": add_edges(graph, ("component", "part_of_product", "product"), component_product),
        "product_has_component": add_edges(graph, ("product", "has_component", "component"), reversed_edges(component_product)),
        "developer_fixed_component": add_edges(graph, ("developer", "fixed_component", "component"), dev_fixed_component),
        "component_rev_fixed_component": add_edges(graph, ("component", "rev_fixed_component", "developer"), reversed_edges(dev_fixed_component)),
        "developer_fixed_product": add_edges(graph, ("developer", "fixed_product", "product"), dev_fixed_product),
        "product_rev_fixed_product": add_edges(graph, ("product", "rev_fixed_product", "developer"), reversed_edges(dev_fixed_product)),
        "developer_touched_component": add_edges(graph, ("developer", "touched_component", "component"), dev_touched_component),
        "component_rev_touched_component": add_edges(graph, ("component", "rev_touched_component", "developer"), reversed_edges(dev_touched_component)),
        "developer_touched_product": add_edges(graph, ("developer", "touched_product", "product"), dev_touched_product),
        "product_rev_touched_product": add_edges(graph, ("product", "rev_touched_product", "developer"), reversed_edges(dev_touched_product)),
    }
    if reporters:
        edge_counts["reporter_rev_reported_by"] = add_edges(graph, ("reporter", "rev_reported_by", "bug"), reversed_edges(bug_reporter))
    if dev_cc_component:
        edge_counts["developer_cc_component"] = add_edges(graph, ("developer", "cc_component", "component"), dev_cc_component)
        edge_counts["component_rev_cc_component"] = add_edges(graph, ("component", "rev_cc_component", "developer"), reversed_edges(dev_cc_component))
        edge_counts["developer_cc_product"] = add_edges(graph, ("developer", "cc_product", "product"), dev_cc_product)
        edge_counts["product_rev_cc_product"] = add_edges(graph, ("product", "rev_cc_product", "developer"), reversed_edges(dev_cc_product))
    for node_type, mapping, edges in (("priority", priorities, bug_priority), ("severity", severities, bug_severity), ("milestone", milestones, bug_milestone)):
        if mapping:
            graph[node_type].num_nodes = len(mapping)
            edge_counts[f"{node_type}_rev_has_{node_type}"] = add_edges(graph, (node_type, f"rev_has_{node_type}", "bug"), reversed_edges(edges))

    if config.recency_expertise:
        add_edge_groups(graph, edge_counts, recency_edges)
    if config.developer_collaboration:
        add_edge_groups(graph, edge_counts, collaboration_edges)
    if config.shortcut_candidates:
        add_edge_groups(
            graph,
            edge_counts,
            build_developer_candidate_edges(bugs, history_mask, label_to_id, bug_component_ids, bug_product_ids, bug_reporter_ids, config.candidate_top_k),
        )
    if config.typed_history_links:
        add_edge_groups(graph, edge_counts, typed_history_edges)

    medium_threshold, high_threshold, threshold_fit_edges = float("nan"), float("nan"), 0
    if config.similarity_top_k > 0:
        timestamps = torch.tensor([bug.created_at.timestamp() for bug in bugs], dtype=torch.float64)
        similarity = temporal_similarity_edges(text_features, timestamps, config.similarity_top_k, config.similarity_batch_size, device)
        if config.relation_aware_similarity:
            medium_threshold, high_threshold, threshold_fit_edges = calibrated_similarity_thresholds(
                similarity, history_mask, config.similarity_medium_quantile, config.similarity_high_quantile
            )
            similarity_groups: EdgeGroups = defaultdict(list)
            for past_idx, current_idx, score in similarity:
                bucket = similarity_score_bucket(score, medium_threshold, high_threshold)
                context = "same_component" if bug_component_ids[past_idx] == bug_component_ids[current_idx] else "cross_component"
                similarity_groups[("bug", f"{context}_similar_{bucket}_past_to_bug", "bug")].append((past_idx, current_idx))
            add_edge_groups(graph, edge_counts, similarity_groups)
        else:
            edge_counts["bug_similar_past_to_bug"] = add_edges(graph, ("bug", "similar_past_to_bug", "bug"), [(s, d) for s, d, _ in similarity])
        if config.shortcut_candidates:
            similar_candidates: EdgeGroups = defaultdict(list)
            for past_idx, current_idx, _score in similarity:
                if bool(history_mask[past_idx]):
                    developer_id = label_to_id.get(bugs[past_idx].assignee)
                    if developer_id is not None:
                        similar_candidates[("developer", "candidate_by_similar_bug", "bug")].append((developer_id, current_idx))
            add_edge_groups(graph, edge_counts, similar_candidates)

    add_node_ids(graph)
    graph.bug_ids = [bug.bug_id for bug in bugs]
    graph["bug"].product_id = torch.tensor(bug_product_ids, dtype=torch.long)
    graph["bug"].component_id = torch.tensor(bug_component_ids, dtype=torch.long)
    graph["bug"].reporter_id = torch.tensor([value if value is not None else -1 for value in bug_reporter_ids], dtype=torch.long)
    graph.label_to_id = label_to_id
    graph.id_to_label = {str(idx): label for label, idx in label_to_id.items()}
    graph.text_encoder = encoder_name
    graph.graph_config = asdict(config)
    graph.similarity_config = {
        "top_k": config.similarity_top_k,
        "source": encoder_name,
        "edge_direction": "past_bug_to_current_bug",
        "relation_aware": config.relation_aware_similarity and config.similarity_top_k > 0,
        "medium_threshold": medium_threshold,
        "high_threshold": high_threshold,
        "threshold_fit_history_edges": threshold_fit_edges,
    }
    graph.edge_counts = edge_counts
    graph.split_counts = {
        "history": int(history_mask.sum().item()),
        "train": int(train_mask.sum().item()),
        "val": int(val_mask.sum().item()),
        "test": int(test_mask.sum().item()),
    }
    graph.split_boundaries = boundaries
    graph.candidate_diagnostics = candidate_edge_diagnostics(graph, y, {"train": train_mask, "val": val_mask, "test": test_mask})
    return graph


def graph_summary(graph: HeteroData, raw_stats: dict[str, int], encoder: TextEncoderConfig) -> dict[str, object]:
    return {
        "text_encoder": graph.text_encoder,
        "encoder_config": asdict(encoder),
        "graph_config": graph.graph_config,
        "node_counts": {node_type: graph[node_type].num_nodes for node_type in graph.node_types},
        "edge_counts": graph.edge_counts,
        "split_counts": graph.split_counts,
        "split_boundaries": graph.split_boundaries,
        "num_classes": len(graph.label_to_id),
        "bug_feature_dim": graph["bug"].x.size(-1),
        "similarity_config": graph.similarity_config,
        "candidate_diagnostics": graph.candidate_diagnostics,
        "raw_stats": raw_stats,
    }


def build_graph(
    project_dir: Path,
    encoder: TextEncoderConfig,
    config: GraphConfig,
    device: torch.device,
    output_stem: Path,
) -> HeteroData:
    """Load the export, split it, embed the text, assemble the graph and write every output."""
    bugs, raw_stats = load_raw_bugs(project_dir)
    if not bugs:
        raise ValueError(f"No bugs loaded from {project_dir}")
    bugs.sort(key=lambda bug: (bug.created_at, bug_sort_key(bug.bug_id)))
    bugs, label_to_id, masks, boundaries, filter_stats = filter_to_causally_observed_labels(
        bugs, config.history_ratio, config.train_ratio, config.val_ratio,
        config.min_history_dev_bugs, config.min_train_dev_bugs, config.max_developers,
    )
    raw_stats = {**raw_stats, **filter_stats}
    text_features = encode_texts([bug.initial_text for bug in bugs], encoder, device)
    graph = assemble_graph(bugs, label_to_id, masks, boundaries, text_features, encoder.name, config, device)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, output_stem.with_suffix(".pt"))
    write_json(output_stem.parent / f"{output_stem.name}_labels.json", label_to_id)
    write_json(output_stem.parent / f"{output_stem.name}_summary.json", graph_summary(graph, raw_stats, encoder))
    write_split_csv(output_stem.parent / f"{output_stem.name}_splits.csv", bugs, masks)
    print(f"Saved graph {output_stem.with_suffix('.pt')}: {graph.split_counts}, {len(label_to_id)} classes")
    return graph

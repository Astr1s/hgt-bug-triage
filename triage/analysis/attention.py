"""What the trained HGT and HAN arms attend to, read from saved checkpoints."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax

from ..models import HANBugAssigner, HGTBugAssigner
from ..util import write_json

EdgeType = tuple[str, str, str]


def load_checkpoint(arm: str, data: HeteroData, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("training_config") or payload["model_args"]
    if payload["num_classes"] != int(data["bug"].y.max().item()) + 1:
        raise ValueError(f"{checkpoint} was trained on {payload['num_classes']} classes; the graph has a different label space")
    cls = HGTBugAssigner if arm == "hgt" else HANBugAssigner
    model = cls.from_data(
        data, num_classes=payload["num_classes"], hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"], heads=config["heads"], dropout=config["dropout"],
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()


def _alpha_hook(module: torch.nn.Module, msg_kwargs: tuple, out: torch.Tensor) -> None:
    """Recompute HGTConv's per-edge attention from its message arguments (PyG 2.7 arithmetic)."""
    kw = msg_kwargs[0]
    alpha = (kw["q_i"] * kw["k_j"]).sum(dim=-1) * kw["edge_attr"]
    alpha = alpha / math.sqrt(kw["q_i"].size(-1))
    module._alpha = softmax(alpha, kw["index"], kw["ptr"], kw["size_i"]).detach()


def edge_spans(data: HeteroData) -> list[tuple[int, int, EdgeType]]:
    """Positions of each relation in HGTConv's concatenated edge tensor."""
    spans, start = [], 0
    for edge_type, edge_index in data.edge_index_dict.items():
        spans.append((start, start + edge_index.size(1), edge_type))
        start += edge_index.size(1)
    return spans


def _advance(model: torch.nn.Module, layer_idx: int, x_dict: dict, out_dict: dict) -> dict:
    """One residual + layer-norm step, exactly as ``HGTEncoder.encode`` in eval mode."""
    next_x = {}
    for node_type, x in x_dict.items():
        message = out_dict.get(node_type)
        updated = x if message is None else x + torch.nn.functional.gelu(message)
        next_x[node_type] = model.layer_norms[layer_idx][node_type](updated)
    return next_x


def hgt_relation_shares(model: torch.nn.Module, data: HeteroData, mask: torch.Tensor) -> list[dict[EdgeType, float]]:
    """Per layer: share of the masked bugs' attention received by each relation type."""
    spans = edge_spans(data)
    x_dict = model._features(data)
    assert list(x_dict.keys())[0] == "bug", "bug must be the first node type for destination offsets"
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(module: torch.nn.Module, inputs: tuple) -> None:
        captured["edge_index"] = inputs[0]

    results = []
    for layer_idx, conv in enumerate(model.convs):
        handles = (conv.register_propagate_forward_pre_hook(pre_hook), conv.register_message_forward_hook(_alpha_hook))
        out_dict = conv(x_dict, data.edge_index_dict)
        for handle in handles:
            handle.remove()
        alpha = conv._alpha.mean(dim=1)
        dst = captured["edge_index"][1]
        in_mask = (dst < mask.numel()) & mask[dst.clamp(max=mask.numel() - 1)]
        shares = {edge_type: float(alpha[a:b][in_mask[a:b]].sum().item()) for a, b, edge_type in spans}
        total = sum(shares.values())
        results.append({edge_type: value / total for edge_type, value in shares.items()})
        x_dict = _advance(model, layer_idx, x_dict, out_dict)
    return results


def han_semantic_attention(model: torch.nn.Module, data: HeteroData) -> list[dict[EdgeType, float]]:
    """Per layer: HANConv's semantic attention over the relations that target bugs."""
    x_dict = model._features(data)
    into_bug = [edge_type for edge_type in data.edge_index_dict if edge_type[2] == "bug"]
    results = []
    for layer_idx, conv in enumerate(model.convs):
        out_dict, semantic = conv(x_dict, data.edge_index_dict, return_semantic_attention_weights=True)
        results.append(dict(zip(into_bug, semantic["bug"].detach().flatten().tolist())))
        x_dict = _advance(model, layer_idx, x_dict, out_dict)
    return results


def _test_top1(model: torch.nn.Module, data: HeteroData, mask: torch.Tensor) -> float:
    with torch.no_grad():
        return float((model(data).argmax(1)[mask] == data["bug"].y[mask]).float().mean().item())


def _drop(data: HeteroData, predicate: Callable[[EdgeType], bool]) -> HeteroData:
    clone = data.clone()
    for edge_type in list(clone.edge_index_dict):
        if predicate(edge_type):
            clone[edge_type].edge_index = clone[edge_type].edge_index[:, :0]
    return clone


DELETION_FAMILIES: dict[str, Callable[[EdgeType], bool]] = {
    "candidates": lambda et: "candidate" in et[1],
    "similarity": lambda et: "similar_" in et[1] and et[0] == "bug",
    "reporter": lambda et: et[1] == "rev_reported_by",
    "component_product": lambda et: et[1] in ("rev_has_component", "rev_has_product"),
    "all_expertise": lambda et: et[0] == "developer" and et[2] in ("component", "product"),
}


def relation_key(edge_type: EdgeType) -> str:
    return "/".join(edge_type)


def inspect_attention(graph_path: Path, checkpoints: dict[str, Path], device: torch.device, output_json: Path) -> dict[str, object]:
    """``checkpoints`` maps ``"hgt"`` and ``"han"`` to one checkpoint each (the same seed)."""
    data = torch.load(graph_path, map_location="cpu", weights_only=False).to(device)
    test_mask = data["bug"].test_mask.bool()
    y = data["bug"].y
    report: dict[str, object] = {"relations_into_bug": [relation_key(et) for et in data.edge_index_dict if et[2] == "bug"], "arms": {}}
    with torch.no_grad():
        for arm, checkpoint in checkpoints.items():
            model = load_checkpoint(arm, data, checkpoint, device)
            top1 = float((model(data).argmax(1)[test_mask] == y[test_mask]).float().mean().item())
            if arm == "hgt":
                layers = hgt_relation_shares(model, data, test_mask)
            else:
                layers = han_semantic_attention(model, data)
            report["arms"][arm] = {
                "checkpoint": str(checkpoint),
                "test_top1": top1,
                "attention_by_layer": [{relation_key(et): value for et, value in layer.items()} for layer in layers],
            }
            print(f"{arm}: test top-1 {top1:.4f}; {len(layers)} layers of attention read")
    write_json(output_json, report)
    return report


def inference_deletion(graph_path: Path, checkpoints: dict[str, dict[int, Path]], device: torch.device, output_json: Path) -> dict[str, object]:
    """``checkpoints[arm][seed]`` = checkpoint path. Reports test top-1 change per deleted family."""
    data = torch.load(graph_path, map_location="cpu", weights_only=False).to(device)
    test_mask = data["bug"].test_mask.bool()
    report: dict[str, object] = {"families": list(DELETION_FAMILIES), "results": []}
    for arm, per_seed in checkpoints.items():
        for seed, checkpoint in sorted(per_seed.items()):
            model = load_checkpoint(arm, data, checkpoint, device)
            base = _test_top1(model, data, test_mask)
            deltas = {name: _test_top1(model, _drop(data, predicate), test_mask) - base for name, predicate in DELETION_FAMILIES.items()}
            report["results"].append({"arm": arm, "seed": seed, "full_test_top1": base, "delta_test_top1": deltas})
            print(f"{arm} seed {seed}: full {base:.4f} | " + " | ".join(f"{k} {v:+.4f}" for k, v in deltas.items()))
    write_json(output_json, report)
    return report

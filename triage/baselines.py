"""Classical baselines evaluated on exactly the graph's splits."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .metrics import bootstrap_ranking_intervals, ranking_metrics
from .raw import clean_value, index_files, load_buglist, parse_bug_text
from .util import seed_everything, write_json


def smoothed_log_distribution(counts: Counter[int], num_classes: int, alpha: float) -> np.ndarray:
    values = np.full(num_classes, alpha, dtype=np.float64)
    for label, count in counts.items():
        values[label] += count
    return np.log(values / values.sum())


def frequency_scores(values: np.ndarray | None, labels: np.ndarray, train_indices: np.ndarray, num_classes: int, alpha: float) -> np.ndarray:
    """Smoothed label frequencies, globally or conditioned on one categorical value."""
    global_scores = smoothed_log_distribution(Counter(int(labels[idx]) for idx in train_indices), num_classes, alpha)
    scores = np.repeat(global_scores[None, :], len(labels), axis=0)
    if values is None:
        return scores
    grouped: defaultdict[int, Counter[int]] = defaultdict(Counter)
    for idx in train_indices:
        if int(values[idx]) >= 0:
            grouped[int(values[idx])][int(labels[idx])] += 1
    for idx, value in enumerate(values):
        if int(value) in grouped:
            scores[idx] = smoothed_log_distribution(grouped[int(value)], num_classes, alpha)
    return scores


def categorical_naive_bayes_scores(
    feature_values: list[np.ndarray], labels: np.ndarray, train_indices: np.ndarray, num_classes: int, alpha: float, prior: str
) -> np.ndarray:
    """``prior="empirical"`` uses smoothed training class frequencies; ``"uniform"`` uses -log K."""
    if prior == "empirical":
        class_counts = np.bincount(labels[train_indices], minlength=num_classes).astype(np.float64)
        log_prior = np.log((class_counts + alpha) / (class_counts.sum() + alpha * num_classes))
    elif prior == "uniform":
        log_prior = np.full(num_classes, -np.log(num_classes), dtype=np.float64)
    else:
        raise ValueError(f"Unknown Naive Bayes prior: {prior!r}")
    scores = np.repeat(log_prior[None, :], len(labels), axis=0)
    for values in feature_values:
        cardinality = max(int(values.max(initial=-1)) + 1, 1)
        counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        observed_per_class = np.zeros(num_classes, dtype=np.float64)
        for idx in train_indices:
            value = int(values[idx])
            if value < 0:
                continue
            counts[(int(labels[idx]), value)] += 1
            observed_per_class[int(labels[idx])] += 1
        for idx, value in enumerate(values):
            value = int(value)
            if value < 0:
                continue
            numerators = np.array([counts[(label, value)] + alpha for label in range(num_classes)], dtype=np.float64)
            scores[idx] += np.log(numerators / (observed_per_class + alpha * cardinality))
    return scores


def linear_scores(train_x: object, all_x: object, labels: np.ndarray, train_indices: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.linear_model import SGDClassifier

    classifier = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-5, class_weight="balanced", max_iter=2000, tol=1e-4, random_state=seed, n_jobs=-1
    )
    classifier.fit(train_x, labels[train_indices])
    raw_scores = classifier.decision_function(all_x)
    if raw_scores.ndim == 1:
        raw_scores = np.column_stack([-raw_scores, raw_scores])
    return np.asarray(raw_scores, dtype=np.float64)


def evaluate_scores(scores: np.ndarray, labels: np.ndarray, masks: dict[str, np.ndarray], bootstrap_samples: int, seed: int) -> dict[str, float]:
    results: dict[str, float] = {}
    score_tensor = torch.from_numpy(scores).float()
    label_tensor = torch.from_numpy(labels).long()
    for split_name, mask in masks.items():
        split_scores, split_labels = score_tensor[mask], label_tensor[mask]
        metrics = ranking_metrics(split_scores, split_labels)
        if split_name == "test":
            metrics.update(bootstrap_ranking_intervals(split_scores, split_labels, bootstrap_samples, seed))
        results.update({f"{split_name}_{key}": value for key, value in metrics.items()})
    return results


def load_initial_texts(graph: HeteroData, project_dir: Path) -> list[str]:
    buglist = load_buglist(project_dir)
    text_paths = index_files(project_dir / "Bugs", ".txt")
    texts: list[str] = []
    for bug_id in graph.bug_ids:
        summary = clean_value(buglist.get(str(bug_id), {}).get("Summary", ""), "")
        path = text_paths.get(str(bug_id))
        texts.append(parse_bug_text(path, summary).text if path is not None else summary)
    return texts


def evaluate_classical_baselines(
    graph_path: Path,
    project_dir: Path,
    output_json: Path,
    seed: int = 42,
    bootstrap_samples: int = 1000,
    alpha: float = 1.0,
    include_tfidf: bool = True,
    tfidf_max_features: int = 100_000,
    tfidf_min_df: int = 2,
) -> dict[str, object]:
    seed_everything(seed)
    graph: HeteroData = torch.load(graph_path, map_location="cpu", weights_only=False)
    labels = graph["bug"].y.numpy()
    train_indices = np.flatnonzero(graph["bug"].train_mask.numpy().astype(bool))
    masks = {"val": graph["bug"].val_mask.numpy().astype(bool), "test": graph["bug"].test_mask.numpy().astype(bool)}
    num_classes = int(labels.max()) + 1
    product = graph["bug"].product_id.numpy()
    component = graph["bug"].component_id.numpy()
    reporter = graph["bug"].reporter_id.numpy()
    features = graph["bug"].x.numpy()

    builders: dict[str, Callable[[], np.ndarray]] = {
        "majority": lambda: frequency_scores(None, labels, train_indices, num_classes, alpha),
        "product_frequency": lambda: frequency_scores(product, labels, train_indices, num_classes, alpha),
        "component_frequency": lambda: frequency_scores(component, labels, train_indices, num_classes, alpha),
        "reporter_frequency": lambda: frequency_scores(reporter, labels, train_indices, num_classes, alpha),
        "metadata_naive_bayes": lambda: categorical_naive_bayes_scores([product, component, reporter], labels, train_indices, num_classes, alpha, "empirical"),
        "metadata_naive_bayes_uniform": lambda: categorical_naive_bayes_scores([product, component, reporter], labels, train_indices, num_classes, alpha, "uniform"),
        "graph_embedding_linear": lambda: linear_scores(features[train_indices], features, labels, train_indices, seed),
    }
    results: dict[str, object] = {
        "graph_path": str(Path(graph_path).resolve()),
        "seed": seed,
        "num_classes": num_classes,
        "split_counts": {name: int(mask.sum()) for name, mask in masks.items()} | {"train": len(train_indices)},
        "baselines": {},
    }
    for name, build in builders.items():
        print(f"Evaluating baseline: {name}", flush=True)
        results["baselines"][name] = evaluate_scores(build(), labels, masks, bootstrap_samples, seed + 10_000)

    if include_tfidf:
        from sklearn.feature_extraction.text import TfidfVectorizer

        print("Loading report-time text for the TF-IDF baseline", flush=True)
        texts = load_initial_texts(graph, project_dir)
        vectorizer = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=tfidf_min_df, max_features=tfidf_max_features, sublinear_tf=True
        )
        train_x = vectorizer.fit_transform([texts[idx] for idx in train_indices])
        results["tfidf_vocabulary_size"] = len(vectorizer.vocabulary_)
        results["baselines"]["tfidf_sgd"] = evaluate_scores(
            linear_scores(train_x, vectorizer.transform(texts), labels, train_indices, seed), labels, masks, bootstrap_samples, seed + 20_000
        )

    write_json(output_json, results)
    print(f"Saved baseline metrics: {Path(output_json).resolve()}")
    return results

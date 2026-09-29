"""Ranking metrics, bootstrap intervals and the exact paired sign-flip test."""
from __future__ import annotations

import itertools
import statistics

import torch


def ranking_metrics(logits: torch.Tensor, y: torch.Tensor, top_ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    """Top-k, class-balanced top-k, macro precision/recall/F1, MRR and NDCG at rank 1 hit."""
    if y.numel() == 0:
        empty = {f"top{k}": 0.0 for k in top_ks} | {f"balanced_top{k}": 0.0 for k in top_ks}
        return empty | {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0, "mrr": 0.0, "ndcg": 0.0}

    max_k = min(max(top_ks), logits.size(1))
    top = logits.topk(max_k, dim=1).indices
    metrics: dict[str, float] = {}
    labels = y.unique(sorted=True)
    for k in top_ks:
        actual_k = min(k, logits.size(1))
        correct = (top[:, :actual_k] == y.view(-1, 1)).any(dim=1)
        metrics[f"top{k}"] = float(correct.float().mean().item())
        per_class = [correct[y == label].float().mean() for label in labels]
        metrics[f"balanced_top{k}"] = float(torch.stack(per_class).mean().item())

    predictions = top[:, 0]
    precision_list, recall_list, f1_list = [], [], []
    for label in labels:
        true_positive = ((predictions == label) & (y == label)).sum().float()
        precision = true_positive / (predictions == label).sum().float().clamp_min(1.0)
        recall = true_positive / (y == label).sum().float().clamp_min(1.0)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(2.0 * precision * recall / (precision + recall).clamp_min(1e-12))
    metrics["macro_precision"] = float(torch.stack(precision_list).mean().item())
    metrics["macro_recall"] = float(torch.stack(recall_list).mean().item())
    metrics["macro_f1"] = float(torch.stack(f1_list).mean().item())

    true_scores = logits.gather(1, y.view(-1, 1))
    ranks = (logits > true_scores).sum(dim=1).float() + 1.0
    metrics["mrr"] = float((1.0 / ranks).mean().item())
    metrics["ndcg"] = float((1.0 / torch.log2(ranks + 1.0)).mean().item())
    return metrics


def bootstrap_ranking_intervals(logits: torch.Tensor, y: torch.Tensor, samples: int, seed: int) -> dict[str, float]:
    """95% percentile bootstrap intervals for top-1, top-3 and MRR over examples."""
    if samples <= 0 or y.numel() == 0:
        return {}
    max_k = min(3, logits.size(1))
    top = logits.topk(max_k, dim=1).indices
    top1 = (top[:, 0] == y).float().cpu()
    top3 = (top == y.view(-1, 1)).any(dim=1).float().cpu()
    true_scores = logits.gather(1, y.view(-1, 1))
    reciprocal_rank = (1.0 / ((logits > true_scores).sum(dim=1).float() + 1.0)).cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(y.numel(), (samples, y.numel()), generator=generator)
    result: dict[str, float] = {}
    for name, values in {"top1": top1, "top3": top3, "mrr": reciprocal_rank}.items():
        estimates = values[indices].mean(dim=1)
        result[f"{name}_ci95_low"] = float(torch.quantile(estimates, 0.025).item())
        result[f"{name}_ci95_high"] = float(torch.quantile(estimates, 0.975).item())
    return result


def mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def sign_flip_p(values: list[float]) -> float:
    """Exact two-sided sign-flip p-value for the mean of paired differences."""
    if len(values) < 2:
        return 1.0
    observed = abs(statistics.fmean(values))
    extreme = sum(
        1
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        if abs(statistics.fmean(sign * value for sign, value in zip(signs, values))) >= observed - 1e-15
    )
    return extreme / 2 ** len(values)


def paired_delta_statistics(values: list[float]) -> dict[str, float | int]:
    stats = mean_std(values)
    std = float(stats["std"])
    stats["effect_size_dz"] = float(stats["mean"]) / std if std > 0.0 else 0.0
    stats["sign_flip_p_two_sided"] = sign_flip_p(values)
    return stats

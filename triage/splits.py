"""Chronological history / train / validation / test periods with label censoring."""
from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

from .raw import RawBug, bug_sort_key

Masks = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def label_observed_by(bug: RawBug, cutoff: datetime) -> bool:
    return bug.fixed_at is not None and bug.fixed_at <= cutoff


def build_causal_splits(
    bugs: list[RawBug],
    history_ratio: float,
    train_ratio: float,
    val_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, str], dict[str, int]]:
    if not bugs:
        raise ValueError("Cannot split an empty bug list")
    if history_ratio <= 0.0 or train_ratio <= 0.0 or val_ratio <= 0.0:
        raise ValueError("history, train, and validation ratios must be positive")
    if history_ratio + train_ratio + val_ratio >= 1.0:
        raise ValueError("history + train + validation ratios must be less than 1")

    order = sorted(range(len(bugs)), key=lambda idx: (bugs[idx].created_at, bug_sort_key(bugs[idx].bug_id)))
    history_end = max(1, int(len(order) * history_ratio))
    train_end = max(history_end + 1, int(len(order) * (history_ratio + train_ratio)))
    val_end = max(train_end + 1, int(len(order) * (history_ratio + train_ratio + val_ratio)))
    train_end = min(train_end, len(order))
    val_end = min(val_end, len(order))

    history_cutoff = bugs[order[history_end - 1]].created_at
    train_cutoff = bugs[order[train_end - 1]].created_at
    history_mask = torch.zeros(len(bugs), dtype=torch.bool)
    train_mask = torch.zeros(len(bugs), dtype=torch.bool)
    val_mask = torch.zeros(len(bugs), dtype=torch.bool)
    test_mask = torch.zeros(len(bugs), dtype=torch.bool)

    for idx in order[:history_end]:
        history_mask[idx] = label_observed_by(bugs[idx], history_cutoff)
    for idx in order[history_end:train_end]:
        train_mask[idx] = label_observed_by(bugs[idx], train_cutoff)
    val_mask[order[train_end:val_end]] = True
    test_mask[order[val_end:]] = True

    boundaries = {
        "min_created_at": bugs[order[0]].created_at.isoformat(),
        "history_end_created_at": history_cutoff.isoformat(),
        "train_end_created_at": train_cutoff.isoformat(),
        "val_end_created_at": bugs[order[val_end - 1]].created_at.isoformat() if val_end > train_end else "",
        "max_created_at": bugs[order[-1]].created_at.isoformat(),
    }
    observed_history = int(history_mask.sum().item())
    observed_train = int(train_mask.sum().item())
    stats = {
        "history_period_rows": history_end,
        "history_observed_rows": observed_history,
        "history_censored_rows": history_end - observed_history,
        "train_period_rows": train_end - history_end,
        "train_observed_rows": observed_train,
        "train_censored_rows": (train_end - history_end) - observed_train,
    }
    return history_mask, train_mask, val_mask, test_mask, boundaries, stats


def observed_label_map(
    bugs: list[RawBug],
    history_mask: torch.Tensor,
    train_mask: torch.Tensor,
    min_history_dev_bugs: int,
    min_train_dev_bugs: int,
    max_developers: int,
) -> dict[str, int]:
    """Developers with enough observed history and training labels, most active first."""
    history_counts = Counter(bug.assignee for idx, bug in enumerate(bugs) if bool(history_mask[idx]))
    train_counts = Counter(bug.assignee for idx, bug in enumerate(bugs) if bool(train_mask[idx]))
    developers = [
        developer
        for developer, count in sorted(history_counts.items(), key=lambda item: (-item[1], item[0].lower()))
        if count >= min_history_dev_bugs and train_counts[developer] >= min_train_dev_bugs
    ]
    if max_developers > 0:
        developers = developers[:max_developers]
    return {developer: idx for idx, developer in enumerate(developers)}


def filter_to_causally_observed_labels(
    bugs: list[RawBug],
    history_ratio: float,
    train_ratio: float,
    val_ratio: float,
    min_history_dev_bugs: int,
    min_train_dev_bugs: int,
    max_developers: int,
) -> tuple[list[RawBug], dict[str, int], Masks, dict[str, str], dict[str, int]]:
    """Iterate the eligibility rule until the developer set is stable."""
    original_bug_count = len(bugs)
    original_developer_count = len({bug.assignee for bug in bugs})
    current = list(bugs)
    for _ in range(9):
        history_mask, train_mask, val_mask, test_mask, boundaries, split_stats = build_causal_splits(
            current, history_ratio, train_ratio, val_ratio
        )
        label_to_id = observed_label_map(
            current, history_mask, train_mask, min_history_dev_bugs, min_train_dev_bugs, max_developers
        )
        if not label_to_id:
            raise ValueError("No assignee satisfies the causal history/train thresholds.")
        filtered = [bug for bug in current if bug.assignee in label_to_id]
        if len(filtered) == len(current):
            stats = {
                **split_stats,
                "removed_cold_label_bugs": original_bug_count - len(filtered),
                "removed_cold_labels": original_developer_count - len(label_to_id),
            }
            return filtered, label_to_id, (history_mask, train_mask, val_mask, test_mask), boundaries, stats
        current = filtered
    raise RuntimeError("Causal label filtering did not converge")


def write_split_csv(path: Path, bugs: list[RawBug], masks: Masks) -> None:
    history_mask, train_mask, val_mask, test_mask = masks
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["bug_id", "created_at_utc", "assignee", "split", "product", "component", "reporter"]
        )
        writer.writeheader()
        for idx, bug in enumerate(bugs):
            split = (
                "history" if history_mask[idx] else "train" if train_mask[idx]
                else "val" if val_mask[idx] else "test" if test_mask[idx] else "censored"
            )
            writer.writerow(
                {
                    "bug_id": bug.bug_id,
                    "created_at_utc": bug.created_at.isoformat(),
                    "assignee": bug.assignee,
                    "split": split,
                    "product": bug.initial_product,
                    "component": bug.initial_component,
                    "reporter": bug.reporter,
                }
            )

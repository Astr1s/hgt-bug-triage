"""Label drift between periods and reporter self-assignment."""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from ..util import read_json, write_json


def normalise(text: str) -> str:
    return re.sub(r"[^a-z]+", " ", text.lower()).strip()


def local_part(address: str) -> str:
    return normalise(address.split("@")[0])


def drift_and_reporter_stats(graph_path: Path, splits_csv: Path, labels_json: Path, output_json: Path) -> dict[str, object]:
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    y = graph["bug"].y.numpy()
    train = graph["bug"].train_mask.numpy().astype(bool)
    test = graph["bug"].test_mask.numpy().astype(bool)
    num_classes = int(y.max()) + 1
    train_counts = np.bincount(y[train], minlength=num_classes)
    test_counts = np.bincount(y[test], minlength=num_classes)
    by_train = np.argsort(-train_counts)
    by_test = np.argsort(-test_counts)
    rho, p = spearmanr(train_counts, test_counts)
    drift = {
        "training_reports": int(train.sum()),
        "test_reports": int(test.sum()),
        "top1_training_developer_share_of_training": float(train_counts[by_train[0]] / train.sum()),
        "top1_training_developer_share_of_test": float(test_counts[by_train[0]] / test.sum()),
        "top10_training_developers_share_of_training": float(train_counts[by_train[:10]].sum() / train.sum()),
        "top10_training_developers_share_of_test": float(test_counts[by_train[:10]].sum() / test.sum()),
        "top10_overlap_between_periods": len(set(by_train[:10].tolist()) & set(by_test[:10].tolist())),
        "developers_with_no_test_report": int((test_counts == 0).sum()),
        "developers": num_classes,
        "spearman_rho_train_vs_test_counts": float(rho),
        "spearman_p": float(p),
    }

    with Path(splits_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    developer_names = {local_part(address) for address in read_json(labels_json)}
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        reporter = normalise(row["reporter"])
        totals[row["split"]][0] += 1
        if reporter and reporter == local_part(row["assignee"]):
            totals[row["split"]][1] += 1
        if reporter in developer_names:
            totals[row["split"]][2] += 1
    reporter = {
        split: {"reports": n, "reporter_is_assignee": self_assigned / n, "reporter_is_eligible_developer": is_developer / n}
        for split, (n, self_assigned, is_developer) in totals.items()
    }
    report = {"label_drift": drift, "reporter_by_split": reporter}
    write_json(output_json, report)
    print(f"Top-10 training developers hold {drift['top10_training_developers_share_of_training']:.3f} of training and {drift['top10_training_developers_share_of_test']:.3f} of test reports")
    return report

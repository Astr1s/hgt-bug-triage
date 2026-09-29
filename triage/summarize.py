"""Aggregate the per-seed metrics files in one run directory."""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from .metrics import mean_std, paired_delta_statistics
from .util import read_json, write_json

RUN_RE = re.compile(r"^(?P<model>.+)_seed(?P<seed>\d+)_metrics\.json$")
SUMMARY_METRICS = (
    "test_top1", "test_top3", "test_top5", "test_top10",
    "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_mrr", "test_ndcg",
)


def load_runs(directory: Path) -> dict[str, dict[int, dict]]:
    """``{run_name: {seed: metrics}}`` for every ``*_seed<n>_metrics.json`` in the directory."""
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted(Path(directory).glob("*_seed*_metrics.json")):
        match = RUN_RE.match(path.name)
        if match:
            runs[match.group("model")][int(match.group("seed"))] = read_json(path)
    return dict(runs)


def summarize_run_directory(directory: Path) -> dict[str, object]:
    """Write ``experiment_summary.json`` and ``.csv``: per-run mean/std and paired ablation deltas."""
    runs = load_runs(directory)
    summary: dict[str, object] = {"models": {}, "paired_ablation_deltas": {}}
    rows: list[dict[str, object]] = []
    for model_name, seeded_runs in sorted(runs.items()):
        model_summary = {}
        for metric in SUMMARY_METRICS:
            values = [float(run[metric]) for run in seeded_runs.values() if metric in run]
            if values:
                model_summary[metric] = mean_std(values)
                rows.append({"model": model_name, "metric": metric, **model_summary[metric]})
        summary["models"][model_name] = {"seeds": sorted(seeded_runs), "metrics": model_summary}

    for arch in ("hgt", "han"):
        reference = runs.get(f"{arch}_research_full", {})
        for model_name, seeded_runs in sorted(runs.items()):
            if not model_name.startswith(f"{arch}_ablation_no_"):
                continue
            common = sorted(set(reference) & set(seeded_runs))
            deltas = {
                metric: paired_delta_statistics([float(seeded_runs[s][metric]) - float(reference[s][metric]) for s in common])
                for metric in SUMMARY_METRICS
                if common and all(metric in seeded_runs[s] and metric in reference[s] for s in common)
            }
            summary["paired_ablation_deltas"][model_name] = {"reference": f"{arch}_research_full", "seeds": common, "metrics": deltas}

    write_json(Path(directory) / "experiment_summary.json", summary)
    with (Path(directory) / "experiment_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "n", "mean", "std"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summarised {len(runs)} runs in {directory}")
    return summary

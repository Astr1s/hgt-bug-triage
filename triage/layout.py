"""Where every artifact lives, relative to one artifacts directory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    artifacts: Path

    # graphs/<graph>.pt, _labels.json, _summary.json, _splits.csv
    def graph_stem(self, graph: str) -> Path:
        return self.artifacts / "graphs" / graph

    def graph_path(self, graph: str) -> Path:
        return self.graph_stem(graph).with_suffix(".pt")

    def labels_path(self, graph: str) -> Path:
        return self.artifacts / "graphs" / f"{graph}_labels.json"

    def summary_path(self, graph: str) -> Path:
        return self.artifacts / "graphs" / f"{graph}_summary.json"

    def splits_path(self, graph: str) -> Path:
        return self.artifacts / "graphs" / f"{graph}_splits.csv"

    # runs/<graph>/<run>_seed<seed>_{model.pt,metrics,history,run}.json
    def run_dir(self, graph: str) -> Path:
        return self.artifacts / "runs" / graph

    def run_stem(self, graph: str, run: str, seed: int) -> Path:
        return self.run_dir(graph) / f"{run}_seed{seed}"

    def metrics_path(self, graph: str, run: str, seed: int) -> Path:
        return self.run_dir(graph) / f"{run}_seed{seed}_metrics.json"

    def baselines_path(self, graph: str, seed: int) -> Path:
        return self.run_dir(graph) / f"classical_baselines_seed{seed}.json"

    # analysis/*.json
    @property
    def analysis_dir(self) -> Path:
        return self.artifacts / "analysis"

    @property
    def results_path(self) -> Path:
        return self.artifacts / "results.md"

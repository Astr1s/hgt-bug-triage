#!/usr/bin/env python3
"""Run the experiment pipeline end to end, or one stage of it."""
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from triage import experiments
from triage.layout import Layout
from triage.util import resolve_device

ROOT = Path(__file__).resolve().parent


class Tee:
    """Duplicate stdout into a per-run log file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.handle.write(text)

    def flush(self) -> None:
        self.stdout.flush()
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def parse_graphs(value: str) -> list[str]:
    names = list(experiments.GRAPHS) if value == "all" else [v.strip() for v in value.split(",") if v.strip()]
    unknown = [n for n in names if n not in experiments.GRAPHS]
    if unknown:
        raise SystemExit(f"Unknown graph(s) {unknown}; choose from {list(experiments.GRAPHS)}")
    return names


def parse_seeds(value: str | None) -> tuple[int, ...] | None:
    return tuple(int(v) for v in value.split(",") if v.strip()) if value else None


def stage_build_graph(args: argparse.Namespace, layout: Layout) -> None:
    from triage.graph import build_graph

    device = resolve_device(args.device)
    for name in parse_graphs(args.graph):
        spec = experiments.GRAPHS[name]
        if layout.graph_path(name).exists() and not args.force:
            print(f"Graph {name} exists at {layout.graph_path(name)}; skipping (use --force to rebuild).")
            continue
        print(f"Building graph {name}: {spec.description}")
        build_graph(Path(args.raw_root) / "Eclipse", spec.encoder, spec.config, device, layout.graph_stem(name))


def stage_baselines(args: argparse.Namespace, layout: Layout) -> None:
    from triage.baselines import evaluate_classical_baselines

    for name in parse_graphs(args.graph):
        output = layout.baselines_path(name, experiments.BASELINE_SEED)
        if output.exists() and not args.force:
            print(f"Baselines for {name} exist at {output}; skipping.")
            continue
        evaluate_classical_baselines(
            layout.graph_path(name), Path(args.raw_root) / "Eclipse", output,
            seed=experiments.BASELINE_SEED, include_tfidf=not args.no_tfidf,
        )


def stage_train(args: argparse.Namespace, layout: Layout) -> None:
    from triage.summarize import summarize_run_directory
    from triage.train import train_run

    device = resolve_device(args.device)
    override = parse_seeds(args.seeds)
    if args.run:
        jobs = [(args.graph, args.run, override or experiments.CORE_SEEDS)]
    else:
        jobs = list(experiments.SUITES[args.suite])
    touched: set[str] = set()
    for graph, run, seeds in jobs:
        spec = experiments.RUNS[(graph, run)]
        for seed in (override or seeds):
            stem = layout.run_stem(graph, run, seed)
            if layout.metrics_path(graph, run, seed).exists() and not args.force:
                print(f"Skipping completed run {stem.name}")
                continue
            if not layout.graph_path(graph).exists():
                raise SystemExit(f"Graph {graph} not built; run `python pipeline.py build-graph --graph {graph}` first.")
            tee = Tee(stem.with_suffix(".log"))
            with contextlib.redirect_stdout(tee):
                try:
                    train_run(layout.graph_path(graph), spec.train_config(seed), stem, device)
                finally:
                    tee.close()
            touched.add(graph)
    for graph in sorted(touched):
        summarize_run_directory(layout.run_dir(graph))


def stage_summarize(args: argparse.Namespace, layout: Layout) -> None:
    from triage.summarize import summarize_run_directory

    for name in experiments.GRAPHS:
        if layout.run_dir(name).exists():
            summarize_run_directory(layout.run_dir(name))


def stage_analyse(args: argparse.Namespace, layout: Layout) -> None:
    which = set(args.which.split(",")) if args.which != "all" else {"eligibility", "tenure", "leaks", "drift", "attention", "deletion"}
    project_dir = Path(args.raw_root) / "Eclipse"
    analysis = layout.analysis_dir
    conditions = [(name, experiments.GRAPHS[name].config) for name in ("dev97", "dev209") if layout.labels_path(name).exists()]

    if "eligibility" in which:
        from triage.analysis.eligibility import Condition, audit_eligibility

        audit_eligibility(
            project_dir,
            [Condition(n, layout.labels_path(n), c.min_history_dev_bugs, c.min_train_dev_bugs, c.max_developers) for n, c in conditions],
            analysis / "eligibility_bug_cache.json",
            analysis / "eligibility_audit.json",
        )
    if "tenure" in which:
        from triage.analysis.tenure import TenureCondition, compute_tenure_bias

        compute_tenure_bias(
            project_dir,
            [TenureCondition(n, layout.labels_path(n), layout.summary_path(n), c.min_history_dev_bugs, c.min_train_dev_bugs, c.max_developers) for n, c in conditions],
            analysis / "tenure_bias.json",
        )
    if "leaks" in which:
        from triage.analysis.leaks import count_residual_leaks

        count_residual_leaks(project_dir, layout.splits_path("dev97"), layout.summary_path("dev97"), analysis / "residual_leaks.json")
    if "drift" in which:
        from triage.analysis.drift import drift_and_reporter_stats

        drift_and_reporter_stats(layout.graph_path("dev97"), layout.splits_path("dev97"), layout.labels_path("dev97"), analysis / "drift_reporter_stats.json")
    if "attention" in which or "deletion" in which:
        from triage.analysis.attention import inference_deletion, inspect_attention

        device = resolve_device(args.device)

        def checkpoint(arm: str, seed: int) -> Path:
            return layout.run_dir("dev97") / f"{arm}_research_full_seed{seed}_model.pt"

        if "attention" in which:
            inspect_attention(layout.graph_path("dev97"), {arm: checkpoint(arm, 42) for arm in ("hgt", "han")}, device, analysis / "attention_seed42.json")
        if "deletion" in which:
            inference_deletion(
                layout.graph_path("dev97"),
                {arm: {seed: checkpoint(arm, seed) for seed in (42, 73) if checkpoint(arm, seed).exists()} for arm in ("hgt", "han")},
                device, analysis / "inference_deletion.json",
            )


def stage_report(args: argparse.Namespace, layout: Layout) -> None:
    from triage.report import write_report

    write_report(layout)


def stage_figures(args: argparse.Namespace, layout: Layout) -> None:
    from triage.figures import make_figures

    make_figures(layout, Path(args.figures_dir))


def stage_all(args: argparse.Namespace, layout: Layout) -> None:
    args.graph = "all"
    stage_build_graph(args, layout)
    args.graph = ",".join(experiments.BASELINE_GRAPHS)
    stage_baselines(args, layout)
    args.suite, args.run = "all", None
    stage_train(args, layout)
    stage_summarize(args, layout)
    args.which = "all"
    stage_analyse(args, layout)
    stage_report(args, layout)
    stage_figures(args, layout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", default="F:/Thesis/RAW DATA", help="folder containing the Eclipse/ export (buglist, Bugs, BugHistory)")
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--figures-dir", default=str(ROOT / "figures"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("build-graph", help="build one or more graphs from the raw export")
    p.add_argument("--graph", default="all", help="comma-separated graph names or 'all'")
    p = sub.add_parser("baselines", help="classical baselines on a graph")
    p.add_argument("--graph", default=",".join(experiments.BASELINE_GRAPHS))
    p.add_argument("--no-tfidf", action="store_true", help="skip the TF-IDF text baseline (needs the raw export)")
    p = sub.add_parser("train", help="train the runs of a suite")
    p.add_argument("--suite", default="all", choices=list(experiments.SUITES))
    p.add_argument("--run", help="train one run by name instead of a suite (with --graph)")
    p.add_argument("--graph", default="dev97")
    p.add_argument("--seeds", help="comma-separated seeds overriding the suite's seeds")
    sub.add_parser("summarize", help="aggregate seeds per run directory")
    p = sub.add_parser("analyse", help="run the analyses")
    p.add_argument("--which", default="all", help="comma-separated subset of eligibility,tenure,leaks,drift,attention,deletion")
    sub.add_parser("report", help="write artifacts/results.md")
    sub.add_parser("figures", help="write the figures")
    sub.add_parser("all", help="run every stage in order")

    args = parser.parse_args(argv)
    layout = Layout(Path(args.artifacts))
    stages = {
        "build-graph": stage_build_graph, "baselines": stage_baselines, "train": stage_train, "summarize": stage_summarize,
        "analyse": stage_analyse, "report": stage_report, "figures": stage_figures, "all": stage_all,
    }
    stages[args.stage](args, layout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

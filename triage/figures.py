"""The figures: one claim per figure, black and grey, no prose on the canvas."""
from __future__ import annotations

import glob
import os
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .experiments import CORE_SEEDS, SIX_SEEDS
from .layout import Layout
from .summarize import load_runs
from .util import read_json

BLACK, GREY, LIGHT = "black", "#666666", "#bbbbbb"
TEXTWIDTH_IN = 5.9


def configure_style() -> None:
    for pattern in (
        os.path.expanduser("~/texlive/*/texmf-dist/fonts/truetype/**/Lato-*.ttf"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "MiKTeX", "fonts", "truetype", "**", "Lato-*.ttf"),
    ):
        for ttf in glob.glob(pattern, recursive=True):
            font_manager.fontManager.addfont(ttf)
    has_lato = any("Lato" in f.name for f in font_manager.fontManager.ttflist)
    plt.rcParams.update({
        "font.family": "Lato" if has_lato else "DejaVu Sans",
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
    })


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_dir / name}.pdf")


def fig_paired(metrics: dict, out_dir: Path) -> None:
    """Per-seed macro recall difference, HAN minus HGT. Filled: first six seeds; hollow: the rest."""
    shared = sorted(set(metrics["han_research_full"]) & set(metrics["hgt_research_full"]))
    key = "test_macro_recall"
    deltas = sorted((metrics["han_research_full"][s][key] - metrics["hgt_research_full"][s][key], s) for s in shared)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.6, 3.2))
    ys = list(range(len(deltas)))
    for y, (d, s) in zip(ys, deltas):
        ax.plot([d], [y], linestyle="none", marker="o", markersize=4, markerfacecolor=BLACK if s in SIX_SEEDS else "white", markeredgecolor=BLACK, markeredgewidth=0.7)
    ax.axvline(0, color=BLACK, linewidth=0.6)
    ax.axvline(statistics.mean(d for d, _ in deltas), color=GREY, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.set_yticks(ys)
    ax.set_yticklabels([str(s) for _, s in deltas], fontsize=7, color=GREY)
    ax.set_ylim(-0.7, len(deltas) - 0.3)
    ax.set_xlabel("HAN $-$ HGT, test macro recall")
    ax.xaxis.grid(True, linewidth=0.3, color=LIGHT)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, out_dir, "paired_macro_recall")


def fig_ablations(metrics: dict, out_dir: Path) -> None:
    """Change in test score when each family is removed, three-seed means; control row set off by a rule."""
    families = [
        ("hgt_ablation_no_candidates", "Candidates"),
        ("hgt_ablation_no_relation_similarity", "Relation similarity"),
        ("hgt_ablation_no_collaboration", "Collaboration"),
        ("hgt_ablation_no_recency", "Recency"),
        ("hgt_ablation_no_typed_history", "Typed history (control)"),
    ]
    keys = [("test_top1", "Top-1", "o"), ("test_macro_recall", "macro recall", "s"), ("test_mrr", "MRR", "^")]
    seeds = list(CORE_SEEDS)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.8, 2.6))
    ys = list(range(len(families)))[::-1]
    for key, label, marker in keys:
        base = statistics.mean(metrics["hgt_research_full"][s][key] for s in seeds)
        deltas = [statistics.mean(metrics[c][s][key] for s in seeds) - base for c, _ in families]
        ax.plot(deltas, ys, linestyle="none", marker=marker, markersize=5, markerfacecolor="white" if marker != "o" else BLACK, markeredgecolor=BLACK, markeredgewidth=0.7, label=label)
    ax.axvline(0, color=BLACK, linewidth=0.6)
    ax.axhline(0.5, color=GREY, linewidth=0.5, linestyle=(0, (2, 2)))
    ax.set_yticks(ys)
    ax.set_yticklabels([n for _, n in families])
    ax.set_xlabel("change in test score when the family is removed")
    ax.xaxis.grid(True, linewidth=0.3, color=LIGHT)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, handletextpad=0.3, columnspacing=1.6)
    save(fig, out_dir, "ablations")


def fig_timeline(summary: dict, out_dir: Path) -> None:
    """The four chronological periods to scale, with boundary dates and counts."""
    b = {k: datetime.fromisoformat(v) for k, v in summary["split_boundaries"].items()}
    t0, t1 = b["min_created_at"], b["max_created_at"]
    span = (t1 - t0).days
    edges = [t0, b["history_end_created_at"], b["train_end_created_at"], b["val_end_created_at"], t1]
    names = ["History", "Training", "Validation", "Test"]
    labelled = [summary["split_counts"][k] for k in ("history", "train", "val", "test")]
    total = summary["node_counts"]["bug"]
    reports = [round(total * p) for p in (0.50, 0.20, 0.15, 0.15)]
    fills = ["#d9d9d9", "#f2f2f2", "#d9d9d9", "#f2f2f2"]

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 1.9))
    xs = [(e - t0).days / span for e in edges]
    for i, name in enumerate(names):
        ax.barh(0, xs[i + 1] - xs[i], left=xs[i], height=0.5, color=fills[i], edgecolor=BLACK, linewidth=0.6)
        ax.text((xs[i] + xs[i + 1]) / 2, 0.0, name, ha="center", va="center", fontsize=8.5)
    date_x = xs[:]
    date_x[1] -= 0.012
    date_x[2] += 0.012
    for x, e in zip(date_x, edges):
        ax.text(x, 0.32, e.strftime("%Y-%m-%d"), ha="center", va="bottom", fontsize=7)
    for x in xs:
        ax.plot([x, x], [-0.25, 0.30], color=BLACK, linewidth=0.6)
    offsets = [-0.33, -0.33, -0.60, -0.33]
    for i in range(4):
        cx = (xs[i] + xs[i + 1]) / 2
        ax.text(cx, offsets[i], f"{reports[i]:,} reports\n{labelled[i]:,} labelled", ha="center", va="top", fontsize=7, color=GREY, linespacing=1.15)
        if offsets[i] < -0.4:
            ax.plot([cx, cx], [-0.26, offsets[i] + 0.02], color=GREY, linewidth=0.5)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-1.0, 0.65)
    ax.axis("off")
    save(fig, out_dir, "timeline")


def fig_topk_baselines(metrics: dict, baselines: dict, out_dir: Path) -> None:
    """Graph models against the metadata classifier at every k; neural arms over every shared seed."""
    shared = sorted(set(metrics["han_research_full"]) & set(metrics["hgt_research_full"]))
    ks = [1, 3, 5, 10]
    b = baselines["baselines"]
    series = [
        ("Naive Bayes, uniform prior", [b["metadata_naive_bayes_uniform"][f"test_top{k}"] for k in ks], dict(color=GREY, marker="D", markerfacecolor="white", linestyle="-")),
        ("Naive Bayes, empirical prior", [b["metadata_naive_bayes"][f"test_top{k}"] for k in ks], dict(color=BLACK, marker="s", markerfacecolor="white", linestyle="-")),
        ("HAN full", [statistics.mean(metrics["han_research_full"][s][f"test_top{k}"] for s in shared) for k in ks], dict(color=BLACK, marker="o", linestyle="-")),
        ("HGT full", [statistics.mean(metrics["hgt_research_full"][s][f"test_top{k}"] for s in shared) for k in ks], dict(color=BLACK, marker="^", markerfacecolor="white", linestyle="--")),
    ]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.6, 2.8))
    for name, ys, style in series:
        ax.plot(ks, ys, linewidth=0.8, markersize=4.5, markeredgewidth=0.7, label=name, **style)
    ax.set_xticks(ks)
    ax.set_xlabel("k")
    ax.set_ylabel("Top-k accuracy on the test period")
    ax.set_ylim(0.3, 0.9)
    ax.yaxis.grid(True, linewidth=0.3, color=LIGHT)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, out_dir, "topk_baselines")


def fig_training_curves(run_dir: Path, out_dir: Path) -> None:
    """Training and validation macro recall per epoch for the two full arms over the first six seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.5), sharex=True)
    for cfg, colour, label in (("han_research_full", BLACK, "HAN"), ("hgt_research_full", GREY, "HGT")):
        first = True
        for s in SIX_SEEDS:
            path = run_dir / f"{cfg}_seed{s}_history.json"
            if not path.exists():
                continue
            h = read_json(path)
            epochs = [r["epoch"] for r in h]
            axes[0].plot(epochs, [r["train_macro_recall"] for r in h], color=colour, linewidth=0.6, alpha=0.9, label=label if first else None)
            axes[1].plot(epochs, [r["val_macro_recall"] for r in h], color=colour, linewidth=0.6, alpha=0.9, label=label if first else None)
            first = False
    axes[0].set_ylabel("training macro recall")
    axes[1].set_ylabel("validation macro recall")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, linewidth=0.3, color=LIGHT)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.subplots_adjust(wspace=0.35)
    save(fig, out_dir, "training_curves")


def make_figures(layout: Layout, out_dir: Path) -> None:
    configure_style()
    metrics = load_runs(layout.run_dir("dev97"))
    summary_path = layout.summary_path("dev97")
    baselines_path = layout.baselines_path("dev97", 42)
    if {"han_research_full", "hgt_research_full"} <= set(metrics):
        fig_paired(metrics, out_dir)
        fig_training_curves(layout.run_dir("dev97"), out_dir)
        if baselines_path.exists():
            fig_topk_baselines(metrics, read_json(baselines_path), out_dir)
    if all(f"hgt_ablation_no_{f}" in metrics for f in ("candidates", "relation_similarity", "collaboration", "recency", "typed_history")):
        fig_ablations(metrics, out_dir)
    if summary_path.exists():
        fig_timeline(read_json(summary_path), out_dir)

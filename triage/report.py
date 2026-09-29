"""Render every result in the artifacts directory as one Markdown document."""
from __future__ import annotations

import statistics as st
from pathlib import Path

from .experiments import CORE_SEEDS, FAMILIES
from .layout import Layout
from .metrics import sign_flip_p
from .summarize import load_runs
from .util import optional_json

METRICS = [
    ("test_top1", "Top-1"), ("test_top3", "Top-3"), ("test_top5", "Top-5"), ("test_top10", "Top-10"),
    ("test_macro_recall", "Macro recall"), ("test_macro_f1", "Macro F1"), ("test_mrr", "MRR"), ("test_ndcg", "NDCG"),
]
LIFT_METRICS = [("test_top1", "Top-1"), ("test_top5", "Top-5"), ("test_top10", "Top-10"), ("test_macro_recall", "Macro recall"), ("test_macro_f1", "Macro F1"), ("test_mrr", "MRR")]
MAIN_ORDER = [
    "hgt_research_full", "han_research_full", "hgt_hybrid_full", "hgt_causal_base", "han_causal_base",
    "hgt_notext_full", "han_notext_full", "feature_only_wide", "feature_only",
    *(f"hgt_ablation_no_{family}" for family in FAMILIES), "han_ablation_no_recency",
]

Runs = dict[str, dict[int, dict]]


def mean(runs: Runs, model: str, key: str, seeds: list[int] | None = None) -> float:
    chosen = seeds if seeds is not None else sorted(runs[model])
    return st.fmean(runs[model][s][key] for s in chosen)


def table(runs: Runs, models: list[str], seeds: list[int] | None = None) -> list[str]:
    lines = ["| Run | n | " + " | ".join(name for _, name in METRICS) + " | Parameters |", "|---|--:|" + "--:|" * (len(METRICS) + 1)]
    for model in models:
        if model not in runs:
            continue
        chosen = [s for s in sorted(runs[model]) if seeds is None or s in seeds]
        if not chosen:
            continue
        cells = " | ".join(f"{mean(runs, model, key, chosen):.4f}" for key, _ in METRICS)
        lines.append(f"| `{model}` | {len(chosen)} | {cells} | {runs[model][chosen[0]]['trainable_parameters']:,} |")
    return lines


def paired(runs: Runs, a: str, b: str) -> list[str]:
    common = sorted(set(runs[a]) & set(runs[b]))
    lines = [
        f"Shared seeds: {common} (n = {len(common)}, smallest attainable two-sided p = {2 / 2 ** len(common):.4g}).", "",
        f"| Metric | `{a}` | `{b}` | Δ (first − second) | SD | Seeds favouring first | Sign-flip p |",
        "|---|--:|--:|--:|--:|:--:|--:|",
    ]
    for key, name in METRICS:
        deltas = [runs[a][s][key] - runs[b][s][key] for s in common]
        lines.append(
            f"| {name} | {mean(runs, a, key, common):.4f} | {mean(runs, b, key, common):.4f} | {st.fmean(deltas):+.4f} "
            f"| {st.stdev(deltas) if len(deltas) > 1 else 0.0:.4f} | {sum(d > 0 for d in deltas)}/{len(deltas)} | {sign_flip_p(deltas):.4f} |"
        )
    return lines


def baseline_table(baselines: dict) -> list[str]:
    keys = [(k, n) for k, n in METRICS if k != "test_ndcg"]
    lines = ["| Baseline | " + " | ".join(n for _, n in keys) + " |", "|---|" + "--:|" * len(keys)]
    for name, values in baselines["baselines"].items():
        lines.append(f"| `{name}` | " + " | ".join(f"{values[k]:.4f}" for k, _ in keys) + " |")
    return lines


def render(layout: Layout) -> str:
    out: list[str] = []
    w = out.append
    dev97 = load_runs(layout.run_dir("dev97"))
    dev209 = load_runs(layout.run_dir("dev209"))
    mpnet = load_runs(layout.run_dir("mpnet"))
    summaries = {name: optional_json(layout.summary_path(name)) for name in ("dev97", "dev209", "mpnet")}
    baselines = {name: optional_json(layout.baselines_path(name, 42)) for name in ("dev97", "dev209")}

    w("# Results\n")
    w("Generated from the files under `artifacts/`; every number below is recomputed from them on each run.\n")

    # 1. Graphs
    present = [name for name, summary in summaries.items() if summary]
    if present:
        w("## 1. Graphs\n")
        w("| | " + " | ".join(f"`{name}`" for name in present) + " |")
        w("|---|" + "--:|" * len(present))
        rows = [
            ("Text encoder", lambda s: s["text_encoder"]),
            ("Developers (classes)", lambda s: f"{s['num_classes']}"),
            ("Bugs", lambda s: f"{s['node_counts']['bug']:,}"),
            ("Reporters", lambda s: f"{s['node_counts'].get('reporter', 0):,}"),
            ("Components", lambda s: f"{s['node_counts']['component']:,}"),
            ("Relation types", lambda s: f"{len(s['edge_counts'])}"),
            ("Edges", lambda s: f"{sum(s['edge_counts'].values()):,}"),
            *((f"{split.capitalize()} bugs", (lambda s, sp=split: f"{s['split_counts'][sp]:,}")) for split in ("history", "train", "val", "test")),
            ("History ends", lambda s: s["split_boundaries"]["history_end_created_at"][:10]),
            ("Training ends", lambda s: s["split_boundaries"]["train_end_created_at"][:10]),
            ("Validation ends", lambda s: s["split_boundaries"]["val_end_created_at"][:10]),
            ("Candidate oracle recall (test)", lambda s: f"{s['candidate_diagnostics']['test']['oracle_recall']:.4f}"),
        ]
        for label, fn in rows:
            w(f"| {label} | " + " | ".join(fn(summaries[name]) for name in present) + " |")
        w("")
        for name in present:
            r = summaries[name]["raw_stats"]
            w(f"Data reduction for `{name}`: {r['buglist_rows']:,} buglist rows; {r['dropped_placeholder_bug_rows']:,} queue-owned rows dropped; "
              f"{r['candidate_bugs']:,} individually assigned bugs across {r['selected_developers']:,} developers; "
              f"{r['removed_cold_label_bugs']:,} bugs removed by the eligibility rule; {summaries[name]['node_counts']['bug']:,} bugs in the graph.")
        w("")

    audit = optional_json(layout.analysis_dir / "eligibility_audit.json")
    if audit:
        w("### Eligibility rule replay\n")
        w("| Condition | Cap | Passes | Reproduces label map | Pass-1 history cutoff | Pass-1 passing | Final kept | Cap bound on passes | Cut by cap | Exclusion reasons |")
        w("|---|--:|--:|:--:|---|--:|--:|---|--:|---|")
        for cname, cond in audit["conditions"].items():
            passes = cond["passes"]
            cap_passes = [p["pass"] for p in passes if p["cap_bound"]]
            reasons = ", ".join(f"{k} {v}" for k, v in sorted(cond["exclusion_reason_counts"].items()))
            w(f"| {cname} | {cond['rule']['max_developers']} | {len(passes)} | {'yes' if cond['reproduction_exact'] else 'no'} | {passes[0]['history_cutoff'][:10]} "
              f"| {passes[0]['developers_passing_thresholds']} | {passes[-1]['developers_kept']} | {cap_passes or 'none'} | {len(cond['developers_cut_by_cap'])} | {reasons} |")
        w("")

    # 2. Main results
    if dev97:
        w(f"## 2. Primary condition (`dev97`), seeds {list(CORE_SEEDS)}\n")
        w("\n".join(table(dev97, MAIN_ORDER, list(CORE_SEEDS))))
        w("")
        extra = sorted(set(dev97.get("hgt_research_full", {})) - set(CORE_SEEDS))
        if extra:
            w(f"`hgt_research_full` and `han_research_full` also have seeds {extra}; section 4 uses every shared seed.\n")

    # 3. Baselines
    if baselines["dev97"]:
        w("## 3. Classical baselines (`dev97`, seed 42)\n")
        w("\n".join(baseline_table(baselines["dev97"])))
        w("")

    # 4. HAN vs HGT
    if "han_research_full" in dev97 and "hgt_research_full" in dev97:
        common = sorted(set(dev97["han_research_full"]) & set(dev97["hgt_research_full"]))
        w(f"## 4. HAN against HGT, paired over {len(common)} seeds (`dev97`)\n")
        w("\n".join(paired(dev97, "han_research_full", "hgt_research_full")))
        w("")
        deltas = [dev97["han_research_full"][s]["test_macro_recall"] - dev97["hgt_research_full"][s]["test_macro_recall"] for s in common]
        w("Per-seed macro recall difference (HAN − HGT): " + ", ".join(f"{s}: {d:+.4f}" for s, d in zip(common, deltas)) + ".")
        bonferroni = 0.05 / len(METRICS)
        survivors = [name for key, name in METRICS if sign_flip_p([dev97["han_research_full"][s][key] - dev97["hgt_research_full"][s][key] for s in common]) < bonferroni]
        w(f"Metrics with p below the Bonferroni threshold {bonferroni:.4f} over {len(METRICS)} metrics: {', '.join(survivors) if survivors else 'none'}.\n")

    # 5. Ablations
    ablations = sorted(k for k in dev97 if k.startswith("hgt_ablation_no_"))
    if ablations and "hgt_research_full" in dev97:
        w("## 5. Relation-family ablations, paired against the full arm (`dev97`)\n")
        w("Negative values mean the family helps.\n")
        w("| Ablation | Reference | n | Top-1 Δ | Macro recall Δ | MRR Δ | Seeds favouring ablation (Top-1) | Sign-flip p (Top-1) |")
        w("|---|---|--:|--:|--:|--:|:--:|--:|")
        for model in ablations + (["han_ablation_no_recency"] if "han_ablation_no_recency" in dev97 else []):
            reference = "han_research_full" if model.startswith("han_") else "hgt_research_full"
            common = sorted(set(dev97[model]) & set(dev97[reference]))
            if not common:
                continue
            d = {key: [dev97[model][s][key] - dev97[reference][s][key] for s in common] for key in ("test_top1", "test_macro_recall", "test_mrr")}
            w(f"| `{model}` | `{reference}` | {len(common)} | {st.fmean(d['test_top1']):+.4f} | {st.fmean(d['test_macro_recall']):+.4f} | {st.fmean(d['test_mrr']):+.4f} "
              f"| {sum(x > 0 for x in d['test_top1'])}/{len(common)} | {sign_flip_p(d['test_top1']):.4f} |")
        w("")

    # 6. Operator comparison without research families
    if "han_causal_base" in dev97 and "hgt_causal_base" in dev97:
        w("## 6. HAN against HGT on the causal base (all research families dropped, `dev97`)\n")
        w("\n".join(paired(dev97, "han_causal_base", "hgt_causal_base")))
        w("")

    # 7. Text contribution
    for arch in ("hgt", "han"):
        full, notext = f"{arch}_research_full", f"{arch}_notext_full"
        if notext in dev97 and full in dev97:
            w(f"## 7{'a' if arch == 'hgt' else 'b'}. Bug features zeroed, {arch.upper()} (`dev97`)\n")
            w("\n".join(paired(dev97, notext, full)))
            common = sorted(set(dev97[notext]) & set(dev97[full]))
            w(f"\nBest epochs of the zeroed arm: {[dev97[notext][s]['best_epoch'] for s in common]} (cap {int(dev97[notext][common[0]]['epochs_ran']) if len(common) else ''}).\n")

    # 8. Capacity-matched MLP
    if "feature_only_wide" in dev97 and "feature_only" in dev97:
        w("## 8. Feature-only MLP at two widths (`dev97`)\n")
        w("\n".join(table(dev97, ["feature_only_wide", "feature_only"], list(CORE_SEEDS))))
        wide = dev97["feature_only_wide"]
        w(f"\nBest epochs, wide: {[wide[s]['best_epoch'] for s in sorted(wide)]}.\n")

    # 9. Encoder comparison
    if mpnet:
        w("## 9. Encoder comparison (`mpnet` graph: all-mpnet-base-v2, recency omitted)\n")
        w("\n".join(table(mpnet, sorted(mpnet))))
        if "hgt_ablation_no_recency" in dev97:
            w("")
            w("BERT counterpart with the same relation set:\n")
            w("\n".join(table(dev97, ["hgt_ablation_no_recency"], list(CORE_SEEDS))))
        if "hgt_mpnet_no_recency" in mpnet:
            w("")
            w("| Run | Mean best validation macro recall | Mean test macro recall |")
            w("|---|--:|--:|")
            for name, runs in (("hgt_mpnet_no_recency", mpnet), ("hgt_ablation_no_recency", dev97)):
                if name in runs:
                    w(f"| `{name}` | {mean(runs, name, 'best_validation_score'):.4f} | {mean(runs, name, 'test_macro_recall'):.4f} |")
        w("")

    # 10. Label space
    if dev209 and baselines["dev97"] and baselines["dev209"]:
        nb97 = baselines["dev97"]["baselines"]["metadata_naive_bayes"]
        nb209 = baselines["dev209"]["baselines"]["metadata_naive_bayes"]
        w("## 10. Label-space scaling: `dev97` against `dev209`\n")
        w("The two conditions have different test sets, so only the lift over the within-condition Naive Bayes baseline is comparable.\n")
        w("| Condition | Model | " + " | ".join(n for _, n in LIFT_METRICS) + " |")
        w("|---|---|" + "--:|" * len(LIFT_METRICS))
        for label, runs, nb in (("97 classes", dev97, nb97), ("209 classes", dev209, nb209)):
            for model in ("hgt_research_full", "han_research_full"):
                if model in runs:
                    w(f"| {label} | `{model}` | " + " | ".join(f"{mean(runs, model, k):.4f}" for k, _ in LIFT_METRICS) + " |")
            w(f"| {label} | `metadata_naive_bayes` | " + " | ".join(f"{nb[k]:.4f}" for k, _ in LIFT_METRICS) + " |")
        w("")
        w("Lift over `metadata_naive_bayes`:\n")
        w("| Condition | Model | " + " | ".join(n for _, n in LIFT_METRICS) + " |")
        w("|---|---|" + "--:|" * len(LIFT_METRICS))
        for model in ("hgt_research_full", "han_research_full"):
            if model in dev97 and model in dev209:
                lift97 = {k: mean(dev97, model, k) - nb97[k] for k, _ in LIFT_METRICS}
                lift209 = {k: mean(dev209, model, k) - nb209[k] for k, _ in LIFT_METRICS}
                w(f"| 97 classes | `{model}` | " + " | ".join(f"{lift97[k]:+.4f}" for k, _ in LIFT_METRICS) + " |")
                w(f"| 209 classes | `{model}` | " + " | ".join(f"{lift209[k]:+.4f}" for k, _ in LIFT_METRICS) + " |")
                w(f"| change (209 − 97) | `{model}` | " + " | ".join(f"**{lift209[k] - lift97[k]:+.4f}**" for k, _ in LIFT_METRICS) + " |")
        w("")
        w("Parameters at 209 classes: " + ", ".join(f"`{m}` {dev209[m][sorted(dev209[m])[0]]['trainable_parameters']:,}" for m in ("hgt_research_full", "han_research_full") if m in dev209) + ".")
        w("Best epochs at 209 classes: " + "; ".join(f"`{m}` {[dev209[m][s]['best_epoch'] for s in sorted(dev209[m])]}" for m in ("hgt_research_full", "han_research_full") if m in dev209) + ".\n")
        w("Classical baselines at 209 classes (seed 42):\n")
        w("\n".join(baseline_table(baselines["dev209"])))
        w("")

    # 11. Validation vs test
    models = [m for m in ("hgt_hybrid_full", "hgt_causal_base", "hgt_research_full", "hgt_ablation_no_recency") if m in dev97 and 42 in dev97[m]]
    if models:
        w("## 11. Validation against test macro recall, seed 42 (`dev97`)\n")
        w("| Run | Best validation macro recall | Test macro recall | Difference | Best epoch |")
        w("|---|--:|--:|--:|--:|")
        for m in models:
            r = dev97[m][42]
            w(f"| `{m}` | {r['best_validation_score']:.4f} | {r['test_macro_recall']:.4f} | {r['test_macro_recall'] - r['best_validation_score']:+.4f} | {r['best_epoch']} |")
        w("")

    # 12. Tenure, leaks, drift
    tenure = optional_json(layout.analysis_dir / "tenure_bias.json")
    if tenure:
        w("## 12. Eligibility and tenure\n")
        w(f"Overall volume is bugs per final assignee over {tenure['eligible_bug_rows']:,} individually assigned bugs and {tenure['distinct_assignees']:,} developers.\n")
        w("| Condition | Retained | Rank span (median) | Top-100 excluded | Top-209 excluded | Retained ranked > 209 | Retained ranked > 500 | Share of bugs |")
        w("|---|--:|--:|--:|--:|--:|--:|--:|")
        for cname, c in tenure["conditions"].items():
            w(f"| {cname} | {c['retained_developers']} | {c['retained_rank_min']}–{c['retained_rank_max']} ({c['retained_rank_median']:g}) | {c['top_n']['100']['top_n_excluded']} "
              f"| {c['top_n']['209']['top_n_excluded']} | {c['top_n']['209']['retained_ranked_below_n']} | {c['retained_ranked_beyond_500']} | {100 * c['retained_share_of_eligible_bugs']:.1f}% |")
        w("")
        for cname, c in tenure["conditions"].items():
            w(f"Largest excluded developers, {cname}: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(c["largest_excluded_reason_counts"].items(), key=lambda t: -t[1])) + ".")
            for row in c["largest_excluded_developers"][:6]:
                w(f"- rank {row['overall_rank']}, {row['bug_volume']:,} bugs ({row['bugs_resolved_fixed']:,} fixed): {row['history_observed_bugs']} history-observed, "
                  f"{row['train_observed_bugs']} train-observed, first fix {str(row['first_fixed'])[:7]}: {row['exclusion_reason'].replace('_', ' ')}.")
            w("")

    leaks = optional_json(layout.analysis_dir / "residual_leaks.json")
    if leaks:
        sf, rr = leaks["summary_field_changed"], leaks["reopen_and_reassign"]
        w("## 13. Residual leak counts (`dev97`)\n")
        w("| Leak | Count | Of | Share |")
        w("|---|--:|--:|--:|")
        w(f"| Summary field changed after report | {sf['total']:,} | {leaks['graph_bugs']:,} graph bugs | {100 * sf['total_fraction']:.1f}% |")
        for split, frac in sf["by_split_fraction"].items():
            w(f"| Summary changed, {split} split | {sf['by_split'].get(split, 0):,} | {leaks['split_sizes'][split]:,} | {100 * frac:.1f}% |")
        w(f"| Last assignee change after history cutoff | {rr['history_last_assignee_change_after_cutoff']:,} | {rr['history_split_bugs']:,} history bugs | {100 * rr['history_last_assignee_change_after_cutoff_fraction']:.2f}% |")
        w(f"| Label equals that late assignee | {rr['history_label_equals_late_assignee']:,} | {rr['history_last_assignee_change_after_cutoff']:,} | |")
        w(f"| Last assignee change after training cutoff | {rr['train_last_assignee_change_after_cutoff']:,} | {rr['train_split_bugs']:,} training bugs | {100 * rr['train_last_assignee_change_after_cutoff_fraction']:.2f}% |")
        w("")

    drift = optional_json(layout.analysis_dir / "drift_reporter_stats.json")
    if drift:
        d = drift["label_drift"]
        w("## 14. Label drift and reporter self-assignment (`dev97`)\n")
        w("| Statistic | Value |")
        w("|---|--:|")
        w(f"| Most frequent training developer, share of training / test | {d['top1_training_developer_share_of_training']:.3f} / {d['top1_training_developer_share_of_test']:.3f} |")
        w(f"| Ten most frequent training developers, share of training / test | {d['top10_training_developers_share_of_training']:.3f} / {d['top10_training_developers_share_of_test']:.3f} |")
        w(f"| Top-ten overlap between training and test | {d['top10_overlap_between_periods']} |")
        w(f"| Developers with no test report | {d['developers_with_no_test_report']} of {d['developers']} |")
        w(f"| Spearman ρ, training vs test counts | {d['spearman_rho_train_vs_test_counts']:.3f} (p = {d['spearman_p']:.3f}) |")
        w("")
        w("| Split | Reports | Reporter is assignee | Reporter is an eligible developer |")
        w("|---|--:|--:|--:|")
        for split, r in drift["reporter_by_split"].items():
            w(f"| {split} | {r['reports']:,} | {r['reporter_is_assignee']:.3f} | {r['reporter_is_eligible_developer']:.3f} |")
        w("")

    # 15. Attention
    attention = optional_json(layout.analysis_dir / "attention_seed42.json")
    if attention:
        w("## 15. Attention by relation, final layer, seed 42 (`dev97`)\n")
        for arm, result in attention["arms"].items():
            last = result["attention_by_layer"][-1]
            top = sorted(last.items(), key=lambda kv: -kv[1])[:12]
            w(f"`{arm}` (test top-1 {result['test_top1']:.4f}), twelve largest shares:\n")
            w("| Relation | " + " | ".join(f"Layer {i + 1}" for i in range(len(result["attention_by_layer"]))) + " |")
            w("|---|" + "--:|" * len(result["attention_by_layer"]))
            for relation, _ in top:
                w(f"| `{relation}` | " + " | ".join(f"{layer.get(relation, 0.0):.3f}" for layer in result["attention_by_layer"]) + " |")
            w("")
    deletion = optional_json(layout.analysis_dir / "inference_deletion.json")
    if deletion:
        w("## 16. Inference-time deletion of relation families (`dev97`)\n")
        w("Change in test top-1 when a family's edges are removed at test time, without retraining.\n")
        w("| Arm | Seed | Full | " + " | ".join(deletion["families"]) + " |")
        w("|---|--:|--:|" + "--:|" * len(deletion["families"]))
        for row in deletion["results"]:
            w(f"| {row['arm']} | {row['seed']} | {row['full_test_top1']:.4f} | " + " | ".join(f"{row['delta_test_top1'][f]:+.4f}" for f in deletion["families"]) + " |")
        w("")

    # 17. Hardware
    manifest = next((optional_json(p) for p in sorted(layout.run_dir("dev97").glob("*_run.json"))), None)
    if manifest:
        w("## 17. Hardware and libraries\n")
        w(f"{manifest.get('cuda_device_name') or manifest.get('device')}; CUDA {manifest.get('cuda')}; torch {manifest.get('torch')}; "
          f"torch_geometric {manifest.get('torch_geometric')}; Python {str(manifest.get('python', '')).split()[0]}; {manifest.get('platform')}. "
          f"Training mode: {manifest.get('training_mode') or manifest.get('resolved_training_mode')}.\n")

    return "\n".join(out)


def write_report(layout: Layout) -> Path:
    text = render(layout)
    layout.results_path.parent.mkdir(parents=True, exist_ok=True)
    layout.results_path.write_text(text, encoding="utf-8")
    print(f"Wrote {layout.results_path} ({len(text):,} chars)")
    return layout.results_path

"""How far the eligibility rule departs from selecting developers by overall productivity."""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..raw import bug_sort_key, choose_candidate_bug_ids, clean_value, fixed_time_from_history, index_files, load_buglist, parse_bug_text, parse_history
from ..util import read_json, write_json


@dataclass(frozen=True)
class TenureCondition:
    name: str
    labels_path: Path
    summary_path: Path
    min_history: int
    min_train: int
    max_developers: int


def first_bug_created(bug_ids: list[str], text_paths: dict[str, Path], summary_by_id: dict[str, str], probe: int = 3) -> datetime | None:
    earliest: datetime | None = None
    for bug_id in sorted(bug_ids, key=bug_sort_key)[:probe]:
        created = parse_bug_text(text_paths[bug_id], summary_by_id.get(bug_id, "")).created_at
        if created is not None and (earliest is None or created < earliest):
            earliest = created
    return earliest


def exclusion_diagnostics(
    developer: str,
    bug_ids: list[str],
    history_paths: dict[str, Path],
    text_paths: dict[str, Path],
    summary_by_id: dict[str, str],
    history_cutoff: datetime,
    train_cutoff: datetime,
) -> dict[str, object]:
    """Reconstruct the rule's counters for one developer at the final cutoffs."""
    fixed_total = history_observed = train_observed = censored = 0
    first_fixed: datetime | None = None
    first_seen: datetime | None = None
    resolutions: Counter[str] = Counter()
    for bug_id in bug_ids:
        events = parse_history(history_paths.get(bug_id))
        fixed_at = fixed_time_from_history(events)
        created_at = parse_bug_text(text_paths[bug_id], summary_by_id.get(bug_id, "")).created_at
        if fixed_at is not None:
            fixed_total += 1
            if created_at is not None:
                if created_at <= history_cutoff:
                    if fixed_at <= history_cutoff:
                        history_observed += 1
                    else:
                        censored += 1
                elif created_at <= train_cutoff and fixed_at <= train_cutoff:
                    train_observed += 1
            if first_fixed is None or fixed_at < first_fixed:
                first_fixed = fixed_at
        final_resolution = "none"
        for event in events:
            if event.when is None:
                continue
            if event.what.lower() == "resolution" and event.added:
                final_resolution = event.added.upper()
            involved = event.who.lower() == developer.lower() or (event.what.lower() == "assignee" and event.added.lower() == developer.lower())
            if involved and (first_seen is None or event.when < first_seen):
                first_seen = event.when
        resolutions[final_resolution] += 1
    return {
        "bugs_resolved_fixed": fixed_total,
        "history_observed_bugs": history_observed,
        "history_period_censored_bugs": censored,
        "train_observed_bugs": train_observed,
        "first_fixed": first_fixed.isoformat() if first_fixed else None,
        "first_seen_in_history": first_seen.isoformat() if first_seen else None,
        "final_resolution_counts": dict(resolutions.most_common(6)),
    }


def compute_tenure_bias(
    project_dir: Path,
    conditions: list[TenureCondition],
    output_json: Path,
    top_n: tuple[int, ...] = (100, 209, 500),
    largest_excluded: int = 10,
) -> dict[str, object]:
    buglist = load_buglist(project_dir)
    text_paths = index_files(project_dir / "Bugs", ".txt")
    history_paths = index_files(project_dir / "BugHistory", ".csv")
    eligible_ids, filter_stats = choose_candidate_bug_ids(buglist, text_paths)

    volume: Counter[str] = Counter()
    bugs_by_dev: dict[str, list[str]] = {}
    summary_by_id: dict[str, str] = {}
    for bug_id in eligible_ids:
        assignee = clean_value(buglist[bug_id].get("Assignee", ""))
        volume[assignee] += 1
        bugs_by_dev.setdefault(assignee, []).append(bug_id)
        summary_by_id[bug_id] = clean_value(buglist[bug_id].get("Summary", ""), "")
    ranked = sorted(volume.items(), key=lambda item: (-item[1], item[0].lower()))
    rank_of = {developer: rank for rank, (developer, _) in enumerate(ranked, start=1)}

    report: dict[str, object] = {
        "overall_volume_definition": "bugs per final assignee over rows with an individual assignee and a text file (the graph builder's label-agnostic universe)",
        "buglist_rows": len(buglist),
        "eligible_bug_rows": len(eligible_ids),
        "distinct_assignees": len(volume),
        **filter_stats,
        "conditions": {},
    }
    for condition in conditions:
        boundaries = read_json(condition.summary_path)["split_boundaries"]
        history_cutoff = datetime.fromisoformat(boundaries["history_end_created_at"])
        train_cutoff = datetime.fromisoformat(boundaries["train_end_created_at"])
        retained = set(read_json(condition.labels_path))
        retained_ranks = sorted(rank_of[dev] for dev in retained if dev in rank_of)
        retained_volume = sum(volume[dev] for dev in retained)

        top_n_stats: dict[str, object] = {}
        for n in top_n:
            top = [dev for dev, _ in ranked[:n]]
            excluded = [dev for dev in top if dev not in retained]
            top_n_stats[str(n)] = {
                "top_n_excluded": len(excluded),
                "top_n_excluded_bug_volume": sum(volume[dev] for dev in excluded),
                "top_n_total_bug_volume": sum(volume[dev] for dev in top),
                "retained_ranked_below_n": sum(1 for dev in retained if dev in rank_of and rank_of[dev] > n),
            }

        rows = []
        reasons: Counter[str] = Counter()
        for dev, count in [(d, c) for d, c in ranked if d not in retained][:largest_excluded]:
            diag = exclusion_diagnostics(dev, bugs_by_dev[dev], history_paths, text_paths, summary_by_id, history_cutoff, train_cutoff)
            if diag["history_observed_bugs"] < condition.min_history:
                reason = "late_joiner_too_few_history_observed_fixes"
            elif diag["train_observed_bugs"] < condition.min_train:
                reason = "early_leaver_too_few_train_observed_fixes"
            else:
                reason = "cut_by_max_developers_cap"
            reasons[reason] += 1
            created = first_bug_created(bugs_by_dev[dev], text_paths, summary_by_id)
            rows.append(
                {
                    "developer": f"developer_rank_{rank_of[dev]}",
                    "overall_rank": rank_of[dev],
                    "bug_volume": count,
                    "first_bug_created": created.isoformat() if created else None,
                    **diag,
                    "fixed_share": diag["bugs_resolved_fixed"] / count,
                    "exclusion_reason": reason,
                }
            )

        report["conditions"][condition.name] = {
            "eligibility_rule": {"min_history_fixed_bugs": condition.min_history, "min_train_fixed_bugs": condition.min_train, "max_developers": condition.max_developers},
            "history_cutoff": history_cutoff.isoformat(),
            "train_cutoff": train_cutoff.isoformat(),
            "retained_developers": len(retained),
            "retained_bug_volume": retained_volume,
            "retained_share_of_eligible_bugs": retained_volume / max(len(eligible_ids), 1),
            "retained_rank_min": retained_ranks[0] if retained_ranks else None,
            "retained_rank_median": statistics.median(retained_ranks) if retained_ranks else None,
            "retained_rank_max": retained_ranks[-1] if retained_ranks else None,
            "retained_ranked_beyond_500": sum(1 for rank in retained_ranks if rank > 500),
            "top_n": top_n_stats,
            "largest_excluded_reason_counts": dict(reasons),
            "largest_excluded_developers": rows,
        }
        print(f"[{condition.name}] retained={len(retained)} ranks {retained_ranks[0]}..{retained_ranks[-1]} (median {statistics.median(retained_ranks):g})")
    write_json(output_json, report)
    return report

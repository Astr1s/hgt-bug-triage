"""Replay the developer-eligibility rule and record why each developer dropped out."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..raw import RawBug, choose_candidate_bug_ids, clean_value, fixed_time_from_history, index_files, load_buglist, parse_bug_text, parse_history
from ..splits import build_causal_splits
from ..util import read_json, write_json


@dataclass(frozen=True)
class Condition:
    name: str
    labels_path: Path
    min_history: int
    min_train: int
    max_developers: int
    history_ratio: float = 0.50
    train_ratio: float = 0.20
    val_ratio: float = 0.15


def load_minimal_bugs(project_dir: Path, cache_path: Path, progress_every: int = 20000) -> tuple[list[RawBug], dict[str, int]]:
    """Only (id, assignee, created_at, fixed_at): enough to replay the rule. Cached because it takes minutes."""
    if cache_path.exists():
        payload = read_json(cache_path)
        bugs = [
            RawBug(
                bug_id=row["bug_id"], assignee=row["assignee"], summary="", final_product="", final_component="",
                created_at=datetime.fromisoformat(row["created_at"]), reporter="", initial_text="",
                fixed_at=datetime.fromisoformat(row["fixed_at"]) if row["fixed_at"] else None,
            )
            for row in payload["bugs"]
        ]
        return bugs, payload["stats"]

    buglist = load_buglist(project_dir)
    text_paths = index_files(project_dir / "Bugs", ".txt")
    history_paths = index_files(project_dir / "BugHistory", ".csv")
    selected_ids, filter_stats = choose_candidate_bug_ids(buglist, text_paths)
    bugs: list[RawBug] = []
    missing_created = 0
    for offset, bug_id in enumerate(selected_ids, start=1):
        created_at = parse_bug_text(text_paths[bug_id], "").created_at
        if created_at is None:
            missing_created += 1
            continue
        bugs.append(
            RawBug(
                bug_id=bug_id, assignee=clean_value(buglist[bug_id].get("Assignee", "")), summary="", final_product="",
                final_component="", created_at=created_at, reporter="", initial_text="",
                fixed_at=fixed_time_from_history(parse_history(history_paths.get(bug_id))),
            )
        )
        if progress_every > 0 and offset % progress_every == 0:
            print(f"Loaded {offset:,}/{len(selected_ids):,} raw bugs", flush=True)
    stats = {**filter_stats, "candidate_bugs": len(selected_ids), "missing_created_at": missing_created, "selected_bugs": len(bugs)}
    write_json(
        cache_path,
        {
            "stats": stats,
            "bugs": [
                {"bug_id": b.bug_id, "assignee": b.assignee, "created_at": b.created_at.isoformat(), "fixed_at": b.fixed_at.isoformat() if b.fixed_at else None}
                for b in bugs
            ],
        },
    )
    return bugs, stats


def replay(bugs: list[RawBug], condition: Condition, volume: Counter[str], rank_of: dict[str, int]) -> dict[str, object]:
    def token(developer: str) -> str:
        return f"developer_rank_{rank_of[developer]}"

    current = list(bugs)
    passes: list[dict[str, object]] = []
    exclusions: dict[str, dict[str, object]] = {}
    label_to_id: dict[str, int] = {}
    for pass_index in range(1, 10):
        history_mask, train_mask, _, _, boundaries, _ = build_causal_splits(current, condition.history_ratio, condition.train_ratio, condition.val_ratio)
        history_counts = Counter(bug.assignee for idx, bug in enumerate(current) if bool(history_mask[idx]))
        train_counts = Counter(bug.assignee for idx, bug in enumerate(current) if bool(train_mask[idx]))
        present = sorted({bug.assignee for bug in current}, key=str.lower)
        passing = [
            developer
            for developer, count in sorted(history_counts.items(), key=lambda item: (-item[1], item[0].lower()))
            if count >= condition.min_history and train_counts[developer] >= condition.min_train
        ]
        kept = passing[: condition.max_developers] if condition.max_developers > 0 else passing
        label_to_id = {developer: idx for idx, developer in enumerate(kept)}
        cap_bound = condition.max_developers > 0 and len(passing) > condition.max_developers

        removed_this_pass: list[str] = []
        for developer in present:
            if developer in label_to_id:
                continue
            removed_this_pass.append(developer)
            if history_counts[developer] < condition.min_history:
                reason = "below_history_threshold"
            elif train_counts[developer] < condition.min_train:
                reason = "below_train_threshold"
            else:
                reason = "cut_by_max_developers_cap"
            exclusions[developer] = {
                "pass": pass_index,
                "reason": reason,
                "history_observed_bugs": history_counts[developer],
                "train_observed_bugs": train_counts[developer],
                "history_cutoff_at_pass": boundaries["history_end_created_at"],
                "train_cutoff_at_pass": boundaries["train_end_created_at"],
                "overall_bug_volume": volume[developer],
                "overall_rank": rank_of[developer],
            }
        reasons_this_pass = Counter(exclusions[d]["reason"] for d in removed_this_pass)
        passes.append(
            {
                "pass": pass_index,
                "bugs": len(current),
                "developers_present": len(present),
                "history_cutoff": boundaries["history_end_created_at"],
                "train_cutoff": boundaries["train_end_created_at"],
                "developers_passing_thresholds": len(passing),
                "cap_bound": cap_bound,
                "developers_kept": len(kept),
                "removed_below_history_threshold": reasons_this_pass["below_history_threshold"],
                "removed_below_train_threshold": reasons_this_pass["below_train_threshold"],
                "removed_by_cap": reasons_this_pass["cut_by_max_developers_cap"],
            }
        )
        filtered = [bug for bug in current if bug.assignee in label_to_id]
        if len(filtered) == len(current):
            break
        current = filtered

    committed = set(read_json(condition.labels_path))
    reproduced = set(label_to_id)
    rows = sorted(({"developer": token(dev), **row} for dev, row in exclusions.items()), key=lambda row: row["overall_rank"])
    return {
        "rule": {
            "min_history_observed_bugs": condition.min_history,
            "min_train_observed_bugs": condition.min_train,
            "max_developers": condition.max_developers,
        },
        "committed_developers": len(committed),
        "reproduced_developers": len(reproduced),
        "reproduction_exact": reproduced == committed,
        "passes": passes,
        "cap_bound_on_any_pass": any(p["cap_bound"] for p in passes),
        "exclusion_reason_counts": dict(Counter(row["reason"] for row in exclusions.values())),
        "developers_cut_by_cap": [row for row in rows if row["reason"] == "cut_by_max_developers_cap"],
        "largest_excluded_developers": rows[:25],
    }


def audit_eligibility(project_dir: Path, conditions: list[Condition], cache_path: Path, output_json: Path) -> dict[str, object]:
    bugs, stats = load_minimal_bugs(project_dir, cache_path)
    volume: Counter[str] = Counter(bug.assignee for bug in bugs)
    ranked = sorted(volume.items(), key=lambda item: (-item[1], item[0].lower()))
    rank_of = {developer: rank for rank, (developer, _) in enumerate(ranked, start=1)}
    report: dict[str, object] = {"raw_stats": stats, "conditions": {}}
    for condition in conditions:
        result = replay(bugs, condition, volume, rank_of)
        report["conditions"][condition.name] = result
        print(f"[{condition.name}] reproduced {result['reproduced_developers']} developers; exact: {result['reproduction_exact']}; passes: {len(result['passes'])}")
    write_json(output_json, report)
    return report

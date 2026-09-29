"""Count the bugs touched by the two residual leaks disclosed in the thesis."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from ..raw import index_files, is_placeholder_assignee, parse_history
from ..util import read_json, write_json


def count_residual_leaks(project_dir: Path, splits_csv: Path, summary_json: Path, output_json: Path, progress_every: int = 10000) -> dict[str, object]:
    boundaries = read_json(summary_json)["split_boundaries"]
    cutoffs = {
        "history": datetime.fromisoformat(boundaries["history_end_created_at"]),
        "train": datetime.fromisoformat(boundaries["train_end_created_at"]),
    }
    with Path(splits_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    history_paths = index_files(project_dir / "BugHistory", ".csv")

    split_sizes: Counter[str] = Counter(row["split"] for row in rows)
    summary_changed: Counter[str] = Counter()
    summary_change_events: Counter[str] = Counter()
    missing_history: Counter[str] = Counter()
    any_assignee_change: Counter[str] = Counter()
    last_assignee_after_cutoff: Counter[str] = Counter()
    any_assignee_after_cutoff: Counter[str] = Counter()
    label_matches_last_added: Counter[str] = Counter()
    examples: defaultdict[str, list[dict[str, object]]] = defaultdict(list)

    for offset, row in enumerate(rows, start=1):
        split, bug_id = row["split"], row["bug_id"]
        path = history_paths.get(bug_id)
        if path is None:
            missing_history[split] += 1
            continue
        events = parse_history(path)
        summary_events = [event for event in events if event.what.lower() == "summary"]
        if summary_events:
            summary_changed[split] += 1
            summary_change_events[split] += len(summary_events)
        assignee_events = [event for event in events if event.what.lower() == "assignee" and event.when is not None]
        cutoff = cutoffs.get(split)
        if assignee_events:
            any_assignee_change[split] += 1
            if cutoff is not None:
                last = assignee_events[-1]
                if last.when > cutoff:
                    last_assignee_after_cutoff[split] += 1
                    if last.added.lower() == row["assignee"].lower():
                        label_matches_last_added[split] += 1
                    if len(examples[split]) < 5:
                        examples[split].append(
                            {
                                "bug_id": bug_id,
                                "created_at": row["created_at_utc"],
                                "last_assignee_change": last.when.isoformat(),
                                "from_placeholder_queue": is_placeholder_assignee(last.removed),
                                "label_equals_late_assignee": last.added.lower() == row["assignee"].lower(),
                            }
                        )
                if any(event.when > cutoff for event in assignee_events):
                    any_assignee_after_cutoff[split] += 1
        if progress_every > 0 and offset % progress_every == 0:
            print(f"Scanned {offset}/{len(rows)} bug histories", flush=True)

    total = len(rows)
    history_n, train_n = split_sizes.get("history", 0), split_sizes.get("train", 0)
    report = {
        "graph_bugs": total,
        "split_sizes": dict(split_sizes),
        "history_cutoff": cutoffs["history"].isoformat(),
        "train_cutoff": cutoffs["train"].isoformat(),
        "missing_history_files": dict(missing_history),
        "summary_field_changed": {
            "definition": "graph bugs with at least one history event on the Summary field",
            "total": sum(summary_changed.values()),
            "total_fraction": sum(summary_changed.values()) / max(total, 1),
            "by_split": dict(summary_changed),
            "by_split_fraction": {split: summary_changed[split] / split_sizes[split] for split in split_sizes},
            "summary_change_events": dict(summary_change_events),
        },
        "reopen_and_reassign": {
            "definition": "bugs in the history (training) split whose last Assignee change post-dates the history (training) cutoff",
            "history_split_bugs": history_n,
            "history_bugs_with_any_assignee_change": any_assignee_change.get("history", 0),
            "history_last_assignee_change_after_cutoff": last_assignee_after_cutoff.get("history", 0),
            "history_last_assignee_change_after_cutoff_fraction": last_assignee_after_cutoff.get("history", 0) / max(history_n, 1),
            "history_any_assignee_change_after_cutoff": any_assignee_after_cutoff.get("history", 0),
            "history_label_equals_late_assignee": label_matches_last_added.get("history", 0),
            "train_split_bugs": train_n,
            "train_last_assignee_change_after_cutoff": last_assignee_after_cutoff.get("train", 0),
            "train_last_assignee_change_after_cutoff_fraction": last_assignee_after_cutoff.get("train", 0) / max(train_n, 1),
            "train_any_assignee_change_after_cutoff": any_assignee_after_cutoff.get("train", 0),
            "examples": dict(examples),
        },
    }
    write_json(output_json, report)
    print(f"Summary changed: {report['summary_field_changed']['total']} of {total}; late reassignments in history split: {last_assignee_after_cutoff.get('history', 0)}")
    return report

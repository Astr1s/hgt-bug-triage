"""Parse the raw Bugzilla export into report-time bug records."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

BUG_FILE_RE = re.compile(r"bugs(\d+)", re.IGNORECASE)
DESCRIPTION_RE = re.compile(
    r"^(?:\[reply\]\s*)?Description\s+(?P<who>.*?)\s+"
    r"(?P<stamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<tz>[A-Z]{2,5})\s*$"
)
COMMENT_RE = re.compile(r"^(?:\[reply\]\s*)?Comment\s+\d+\b", re.IGNORECASE)
DATETIME_RE = re.compile(r"(?P<stamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(?P<tz>[A-Z]{2,5})")
BUG_ID_RE = re.compile(r"\d+")

TZ_OFFSETS = {
    "UTC": 0, "GMT": 0, "EDT": -4, "EST": -5, "CDT": -5, "CST": -6,
    "MDT": -6, "MST": -7, "PDT": -7, "PST": -8, "CET": 1, "CEST": 2,
}

OPTIONAL_METADATA_FIELDS = {"priority": "Priority", "severity": "Severity", "milestone": "Target Milestone"}

# Generic Bugzilla owners are routing queues rather than individual developers.
PLACEHOLDER_ASSIGNEE_LOCAL_PARTS = {
    "default", "issues", "needsconfirm", "needs-confirmation", "nobody",
    "swneedsconfirm", "triage", "unassigned", "webmaster",
}


@dataclass(frozen=True)
class TextSnapshot:
    created_at: datetime | None
    reporter: str
    text: str


@dataclass(frozen=True)
class HistoryEvent:
    when: datetime | None
    who: str
    what: str
    added: str
    removed: str


@dataclass
class RawBug:
    bug_id: str
    assignee: str
    summary: str
    final_product: str
    final_component: str
    created_at: datetime
    reporter: str
    initial_text: str
    history: list[HistoryEvent] = field(default_factory=list)
    initial_product: str = "unknown"
    initial_component: str = "unknown"
    initial_priority: str = "unknown"
    initial_severity: str = "unknown"
    initial_milestone: str = "unknown"
    fixed_at: datetime | None = None


def natural_buglist_key(path: Path) -> tuple[int, str]:
    match = BUG_FILE_RE.search(path.stem)
    return (int(match.group(1)) if match else 10**9, path.name)


def bug_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def clean_value(value: object, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan" or text == "---":
        return default
    return text


def is_placeholder_assignee(value: str) -> bool:
    """Whether an assignee is a generic queue or default account."""
    cleaned = clean_value(value, "").strip().lower()
    if not cleaned:
        return True
    local_part = cleaned.split("@", 1)[0]
    compact = re.sub(r"[^a-z0-9]+", "-", local_part).strip("-")
    if "inbox" in compact.split("-"):
        return True
    return compact in PLACEHOLDER_ASSIGNEE_LOCAL_PARTS


def parse_raw_datetime(value: str) -> datetime | None:
    match = DATETIME_RE.search(value.strip())
    if not match:
        return None
    offset_hours = TZ_OFFSETS.get(match.group("tz").upper())
    if offset_hours is None:
        return None
    try:
        local = datetime.strptime(match.group("stamp"), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    tz = timezone(timedelta(hours=offset_hours), name=match.group("tz").upper())
    return local.replace(tzinfo=tz).astimezone(timezone.utc)


def load_buglist(project_dir: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in sorted((project_dir / "buglist").glob("bugs*.csv"), key=natural_buglist_key):
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                normalized = {clean_value(key, ""): clean_value(value, "") for key, value in row.items()}
                bug_id = normalized.get("Bug ID", "")
                if bug_id:
                    rows[bug_id] = normalized
    return rows


def index_files(root: Path, extension: str) -> dict[str, Path]:
    return {path.stem: path for path in root.glob(f"bugs*/*{extension}")}


def parse_bug_text(path: Path, summary: str) -> TextSnapshot:
    """Report-time text: the summary plus the description before the first comment."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return TextSnapshot(created_at=None, reporter="unknown", text=summary)

    created_at: datetime | None = None
    reporter = "unknown"
    body_start = 1
    for idx, line in enumerate(lines[:20]):
        match = DESCRIPTION_RE.match(line.strip())
        if match:
            created_at = parse_raw_datetime(f"{match.group('stamp')} {match.group('tz')}")
            reporter = clean_value(match.group("who"))
            body_start = idx + 1
            break

    body_lines: list[str] = []
    for line in lines[body_start:]:
        if COMMENT_RE.match(line.strip()):
            break
        body_lines.append(line)

    text = f"{summary}\n" + "\n".join(body_lines)
    return TextSnapshot(created_at=created_at, reporter=reporter, text=text.strip() or summary)


def parse_history(path: Path | None) -> list[HistoryEvent]:
    if path is None or not path.exists():
        return []
    events: list[HistoryEvent] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {clean_value(key, ""): clean_value(value, "") for key, value in raw.items()}
            events.append(
                HistoryEvent(
                    when=parse_raw_datetime(row.get("When", "")),
                    who=clean_value(row.get("Who", "")),
                    what=clean_value(row.get("What", "")),
                    added=clean_value(row.get("Added", "")),
                    removed=clean_value(row.get("Removed", "")),
                )
            )
    events.sort(key=lambda event: event.when or datetime.max.replace(tzinfo=timezone.utc))
    return events


def initial_value_from_history(events: list[HistoryEvent], field_name: str, fallback: str = "unknown") -> str:
    """The value a field had at report time: what the first change to it removed."""
    for event in events:
        if event.what.lower() == field_name.lower():
            return clean_value(event.removed, fallback)
    return clean_value(fallback)


def fixed_time_from_history(events: list[HistoryEvent]) -> datetime | None:
    for event in events:
        if event.when is not None and event.what.lower() == "resolution" and event.added.upper() == "FIXED":
            return event.when
    for event in events:
        if event.when is not None and event.what.lower() == "status" and event.added.upper() in {"RESOLVED", "VERIFIED", "CLOSED"}:
            return event.when
    return None


def component_key(product: str, component: str) -> str:
    return f"{product}::{component}"


def split_bug_reference_ids(value: str, allow_plain_numbers: bool = True) -> list[str]:
    """Extract bug ids from Bugzilla-style history fields."""
    if not allow_plain_numbers:
        text = value.lower()
        if "bug" not in text and "show_bug.cgi" not in text:
            return []
    return BUG_ID_RE.findall(value)


def choose_candidate_bug_ids(
    buglist: dict[str, dict[str, str]],
    text_paths: dict[str, Path],
) -> tuple[list[str], dict[str, int]]:
    """Bugs with a text file and an individual (non-queue) assignee, in id order."""
    eligible: list[str] = []
    dropped_placeholders = 0
    for bug_id, row in buglist.items():
        assignee = clean_value(row.get("Assignee", ""))
        if not assignee or bug_id not in text_paths:
            continue
        if is_placeholder_assignee(assignee):
            dropped_placeholders += 1
            continue
        eligible.append(bug_id)
    eligible.sort(key=bug_sort_key)
    return eligible, {"dropped_placeholder_bug_rows": dropped_placeholders}


def load_raw_bugs(project_dir: Path, progress_every: int = 10000) -> tuple[list[RawBug], dict[str, int]]:
    """Load every eligible bug with its report-time snapshot and full history."""
    buglist = load_buglist(project_dir)
    text_paths = index_files(project_dir / "Bugs", ".txt")
    history_paths = index_files(project_dir / "BugHistory", ".csv")
    selected_ids, filter_stats = choose_candidate_bug_ids(buglist, text_paths)

    bugs: list[RawBug] = []
    stats = {
        "buglist_rows": len(buglist),
        "text_files": len(text_paths),
        "history_files": len(history_paths),
        "candidate_bugs": len(selected_ids),
        "missing_created_at": 0,
        "missing_history": 0,
        "developer_count_before_created_filter": len(
            {clean_value(buglist[bug_id].get("Assignee", "")) for bug_id in selected_ids}
        ),
        **filter_stats,
    }

    for offset, bug_id in enumerate(selected_ids, start=1):
        row = buglist[bug_id]
        summary = clean_value(row.get("Summary", ""), "")
        snapshot = parse_bug_text(text_paths[bug_id], summary)
        if snapshot.created_at is None:
            stats["missing_created_at"] += 1
            continue
        history_path = history_paths.get(bug_id)
        if history_path is None:
            stats["missing_history"] += 1
        history = parse_history(history_path)
        final_product = clean_value(row.get("Product", ""))
        final_component = clean_value(row.get("Component", ""))
        bugs.append(
            RawBug(
                bug_id=bug_id,
                assignee=clean_value(row.get("Assignee", "")),
                summary=summary,
                final_product=final_product,
                final_component=final_component,
                created_at=snapshot.created_at,
                reporter=snapshot.reporter,
                initial_text=snapshot.text,
                history=history,
                initial_product=initial_value_from_history(history, "Product", final_product),
                initial_component=initial_value_from_history(history, "Component", final_component),
                initial_priority=initial_value_from_history(history, OPTIONAL_METADATA_FIELDS["priority"]),
                initial_severity=initial_value_from_history(history, OPTIONAL_METADATA_FIELDS["severity"]),
                initial_milestone=initial_value_from_history(history, OPTIONAL_METADATA_FIELDS["milestone"]),
                fixed_at=fixed_time_from_history(history),
            )
        )
        if progress_every > 0 and offset % progress_every == 0:
            print(f"Loaded raw bug snapshots {offset}/{len(selected_ids)}", flush=True)

    stats["selected_bugs"] = len(bugs)
    stats["selected_developers"] = len({bug.assignee for bug in bugs})
    return bugs, stats

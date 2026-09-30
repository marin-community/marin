#!/usr/bin/env python3
"""Conveyor view of a capability-construction run: where every item is, and what is wrong.

Read-only against S3 and Iris.  Local state (object cache, manifest index cache, first-seen
tracking, throughput history) lives in ``ops/state/``.

Run from the marin-construct checkout (it has s3fs), with the CoreWeave keys exported in the
same shell invocation:

  export CW_KEY_ID=$(gcloud secrets versions access latest --secret=cw-object-storage-key-id --project=hai-gcp-models) \
         CW_KEY_SECRET=$(gcloud secrets versions access latest --secret=cw-object-storage-key-secret --project=hai-gcp-models)
  cd /Users/k3sc0re/openathena/marin-construct
  PYTHONPATH=/Users/k3sc0re/openathena/capability_env_gen-dev/scripts \
    uv run --frozen /Users/k3sc0re/openathena/capability_env_gen-dev/ops/conveyor.py            # one-screen summary
    ... ops/conveyor.py --json                                 # full model
    ... ops/conveyor.py --items --run 'hc*' --stage quality_review --disp active
    ... ops/conveyor.py --watch --interval 300                 # edge-triggered lines for a monitor

Status shapes.  Today's ``items/<item>/status.json`` has ``state``, ``issues``, ``custom_images``,
``repairs``, ``repair_budget`` and ``terminal_disposition``; timing comes from the S3 object's
LastModified (content-addressed, so it is when that exact status was first synced), the
``controller/active-operation.json`` record, and first-seen tracking.  When the richer fields land
(``updated_at``, ``state_since``, ``transitions``, ``wait``, ``activity``, ``failure_stage`` and a
run-level ``conveyor.json``) they take precedence automatically.

Sources, cheapest first: a base whose job publishes ``<base>/_live/conveyor.json``
(capability-conveyor-v1, a few KB, re-read only on ETag change) is built from it alone; other
bases fall back to the manifest.  ``--watch`` re-reads those manifests only every
``--manifest-every`` cycles (default 6) and never when the ETag is unchanged.

Every read failure is loud: a missing credential, an empty base listing or an empty/truncated
job list raises, and in ``--watch`` becomes an ``ERROR`` line rather than silence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

RUN = "catalog-full-construct-003"
S3_ROOT = f"s3://marin-us-east-02a/users/muchanem/capability-pipeline/runs/{RUN}"
JOB_PREFIX = "/muchanem/cap-construct-003"
MARIN_DIR = Path("/Users/k3sc0re/openathena/marin-construct")
LIVE_TREE = Path("/Users/k3sc0re/openathena/capability_env_gen")
BUILD_DIR = LIVE_TREE / "build" / "construct-003"
OPS_DIR = Path(__file__).resolve().parent
STATE_DIR = OPS_DIR / "state"

JOB_RE = re.compile(r"^/muchanem/cap-construct-003-(?P<base>.+?)(?:-(?P<suffix>[a-z][0-9]+))?$")
LIVE_JOB_STATES = frozenset({"pending", "building", "running"})
TERMINAL_JOB_STATES = frozenset({"succeeded", "failed", "killed", "worker_failed", "unschedulable"})

# --------------------------------------------------------------------------------------------
# Stage model (mirrors capability_pipeline/synthesis.py + image_pipeline.py as of 2026-09-29)
# --------------------------------------------------------------------------------------------

STAGES: tuple[tuple[str, str], ...] = (
    ("queued", "queued (not started)"),
    ("build", "builder sessions"),
    ("bundle", "bundle checks"),
    ("judge_policy", "judge policy"),
    ("image_review", "image plan review"),
    ("image_capture", "image capture"),
    ("image_publication", "image publication"),
    ("image_cold_pull", "image cold pull"),
    ("image_migration", "image pointer migration"),
    ("build_acceptance", "build acceptance"),
    ("lowering", "schema validation/lowering"),
    ("judge_calibration", "judge calibration"),
    ("runtime_controls", "runtime controls"),
    ("adjudication", "solver/attack adjudication"),
    ("repeated_diagnostics", "repeated diagnostics"),
    ("quality_review", "quality review"),
    ("readmission", "needs readmission"),
    ("controller", "controller/conveyor error"),
    ("accepted", "quality_accepted"),
    ("unknown", "unknown/unreadable"),
)
STAGE_KEYS = tuple(key for key, _ in STAGES)
STAGE_INDEX = {key: index for index, key in enumerate(STAGE_KEYS)}

STATE_STAGE = {
    "pending_build": "build",
    "pending_judge_policy": "judge_policy",
    "pending_build_acceptance": "build_acceptance",
    "pending_schema_validation": "lowering",
    "lowered": "lowering",
    "pending_judge_calibration": "judge_calibration",
    "pending_runtime": "runtime_controls",
    "controls_passed_pending_rollout": "runtime_controls",
    "pending_solver_adjudication": "adjudication",
    "pending_adversary_retry": "adjudication",
    "pending_attack_adjudication": "adjudication",
    "runtime_controls_passed_pending_adversary": "adjudication",
    "validated": "repeated_diagnostics",
    "pending_repeated_diagnostics": "repeated_diagnostics",
    # synthesis re-runs the gate after a health probe (capability_pipeline.conveyor).
    "pending_runtime_infrastructure": "runtime_controls",
    "pending_quality_review": "quality_review",
    "pending_readmission": "readmission",
    "quality_accepted": "accepted",
}
IMAGE_SUBSTATE_STAGE = {
    "pending_review": "image_review",
    "pending_capture": "image_capture",
    "pending_infrastructure": "image_capture",
    "pending_publication": "image_publication",
    "pending_cold_pull": "image_cold_pull",
    "pending_migration": "image_migration",
}
IMAGE_REASON_STAGE = {
    "invalid_authored_image_documents": "image_review",
    "invalid_builder_image_request": "image_review",
    "image_sensitive_paths_present": "image_capture",
}
# capability_pipeline/conveyor.py (capability-conveyor-v1) stage names -> funnel stages.
SCHEDULER_STAGE = {
    "accepted": "accepted", "readmission": "readmission", "adversary_retry": "adjudication",
    "configuration": "judge_policy", "schema_validation": "lowering", "task_bundle": "bundle",
    "duplicate_task_id": "lowering", "build": "build", "builder": "build", "builder_process": "build",
    "builder_continuation": "build", "solver_adjudication": "adjudication", "attack_adjudication": "adjudication",
    "adversary": "adjudication", "image_infrastructure": "image_capture",
    "image_review_transport": "image_review",
    "runtime_infrastructure": "runtime_controls",
    "unclassified_state": "controller", "controller_exception": "controller", "conveyor": "controller",
}
# synthesize_one: states a resume feeds to the bounded repair loop instead of re-running.
REPAIRABLE = frozenset({
    "failed",
    "pending_build_acceptance",
    "pending_judge_calibration",
    "pending_solver_adjudication",
    "pending_attack_adjudication",
    "runtime_controls_passed_pending_adversary",
    "pending_quality_review",
    "pending_repeated_diagnostics",
})
SEMANTIC_CONTROL = re.compile(
    r"[^:]+: (?:reward is below reward_min|reward is above reward_max|"
    r"criterion assertion failed at .+|positive control did not earn >=0\.8|"
    r"negative control earned >0\.2|status 'graded', expected 'extraction_error')"
)
DISPOSITIONS = ("queued", "active", "waiting", "parked", "held", "rejected", "accepted", "unknown")
# active = in the current pass (pending or in progress); waiting = returned this pass, moves next pass;
# parked = progressable but the run has no live job; held = frozen under a plain --resume.
NON_TERMINAL = frozenset({"queued", "active", "waiting", "parked", "held", "unknown"})

DEFAULT_STUCK_HOURS = {
    "build": 12.0, "bundle": 3.0, "judge_policy": 3.0,
    "image_review": 3.0, "image_capture": 3.0, "image_publication": 4.0,
    "image_cold_pull": 3.0, "image_migration": 3.0, "build_acceptance": 4.0,
    "lowering": 3.0, "judge_calibration": 5.0, "runtime_controls": 6.0,
    "adjudication": 5.0, "repeated_diagnostics": 6.0, "quality_review": 5.0,
    "readmission": 8.0, "unknown": 6.0, "repair": 5.0,
}
STALE_SNAPSHOT_MIN = 20.0


def is_known_state(state: Any) -> bool:
    return isinstance(state, str) and (
        state in STATE_STAGE or state == "failed" or state.startswith("pending_image_")
    )


def _issues(status: dict) -> list[str]:
    value = status.get("issues")
    if isinstance(value, list):
        return [str(issue) for issue in value]
    return [str(value)] if value else []


def image_stage(status: dict) -> str:
    images = status.get("custom_images") if isinstance(status.get("custom_images"), dict) else {}
    sub = images.get("state")
    if sub in IMAGE_SUBSTATE_STAGE:
        return IMAGE_SUBSTATE_STAGE[sub]
    if sub == "repairable":
        reason = str(images.get("reason") or "")
        return IMAGE_REASON_STAGE.get(reason, "image_review")
    state = str(status.get("state") or "")
    return IMAGE_SUBSTATE_STAGE.get("pending_" + state.removeprefix("pending_image_"), "image_review")


def failure_stage(status: dict) -> str:
    """Where a failed (or terminally rejected) item failed.  ``failure_stage`` wins when present."""
    declared = status.get("failure_stage")
    if isinstance(declared, str) and declared:
        mapped = declared if declared in STAGE_INDEX else SCHEDULER_STAGE.get(declared)
        if mapped:
            return mapped
        # e.g. "construction": attribute from the issue text below.
    state = status.get("state")
    if state != "failed":
        return state_stage(status, _no_failure=True)
    issues = _issues(status)
    first = issues[0] if issues else ""
    images = status.get("custom_images") if isinstance(status.get("custom_images"), dict) else {}
    if first.startswith("invalid task bundle:") and images.get("state") == "repairable":
        return IMAGE_REASON_STAGE.get(str(images.get("reason") or ""), "image_review")
    if "authored verifier image uses Python" in first or status.get("verifier_image_compatibility_failures"):
        return "runtime_controls"
    if first.startswith("invalid task bundle:"):
        return "bundle"
    if first.startswith("TaskCompendium validation/lowering failed") or first.startswith(
        "duplicate generated TaskSpec id"
    ):
        return "lowering"
    if first.startswith("runtime controls failed") or SEMANTIC_CONTROL.fullmatch(first):
        return "runtime_controls"
    # Issues rewritten by a repair round: fall back to the furthest gate the result carries.
    if status.get("quality_review"):
        return "quality_review"
    if status.get("repeated_diagnostics"):
        return "repeated_diagnostics"
    if status.get("attack_adjudication") or status.get("incomplete_adversary"):
        return "adjudication"
    if status.get("runtime_evidence"):
        return "runtime_controls"
    if status.get("judge_calibration") or status.get("judge_calibration_failure"):
        return "judge_calibration"
    if status.get("taskcompendium"):
        return "runtime_controls"
    if images.get("state") == "ready":
        return "build_acceptance"
    if images:
        return image_stage(status)
    return "bundle"


def state_stage(status: dict | None, _no_failure: bool = False) -> str:
    """Pipeline stage of an item from its status (``None`` = started, no status yet)."""
    if status is None:
        return "build"
    state = status.get("state")
    if state == "pending_build":
        issues = " ".join(_issues(status))
        return "bundle" if "missing final bundle files" in issues else "build"
    if state in STATE_STAGE:
        return STATE_STAGE[state]
    if isinstance(state, str) and state.startswith("pending_image_"):
        return image_stage(status)
    if state == "failed" and not _no_failure:
        return failure_stage(status)
    return "unknown"


def repair_eligible(status: dict) -> bool | None:
    """Mirror of synthesis._fresh_construction_repair_allowed; ``None`` = needs artifact checks."""
    state = status.get("state")
    if state == "pending_attack_adjudication":
        return None  # _proven_exploit_for_repair inspects receipts we do not read here
    if state == "pending_judge_calibration":
        return None if isinstance(status.get("judge_calibration_failure"), dict) else False
    if state == "pending_build_acceptance":
        return True
    if state == "pending_quality_review":
        review = status.get("quality_review")
        return isinstance(review, dict) and review.get("state") in {"repair", "reject", "insufficient_evidence"}
    if state != "failed":
        return False
    issues = _issues(status)
    if not issues:
        return False
    if all(issue.startswith(("invalid task bundle:", "TaskCompendium validation/lowering failed:")) for issue in issues):
        return True
    return all(SEMANTIC_CONTROL.fullmatch(issue) for issue in issues)


def resume_class(status: dict | None) -> tuple[str, str]:
    """What a plain ``--resume`` relaunch would do with this item (heuristic, see synthesize_one).

    Returns (class, reason) with class in: terminal, progress, repair, maybe, frozen.
    """
    if status is None:
        return "progress", "first attempt (builder sessions)"
    if status.get("terminal") is True:
        return "terminal", "status.terminal"
    state = status.get("state")
    if state == "quality_accepted":
        return "terminal", "accepted"
    if status.get("terminal_disposition"):
        return "terminal", f"terminal_disposition={status.get('terminal_disposition')}"
    if not is_known_state(state):
        return "maybe", f"unknown state {state!r}"
    if state == "pending_adversary_retry":
        return "frozen", "pending_adversary_retry needs --retry-adversary"
    if state not in REPAIRABLE:
        return "progress", "resume re-enters the attempt"
    budget = status.get("repair_budget") if isinstance(status.get("repair_budget"), dict) else {}
    used, maximum = budget.get("used"), budget.get("max")
    eligible = repair_eligible(status)
    if eligible is False:
        return "frozen", f"{state} not repair-eligible; resume keeps it"
    if isinstance(used, int) and isinstance(maximum, int) and used >= maximum:
        return "frozen", f"repair budget exhausted ({used}/{maximum})"
    if eligible is None:
        return "maybe", f"{state}: repair eligibility needs artifact checks"
    nxt = (used or 0) + 1
    return "repair", f"repair round {nxt}/{maximum if maximum is not None else '?'} on resume"


def held_in_force(status: dict | None, row: dict | None = None) -> dict | None:
    """The item's ``wait_hold`` when it still applies (state preserved), else ``None``.

    capability_pipeline.conveyor writes a hold -- a step's ``retryable: False`` or a transient
    controller exception on a waiting/fresh item -- as a terminal conveyor row that keeps the
    state, so the next job re-enters the item.  Such an item is held, not finished.
    """
    source = status if isinstance(status, dict) and isinstance(status.get("wait_hold"), dict) else row
    hold = (source or {}).get("wait_hold") if isinstance(source, dict) else None
    if not isinstance(hold, dict):
        return None
    state = status.get("state") if isinstance(status, dict) else (row or {}).get("state")
    if state == "quality_accepted" or hold.get("state") != state:
        return None
    return hold


def _held_reason(hold: dict) -> str:
    return f"conveyor: wait_hold {hold.get('kind') or '?'} until {hold.get('blocked_until') or 'job_relaunch'}"


# --------------------------------------------------------------------------------------------
# Reason normalisation
# --------------------------------------------------------------------------------------------

_EXC_LINE = re.compile(r"^\s*((?:[A-Za-z_][\w.]*\.)?[A-Z]\w*(?:Error|Exception|Exit|Timeout|Interrupt|Failure))\b:?\s?(.*)$")
_SCRUB = (
    (re.compile(r"'[^']*'|\"[^\"]*\""), "'<q>'"),
    (re.compile(r"(?:/[\w.@+-]+){2,}/?"), "<path>"),
    (re.compile(r"\b[0-9a-f]{8,}\b"), "<hex>"),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), "#"),
    (re.compile(r"\s+"), " "),
)
_RULES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"authored reference \S+ failed in Harbor \((\w+)\)"), r"authored reference failed in Harbor (\1)"),
    (re.compile(r"No such file or directory: .*/(workspace/task/[\w.-]+)'?"), r"missing \1"),
    (re.compile(r"^[^:]+: reward is below reward_min$"), "control reward below reward_min"),
    (re.compile(r"^[^:]+: reward is above reward_max$"), "control reward above reward_max"),
    (re.compile(r"^[^:]+: negative control earned >0\.2.*"), "negative control earned >0.2"),
    (re.compile(r"^[^:]+: positive control did not earn >=0\.8.*"), "positive control did not earn >=0.8"),
    (re.compile(r"^[^:]+: criterion assertion failed at .*"), "criterion assertion failed"),
    (re.compile(r"^Controller gate, not fixable.*"), "repair: controller gate not fixable in workspace"),
    (re.compile(r"^control \S+ has invalid (\w+)"), r"control has invalid \1"),
)


def issue_head(issue: str, limit: int = 140) -> str:
    """First line of an issue; for tracebacks, the prefix plus the final exception line."""
    text = str(issue)
    if "Traceback (most recent call last)" in text or "\n" in text:
        prefix = text.split("\n", 1)[0]
        if "Traceback" in prefix:
            prefix = prefix.split("Traceback", 1)[0].rstrip(": ")
        exc = None
        for line in reversed(text.splitlines()):
            match = _EXC_LINE.match(line)
            if match:
                exc = f"{match.group(1)}: {match.group(2)}".rstrip(": ")
                break
        text = f"{prefix}: {exc}" if exc and prefix else (exc or prefix)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def normalize_reason(issue: str) -> str:
    """Collapse an issue string into a stable class (names, paths, numbers scrubbed)."""
    head = issue_head(issue, limit=400)
    gate, _, rest = head.partition(": ")
    gates = {
        "invalid task bundle": "bundle",
        "runtime controls failed": "runtime",
        "TaskCompendium validation/lowering failed": "lowering",
        "native judge calibration is incomplete": "judge-calibration",
        "independent attack adjudication is incomplete": "attack-adjudication",
        "independent semantic review is incomplete": "quality-review",
        "repeated runtime diagnostics are incomplete": "diagnostics",
        "builder session s1 stopped": "builder",
    }
    label = gates.get(gate)
    body = rest if label and rest else head
    for pattern, replacement in _RULES:
        if pattern.search(body):
            body = pattern.sub(replacement, body) if "\\" in replacement else replacement
            break
    for pattern, replacement in _SCRUB:
        body = pattern.sub(replacement, body)
    body = body.strip()
    if len(body) > 90:
        body = body[:89] + "…"
    return f"{label}: {body}" if label else body


def reason_of(status: dict | None) -> str | None:
    if not status:
        return None
    issues = _issues(status)
    if issues:
        return issues[0]
    images = status.get("custom_images") if isinstance(status.get("custom_images"), dict) else {}
    return images.get("reason")


# --------------------------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------------------------


def parse_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), UTC)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if value else None


def fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def hhmm(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%H:%M:%SZ")


# --------------------------------------------------------------------------------------------
# Manifest indexing (pure; the manifest is sorted, indent=2 JSON with one member per line)
# --------------------------------------------------------------------------------------------

INDEX_VERSION = 2
RE_CREATED = re.compile(rb'"created_utc":\s*"([^"]+)"')
RE_FINAL = re.compile(rb'\n\s{0,4}"final":\s*(true|false)')
RE_SNAPSHOT = re.compile(rb'\n\s{0,4}"snapshot_id":\s*"([^"]+)"')
_HEX = rb'"([0-9a-f]{64})"'
# Each needle starts with a literal that is rare in a manifest, so ``re`` can skip ahead in C;
# the member path is then recovered from the start of the line.
NEEDLE_STATUS = re.compile(rb'/status\.json":\s*' + _HEX)
NEEDLE_ACTIVE = re.compile(rb'/controller/active-operation\.json":\s*' + _HEX)
NEEDLE_CONTRACT = re.compile(rb'/contract/accepted\.json":\s*' + _HEX)
NEEDLE_HARBOR = re.compile(rb'/harbor/task\.toml":\s*' + _HEX)
RE_REPAIR_BUDGET = re.compile(rb'"repair-budget/([^/"\n]+)/attempt-(\d+)/')
RE_QUALITY = re.compile(rb'"quality/([^/"\n]+)/attempt-(\d+)/result\.json"')
RE_VALIDATED = re.compile(rb'"validated/([^/"\n]+)/')
RUN_FILES = ("run.json", "submission.json", "terminal.json", "report.json", "conveyor.json", "controller/conveyor.json")
PATH_ITEM_STATUS = re.compile(r"^items/([^/]+)/status\.json$")
PATH_SESSION_STATUS = re.compile(r"^items/([^/]+)/sessions/([^/]+)/status\.json$")
PATH_ITEM_FILE = re.compile(r"^items/([^/]+)/(?:controller/active-operation\.json|contract/accepted\.json|harbor/task\.toml)$")


def _as_lines(raw: bytes) -> bytes:
    """Return a one-member-per-line rendering (compact manifests are re-rendered)."""
    if b"\n" in raw[:256]:
        return raw
    manifest = json.loads(raw)
    head = {k: v for k, v in manifest.items() if k != "files"}
    lines = [b"{"]
    lines.append(b'  "files": {')
    for rel, sha in (manifest.get("files") or {}).items():
        lines.append(f'    {json.dumps(rel)}: "{sha}",'.encode())
    lines.append(b"  },")
    for key, value in head.items():
        lines.append(f'  "{key}": {json.dumps(value)},'.encode())
    lines.append(b"}")
    return b"\n".join(lines)


def _needle_members(raw: bytes, needle: re.Pattern, suffix: bytes) -> Iterable[tuple[str, str]]:
    """Yield (member path, sha) for each line matched by ``needle`` (which starts at ``suffix``)."""
    for m in needle.finditer(raw):
        line_start = raw.rfind(b"\n", 0, m.start()) + 1
        quote = raw.find(b'"', line_start, m.start())
        if quote < 0:
            continue
        path = raw[quote + 1 : m.start()] + suffix
        yield path.decode("utf-8", "replace"), m.group(1).decode()


def index_manifest(raw: bytes) -> dict[str, Any]:
    """Extract only what the conveyor needs from a (possibly 200 MB) manifest."""
    raw = _as_lines(raw)
    created = RE_CREATED.search(raw[:8192]) or RE_CREATED.search(raw)
    tail = raw[-4_000_000:]
    finals = RE_FINAL.findall(tail)
    snapshots = RE_SNAPSHOT.findall(tail)
    items: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "status": None, "active_op": None, "contract": None, "sessions": {},
        "lowered": False, "repair_reserved": 0, "quality_attempts": 0, "validated": False,
    })
    for path, sha in _needle_members(raw, NEEDLE_CONTRACT, b"/contract/accepted.json"):
        m = PATH_ITEM_FILE.match(path)
        if m:
            items[m.group(1)]["contract"] = sha
    for path, sha in _needle_members(raw, NEEDLE_STATUS, b"/status.json"):
        m = PATH_ITEM_STATUS.match(path)
        if m:
            items[m.group(1)]["status"] = sha
    sessions: dict[str, dict[str, str]] = defaultdict(dict)
    for path, sha in _needle_members(raw, NEEDLE_STATUS, b"/status.json"):
        m = PATH_SESSION_STATUS.match(path)
        if m:
            sessions[m.group(1)][m.group(2)] = sha
    known = set(items)
    for name, found in sessions.items():
        if name in known:
            items[name]["sessions"] = found
    for path, sha in _needle_members(raw, NEEDLE_ACTIVE, b"/controller/active-operation.json"):
        m = PATH_ITEM_FILE.match(path)
        if m and m.group(1) in known:
            items[m.group(1)]["active_op"] = sha
    for path, _sha in _needle_members(raw, NEEDLE_HARBOR, b"/harbor/task.toml"):
        m = PATH_ITEM_FILE.match(path)
        if m and m.group(1) in known:
            items[m.group(1)]["lowered"] = True
    for m in RE_REPAIR_BUDGET.finditer(raw):
        name = m.group(1).decode()
        if name in known:
            items[name]["repair_reserved"] = max(items[name]["repair_reserved"], int(m.group(2)))
    for m in RE_QUALITY.finditer(raw):
        name = m.group(1).decode()
        if name in known:
            items[name]["quality_attempts"] = max(items[name]["quality_attempts"], int(m.group(2)))
    for m in RE_VALIDATED.finditer(raw):
        name = m.group(1).decode()
        if name in known:
            items[name]["validated"] = True
    run_files = {}
    for rel in RUN_FILES:
        m = re.search(rb'\n\s*"' + re.escape(rel.encode()) + rb'":\s*' + _HEX, raw)
        if m:
            run_files[rel] = m.group(1).decode()
    return {
        "index_version": INDEX_VERSION,
        "created_utc": created.group(1).decode() if created else None,
        "final": (finals[-1] == b"true") if finals else None,
        "snapshot_id": snapshots[-1].decode() if snapshots else None,
        "run_files": run_files,
        "items": dict(items),
        "member_count": raw.count(b'",\n') + 1,
    }


# --------------------------------------------------------------------------------------------
# Iris jobs
# --------------------------------------------------------------------------------------------


class ReadError(RuntimeError):
    """A read that must fail loud (never be mistaken for 'nothing there')."""


def parse_job_name(name: str) -> tuple[str | None, str | None]:
    match = JOB_RE.match(name)
    if not match:
        return None, None
    return match.group("base"), match.group("suffix")


def parse_job_list(text: str) -> list[dict[str, Any]]:
    jobs = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].startswith("/"):
            continue
        base, suffix = parse_job_name(parts[0])
        jobs.append({
            "name": parts[0], "state": parts[1].lower(), "submitted": parts[2],
            "reason": " ".join(parts[3:]), "base": base, "suffix": suffix,
        })
    return jobs


def fetch_jobs(prefix: str = JOB_PREFIX, limit: int = 5000, timeout: int = 180) -> list[dict[str, Any]]:
    command = ["uv", "run", "--frozen", "iris", "--cluster=marin", "job", "list",
               "--prefix", prefix, "--limit", str(limit)]
    try:
        proc = subprocess.run(command, cwd=MARIN_DIR, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ReadError(f"iris job list failed: {type(error).__name__}: {error}") from error
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["?"]
        raise ReadError(f"iris job list exit {proc.returncode}: {tail[0][:200]}")
    jobs = parse_job_list(proc.stdout)
    if not jobs:
        raise ReadError("iris job list returned zero jobs for the prefix (refusing to read as 'none live')")
    if len(jobs) >= limit:
        raise ReadError(f"iris job list returned {len(jobs)} rows == --limit {limit}: truncated")
    return jobs


def live_jobs_by_base(jobs: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    live: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        if not job.get("base"):
            continue
        state = job.get("state", "")
        if state in LIVE_JOB_STATES or state not in TERMINAL_JOB_STATES:
            live[job["base"]].append(job)
    return live


# --------------------------------------------------------------------------------------------
# S3 access with a content-addressed local cache
# --------------------------------------------------------------------------------------------


def _atomic_write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    if isinstance(data, str):
        data = data.encode()
    tmp.write_bytes(data)
    tmp.replace(path)


def _read_json_file(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


class S3Store:
    """Reads manifests and content-addressed objects; objects are cached forever by sha."""

    def __init__(self, root: str = S3_ROOT, cache_dir: Path = STATE_DIR / "cache",
                 workers: int = 32, manifest_workers: int = 8,
                 log: Callable[[str], None] | None = None) -> None:
        self.root = root
        self.cache_dir = cache_dir
        self.workers = workers
        self.manifest_workers = manifest_workers
        self.log = log or (lambda message: None)
        self.fs = None
        self.prefix = ""
        self.keys_fetched_at = time.time()
        self.stats: Counter = Counter()

    # -- credentials ---------------------------------------------------------------------
    def connect(self) -> None:
        if not os.environ.get("CW_KEY_ID") or not os.environ.get("CW_KEY_SECRET"):
            raise ReadError("CW_KEY_ID/CW_KEY_SECRET are not exported (the CoreWeave helper would no-op silently)")
        import fsspec  # noqa: PLC0415 - only needed against real S3
        from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415

        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["CW_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["CW_KEY_SECRET"]
        configure_coreweave_s3()
        try:
            import s3fs  # noqa: PLC0415

            s3fs.S3FileSystem.clear_instance_cache()
        except ImportError:
            pass
        self.fs, self.prefix = fsspec.core.url_to_fs(self.root.rstrip("/"))

    def refresh_keys(self) -> None:
        """Re-fetch the CoreWeave keys (they expire); never prints them."""
        values = {}
        for env, secret in (("CW_KEY_ID", "cw-object-storage-key-id"), ("CW_KEY_SECRET", "cw-object-storage-key-secret")):
            proc = subprocess.run(
                ["gcloud", "secrets", "versions", "access", "latest", f"--secret={secret}", "--project=hai-gcp-models"],
                capture_output=True, text=True, timeout=60,
            )
            value = proc.stdout.strip()
            if proc.returncode != 0 or not value:
                raise ReadError(f"key refresh failed for {secret} (exit {proc.returncode})")
            values[env] = value
        os.environ.update(values)
        self.keys_fetched_at = time.time()
        self.connect()

    # -- listing / manifests --------------------------------------------------------------
    def list_bases(self) -> list[str]:
        try:
            paths = self.fs.ls(self.prefix, detail=False, refresh=True)
        except Exception as error:  # noqa: BLE001
            raise ReadError(f"listing {self.root} failed: {type(error).__name__}: {error}") from error
        bases = sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in paths)
        if not bases:
            raise ReadError(f"empty listing under {self.root} (credentials or prefix wrong?)")
        return bases

    def _manifest_path(self, base: str) -> str:
        return f"{self.prefix}/{base}/_manifests/latest.json"

    def manifest_index(self, base: str) -> dict[str, Any] | None:
        """Index of <base>/_manifests/latest.json, reusing the cached index when the ETag matches."""
        path = self._manifest_path(base)
        try:
            try:
                info = self.fs.info(path, refresh=True)
            except TypeError:
                info = self.fs.info(path)
        except FileNotFoundError:
            return None
        etag = str(info.get("ETag") or info.get("etag") or "")
        cache_file = self.cache_dir / "manifests" / f"{base}.json"
        cached = _read_json_file(cache_file)
        if cached and etag and cached.get("etag") == etag and cached.get("index_version") == INDEX_VERSION:
            self.stats["manifest_cached"] += 1
            return cached
        for attempt in range(3):
            try:
                raw = self.fs.cat_file(path)
                break
            except FileNotFoundError:
                return None
            except Exception as error:  # noqa: BLE001 - a rewrite mid-read; retry once or twice
                if attempt == 2:
                    raise ReadError(f"{base}: manifest read failed: {type(error).__name__}: {error}") from error
        self.stats["manifest_bytes"] += len(raw)
        self.stats["manifest_fetched"] += 1
        index = index_manifest(raw)
        del raw
        index["etag"] = etag
        index["size"] = info.get("size")
        index["last_modified"] = iso(parse_time(info.get("LastModified")))
        _atomic_write(cache_file, json.dumps(index))
        return index

    def manifest_head(self, base: str) -> dict[str, Any] | None:
        try:
            info = self.fs.info(self._manifest_path(base), refresh=True)
        except FileNotFoundError:
            return None
        return {"etag": str(info.get("ETag") or ""), "last_modified": iso(parse_time(info.get("LastModified")))}

    def live_view(self, base: str) -> dict[str, Any] | None:
        """<base>/_live/conveyor.json (+ report.json), re-read only when its ETag changes."""
        path = f"{self.prefix}/{base}/_live/conveyor.json"
        try:
            info = self.fs.info(path, refresh=True)
        except FileNotFoundError:
            return None
        etag = str(info.get("ETag") or info.get("etag") or "")
        cache_file = self.cache_dir / "live" / f"{base}.json"
        cached = _read_json_file(cache_file)
        if cached and etag and cached.get("etag") == etag:
            self.stats["live_cached"] += 1
            return cached
        raw = self.fs.cat_file(path)
        conveyor = json.loads(raw)
        if not isinstance(conveyor, dict) or not isinstance(conveyor.get("items"), dict):
            raise ReadError(f"{base}: _live/conveyor.json has no items")
        report = None
        try:
            report = json.loads(self.fs.cat_file(f"{self.prefix}/{base}/_live/report.json"))
        except FileNotFoundError:
            pass
        self.stats["live_fetched"] += 1
        self.stats["live_bytes"] += len(raw)
        record = {"etag": etag, "last_modified": iso(parse_time(info.get("LastModified"))),
                  "conveyor": conveyor, "report": report}
        _atomic_write(cache_file, json.dumps(record))
        return record

    def raw_manifest(self, base: str) -> bytes:
        return self.fs.cat_file(self._manifest_path(base))

    # -- objects -------------------------------------------------------------------------
    def _obj_path(self, base: str, sha: str) -> str:
        return f"{self.prefix}/{base}/_objects/{sha}"

    def get_object(self, base: str, sha: str, *, want_data: bool = True) -> dict[str, Any]:
        """Return {"data": parsed JSON|None, "last_modified": iso|None} for an immutable object."""
        cache_file = self.cache_dir / "objects" / base / f"{sha}.json"
        cached = _read_json_file(cache_file)
        if cached is not None and (not want_data or cached.get("has_data")):
            self.stats["object_cached"] += 1
            return cached
        path = self._obj_path(base, sha)
        info = self.fs.info(path)
        record: dict[str, Any] = {"last_modified": iso(parse_time(info.get("LastModified"))), "has_data": False, "data": None}
        if want_data:
            raw = self.fs.cat_file(path)
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ReadError(f"{base}: object {sha[:12]} content does not match its address")
            record["data"] = json.loads(raw)
            record["has_data"] = True
            self.stats["object_bytes"] += len(raw)
        self.stats["object_fetched"] += 1
        _atomic_write(cache_file, json.dumps(record))
        return record

    def prune(self, live: dict[str, set[str]], max_age_s: float = 86400.0) -> int:
        """Drop cached objects no current manifest references once they are older than ``max_age_s``."""
        removed = 0
        root = self.cache_dir / "objects"
        cutoff = time.time() - max_age_s
        for base, keep in live.items():
            for path in (root / base).glob("*.json") if (root / base).is_dir() else ():
                try:
                    if path.stem not in keep and path.stat().st_mtime < cutoff:
                        path.unlink()
                        removed += 1
                except OSError:
                    continue
        return removed

    def get_bytes(self, base: str, sha: str) -> bytes:
        raw = self.fs.cat_file(self._obj_path(base, sha))
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ReadError(f"{base}: object {sha[:12]} content does not match its address")
        return raw


# --------------------------------------------------------------------------------------------
# Collection: S3 + Iris -> snapshots
# --------------------------------------------------------------------------------------------


@dataclass
class Collected:
    snapshots: dict[str, dict[str, Any]]
    jobs: list[dict[str, Any]] | None
    errors: list[str] = field(default_factory=list)
    timings: dict[str, Any] = field(default_factory=dict)
    collected_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _natural(text: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


def _source_count(base: str) -> int | None:
    """Fallback catalog size for a base from the live tree's (read-only) source files."""
    candidates = [BUILD_DIR / "construct-src-003" / f"{base}.json",
                  BUILD_DIR / "construct-src-seeded" / base / "accepted.json"]
    cache = STATE_DIR / "cache" / "source-counts.json"
    counts = _read_json_file(cache, {}) or {}
    for path in candidates:
        if path.is_file():
            key = f"{path}:{path.stat().st_size}:{int(path.stat().st_mtime)}"
            if key in counts:
                return counts[key]
            try:
                value = len(json.loads(path.read_text()))
            except (OSError, ValueError):
                return None
            counts[key] = value
            try:
                _atomic_write(cache, json.dumps(counts))
            except OSError:
                pass
            return value
    return None


def _newest(jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
    dated = [(parse_time(j.get("submitted")), j) for j in jobs]
    dated = [(t, j) for t, j in dated if t]
    return max(dated, key=lambda pair: pair[0])[1] if dated else None


def live_view_current(view: dict[str, Any], live_jobs: list[dict[str, Any]],
                      manifest_head: Callable[[], dict[str, Any] | None], slack_s: float = 1800.0) -> tuple[bool, str]:
    """Is ``_live/conveyor.json`` the freshest account of the run?  (False -> read the manifest.)

    With a live job it must have been written by that job (not before its submission); without
    one it must not be older than the manifest by more than a sync interval (an old-code job may
    have run after the last new-code job and never touched ``_live``)."""
    conveyor = view.get("conveyor") or {}
    updated = parse_time(conveyor.get("updated_at")) or parse_time(view.get("last_modified"))
    if live_jobs:
        newest = _newest(live_jobs)
        submitted = parse_time(newest.get("submitted")) if newest else None
        if updated and submitted and updated >= submitted - timedelta(seconds=60):
            return True, ""
        return False, f"_live view written {iso(updated)} predates live job submitted {iso(submitted)}"
    head = manifest_head()
    modified = parse_time((head or {}).get("last_modified"))
    if head is None or modified is None or updated is None or updated >= modified - timedelta(seconds=slack_s):
        return True, ""
    return False, f"manifest ({iso(modified)}) is newer than the _live view ({iso(updated)})"


def snapshot_from_live(base: str, view: dict[str, Any], live_jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """A run snapshot built only from ``_live/conveyor.json`` (capability-conveyor-v1 rows)."""
    conveyor = view.get("conveyor") or {}
    rows = conveyor.get("items") or {}
    updated = parse_time(conveyor.get("updated_at")) or parse_time(view.get("last_modified"))
    started = parse_time(conveyor.get("started_at"))
    run_name = None
    newest = _newest(live_jobs)
    if newest is not None and started and parse_time(newest.get("submitted")) and \
            started >= parse_time(newest.get("submitted")) - timedelta(seconds=60):
        run_name = newest["name"].rsplit("/", 1)[-1]
    items: dict[str, dict[str, Any]] = {}
    for key, row in rows.items():
        if not isinstance(row, dict):
            continue
        state, klass = row.get("state"), row.get("class")
        if state is None and klass in (None, "queued"):
            continue  # not started yet: counted as queued via accepted_count
        status = None
        if state is not None:
            status = {"key": key, "state": state, "issues": [row["issue"]] if row.get("issue") else [],
                      "state_since": row.get("state_since"), "updated_at": row.get("updated_at"),
                      "failure_stage": row.get("failure_stage")}
            if isinstance(row.get("wait"), dict):
                status["wait"] = row["wait"]
            if isinstance(row.get("wait_hold"), dict):
                status["wait_hold"] = row["wait_hold"]
        items[str(row.get("item") or key)] = {
            "status": status, "status_sha": f"{state}|{row.get('updated_at')}|{klass}",
            "status_written": iso(parse_time(row.get("updated_at"))), "active_op": None, "started_at": None,
            "sessions": 0, "session_info": [], "lowered": False, "repair_reserved": 0, "quality_attempts": 0,
            "validated": False, "has_contract": True, "live_row": row,
        }
    return {
        "base": base, "source": "live", "error": None, "items": items, "run": None, "terminal": None,
        "manifest": {"created_utc": iso(updated), "final": None, "etag": view.get("etag"),
                     "last_modified": view.get("last_modified"), "source": "_live/conveyor.json"},
        "submission": {"run_name": run_name, "concurrency": conveyor.get("concurrency"), "started_utc": iso(started)},
        "conveyor": {k: v for k, v in conveyor.items() if k != "items"},
        "report": view.get("report"),
        "accepted_count": len(rows),
    }


def collect(store: S3Store, *, jobs: list[dict[str, Any]] | None = None, fetch_job_list: bool = True,
            bases: list[str] | None = None, log: Callable[[str], None] | None = None,
            previous: "Collected | None" = None, refresh_manifests: bool = True,
            live_slack_s: float = 1800.0) -> Collected:
    """Read every base (thread pool) and the Iris job list.

    A base with a current ``_live/conveyor.json`` is read from it alone (a few KB).  Only bases
    without one fall back to the manifest (MBs to hundreds of MB); with ``refresh_manifests``
    False those reuse ``previous``'s snapshot instead, and an unchanged ETag never re-downloads."""
    log = log or (lambda message: None)
    errors: list[str] = []
    timings: dict[str, Any] = {}
    store.stats.clear()

    t0 = time.time()
    job_error = None
    if jobs is None and fetch_job_list:
        try:
            jobs = fetch_jobs()
        except ReadError as error:
            job_error = str(error)
            errors.append(f"jobs: {error}")
    timings["jobs_s"] = round(time.time() - t0, 1)

    t1 = time.time()
    all_bases = store.list_bases()
    if bases:
        wanted = set(bases)
        all_bases = [b for b in all_bases if b in wanted or any(fnmatch.fnmatch(b, pat) for pat in wanted)]
    live_jobs = live_jobs_by_base(jobs or [])
    indexes: dict[str, dict | None] = {}
    live_snaps: dict[str, dict[str, Any]] = {}
    reused: dict[str, dict[str, Any]] = {}
    fallback_why: dict[str, str] = {}

    def route(base: str) -> tuple[str, str, Any, str]:
        why = ""
        if hasattr(store, "live_view"):
            view = store.live_view(base)
            if view is not None:
                ok, why = live_view_current(view, live_jobs.get(base, []),
                                            lambda: store.manifest_head(base) if hasattr(store, "manifest_head") else None,
                                            live_slack_s)
                if ok:
                    return base, "live", snapshot_from_live(base, view, live_jobs.get(base, [])), ""
        prev = previous.snapshots.get(base) if previous is not None else None
        if not refresh_manifests and prev is not None and prev.get("source") != "live" and not prev.get("error"):
            return base, "reused", prev, why
        return base, "manifest", store.manifest_index(base), why

    with concurrent.futures.ThreadPoolExecutor(store.manifest_workers) as pool:
        futures = {pool.submit(route, base): base for base in all_bases}
        for future in concurrent.futures.as_completed(futures):
            base = futures[future]
            try:
                _, kind, value, why = future.result()
            except Exception as error:  # noqa: BLE001
                indexes[base] = {"error": f"{type(error).__name__}: {error}"}
                errors.append(f"{base}: read: {type(error).__name__}: {error}")
                continue
            if why:
                fallback_why[base] = why
            if kind == "live":
                live_snaps[base] = value
            elif kind == "reused":
                reused[base] = value
            else:
                indexes[base] = value
    timings["manifest_s"] = round(time.time() - t1, 1)
    timings["live_views"] = len(live_snaps)
    timings["live_fetched"] = store.stats["live_fetched"]
    timings["live_kb"] = round(store.stats["live_bytes"] / 1e3, 1)
    timings["manifest_reused"] = len(reused)
    timings["manifests"] = len(indexes)
    timings["manifest_fetched"] = store.stats["manifest_fetched"]
    timings["manifest_cached"] = store.stats["manifest_cached"]
    timings["manifest_mb"] = round(store.stats["manifest_bytes"] / 1e6, 1)

    # Object fetches: run-level files, statuses, active operations (data) + contract (LastModified).
    t2 = time.time()
    requests: list[tuple[str, str, str, str | None, bool]] = []  # (base, kind, sha, item, want_data)
    for base, index in indexes.items():
        if not index or index.get("error"):
            continue
        for rel, sha in index.get("run_files", {}).items():
            if rel.endswith(("report.json",)):
                continue  # potentially large; not needed for the conveyor
            requests.append((base, rel, sha, None, True))
        for name, rec in index.get("items", {}).items():
            if rec.get("status"):
                requests.append((base, "status", rec["status"], name, True))
            if rec.get("active_op"):
                requests.append((base, "active_op", rec["active_op"], name, True))
            if rec.get("contract"):
                requests.append((base, "contract", rec["contract"], name, False))
            # Builder-session statuses: the only progress signal for a first attempt or a
            # pending_build continuation.  Cached by sha, so finished sessions cost nothing.
            for session, sha in (rec.get("sessions") or {}).items():
                requests.append((base, f"session:{session}", sha, name, True))
    results: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    object_errors = Counter()
    with concurrent.futures.ThreadPoolExecutor(store.workers) as pool:
        futures = {pool.submit(store.get_object, base, sha, want_data=want): (base, kind, name)
                   for base, kind, sha, name, want in requests}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as error:  # noqa: BLE001
                results[key] = {"error": f"{type(error).__name__}: {error}"}
                object_errors[key[0]] += 1
    for base, count in sorted(object_errors.items()):
        errors.append(f"{base}: {count} object read(s) failed")
    if hasattr(store, "prune"):
        referenced: dict[str, set[str]] = defaultdict(set)
        for base, _kind, sha, _name, _want in requests:
            referenced[base].add(sha)
        timings["cache_pruned"] = store.prune(referenced)
    timings["object_s"] = round(time.time() - t2, 1)
    timings["objects"] = len(requests)
    timings["object_fetched"] = store.stats["object_fetched"]

    snapshots: dict[str, dict[str, Any]] = {}
    for base in all_bases:
        if base in live_snaps:
            snapshots[base] = live_snaps[base]
            continue
        if base in reused:
            snapshots[base] = {**reused[base], "source": "reused",
                               "reused_from": reused[base].get("reused_from") or iso(previous.collected_at if previous else None)}
            continue
        index = indexes.get(base)
        snap: dict[str, Any] = {"base": base, "source": "manifest", "manifest": None, "error": None, "items": {},
                                "run": None, "submission": None, "terminal": None, "conveyor": None,
                                "accepted_count": None}
        if index is None:
            snapshots[base] = snap
            continue
        if index.get("error"):
            snap["error"] = index["error"]
            snapshots[base] = snap
            continue
        if base in fallback_why:
            snap["live_fallback"] = fallback_why[base]
        snap["manifest"] = {k: index.get(k) for k in ("created_utc", "final", "snapshot_id", "etag", "size", "last_modified", "member_count")}
        for rel in index.get("run_files", {}):
            rec = results.get((base, rel, None)) or {}
            short = rel.rsplit("/", 1)[-1].removesuffix(".json")
            if short in ("run", "submission", "terminal", "conveyor") and rec.get("data") is not None:
                snap[short] = rec["data"]
        run = snap["run"] or {}
        snap["accepted_count"] = run.get("accepted_count") if isinstance(run.get("accepted_count"), int) else _source_count(base)
        for name, rec in index.get("items", {}).items():
            status_rec = results.get((base, "status", name)) or {}
            active_rec = results.get((base, "active_op", name)) or {}
            contract_rec = results.get((base, "contract", name)) or {}
            status = status_rec.get("data")
            session_info = []
            for session in sorted(rec.get("sessions") or {}, key=_natural):
                srec = results.get((base, f"session:{session}", name)) or {}
                data = srec.get("data") if isinstance(srec.get("data"), dict) else {}
                session_info.append({"session": session, "status": data.get("status") or data.get("state"),
                                     "written": srec.get("last_modified")})
            if rec.get("status") and status_rec.get("error"):
                status = {"state": "__unreadable__", "issues": [status_rec["error"]]}
            snap["items"][name] = {
                "status": status,
                "status_sha": rec.get("status"),
                "status_written": status_rec.get("last_modified"),
                "active_op": active_rec.get("data"),
                "started_at": contract_rec.get("last_modified"),
                "sessions": len(rec.get("sessions") or {}),
                "session_info": session_info,
                "lowered": rec.get("lowered", False),
                "repair_reserved": rec.get("repair_reserved", 0),
                "quality_attempts": rec.get("quality_attempts", 0),
                "validated": rec.get("validated", False),
                "has_contract": bool(rec.get("contract")),
            }
        snapshots[base] = snap
    collected = Collected(snapshots=snapshots, jobs=jobs, errors=errors, timings=timings)
    if job_error:
        collected.timings["jobs_error"] = job_error
    log(f"reads: jobs {timings['jobs_s']}s; live views {timings['live_views']} ({timings['live_fetched']} fetched, "
        f"{timings['live_kb']} KB); reused {timings['manifest_reused']}; manifests {timings['manifests']} "
        f"({timings['manifest_fetched']} fetched {timings['manifest_mb']} MB, {timings['manifest_cached']} cached) "
        f"{timings['manifest_s']}s; objects {timings['objects']} ({timings['object_fetched']} new) {timings['object_s']}s")
    return collected


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------


def _item_timing(item: dict[str, Any], status: dict | None) -> tuple[datetime | None, str]:
    """When the item's current state began (best available evidence) and where that came from."""
    if status:
        since = parse_time(status.get("state_since"))
        if since:
            return since, "state_since"
        transitions = status.get("transitions")
        if isinstance(transitions, list) and transitions:
            last = transitions[-1]
            if isinstance(last, dict) and parse_time(last.get("at")):
                return parse_time(last.get("at")), "transition"
        written = parse_time(item.get("status_written"))
        if written:
            return written, "status_written"
        updated = parse_time(status.get("updated_at"))
        if updated:
            return updated, "updated_at"
    started = parse_time(item.get("started_at"))
    if started:
        return started, "item_started"
    return None, "unknown"


def _activity(item: dict[str, Any], status: dict | None) -> tuple[str | None, datetime | None]:
    row = item.get("live_row")
    if isinstance(row, dict):
        act = row.get("activity")
        if isinstance(act, dict):
            detail = f" {act['kind']}" if act.get("kind") else ""
            return f"{act.get('step') or '?'}{detail}", parse_time(act.get("since"))
        return None, None
    if status and isinstance(status.get("activity"), dict):
        act = status["activity"]
        return str(act.get("step") or "?"), parse_time(act.get("since"))
    op = item.get("active_op")
    if isinstance(op, dict) and op.get("state") == "active":
        name = "repair" if op.get("operation") == "construction_repair" else str(op.get("operation") or "op")
        return f"{name}#{op.get('attempt', '?')}", parse_time(op.get("started_at"))
    if status is None or status.get("state") == "pending_build":
        info = item.get("session_info") or []
        written = [parse_time(s.get("written")) for s in info]
        last = max((t for t in written if t), default=None)
        if item.get("lowered") and status is None:
            return "lowered: calibration/controls", last
        pending = [s for s in info if s.get("status") != "complete"]
        if not info:
            return "build (no session status yet)", None
        if pending:
            return f"build {pending[0]['session']} ({len(info) - len(pending)}/{len(info)} done)", last
        return f"post-build gates ({len(info)} sessions done)", last
    return None, None


def classify_item(base: str, name: str, item: dict[str, Any], *, live: bool,
                  job_start: datetime | None = None) -> dict[str, Any]:
    """One item's conveyor record.  ``live``/``job_start`` describe the base's current job."""
    status = item.get("status")
    if status is not None and not isinstance(status, dict):
        status = {"state": "__unreadable__", "issues": ["status.json is not an object"]}
    state = status.get("state") if status else None
    unreadable = state == "__unreadable__"
    known = status is None or is_known_state(state)
    stage = "unknown" if (unreadable or not known) else state_stage(status)
    rclass, rreason = ("maybe", "unreadable status") if unreadable else resume_class(status)
    row = item.get("live_row") if isinstance(item.get("live_row"), dict) else None
    klass = row.get("class") if row else None
    held = held_in_force(status, row)
    if row is not None:
        # capability-conveyor-v1: the job's own scheduler says what is terminal.  A held row
        # (wait_hold with the state preserved) is terminal for that job only: a relaunch
        # re-enters it, so it is not a finished item.
        if state == "quality_accepted" or (klass == "terminal" and held is None):
            rclass, rreason = "terminal", f"conveyor: terminal ({state})"
        elif held is not None:
            rclass, rreason = "progress", _held_reason(held)
        else:
            rclass, rreason = "progress", f"conveyor: {klass}"
    elif held is not None and not unreadable:
        rclass, rreason = "progress", _held_reason(held)
    since, since_source = _item_timing(item, status)
    activity, activity_since = _activity(item, status)
    written = parse_time(item.get("status_written")) or parse_time((status or {}).get("updated_at"))
    # Where the item is within the current job's single pass over its run (synthesize() calls
    # synthesize_one exactly once per item per job): returned / touched / pending.
    pass_state = None
    if live and job_start is not None:
        if written and written >= job_start:
            pass_state = "returned"
        elif activity_since and activity_since >= job_start:
            pass_state = "touched"
        else:
            pass_state = "pending"
    if isinstance((status or {}).get("activity"), dict):
        pass_state = "touched" if live else pass_state  # explicit activity = in progress now
    if row is not None and live:
        pass_state = {"queued": "pending", "running": "touched", "waiting": "returned"}.get(klass, pass_state)
    if held is not None and not unreadable and rclass != "terminal":
        disposition = "held"  # non-terminal: the supervisor relaunches its base
    elif unreadable or not known:
        disposition = "unknown"
    elif rclass == "terminal":
        disposition = "accepted" if state == "quality_accepted" else "rejected"
    elif rclass == "frozen":
        disposition = "held"
    elif not live:
        disposition = "parked"
    else:
        disposition = "waiting" if pass_state == "returned" else "active"
    budget = (status or {}).get("repair_budget") if isinstance((status or {}).get("repair_budget"), dict) else {}
    wait = (status or {}).get("wait") if isinstance((status or {}).get("wait"), dict) else None
    reason = reason_of(status)
    fstage = None
    if status and (state == "failed" or disposition in ("rejected", "held")):
        fstage = failure_stage(status)
    images = (status or {}).get("custom_images") if isinstance((status or {}).get("custom_images"), dict) else {}
    return {
        "run": base,
        "item": name,
        "state": "(none)" if status is None else state,
        "stage": stage,
        "disposition": disposition,
        "pass_state": pass_state,
        "resume": rclass,
        "resume_reason": rreason,
        "since": iso(since),
        "since_source": since_source,
        "status_written": item.get("status_written"),
        "status_sha": item.get("status_sha"),
        "started_at": item.get("started_at"),
        "activity": activity,
        "activity_since": iso(activity_since),
        "wait_kind": wait.get("kind") if wait else None,
        "wait_attempts": wait.get("attempts") if wait else None,
        "next_attempt_at": wait.get("next_attempt_at") if wait else None,
        "wait_deadline": wait.get("deadline") if wait else None,
        "repairs_used": budget.get("used", item.get("repair_reserved") or None),
        "repairs_max": budget.get("max"),
        "repairs_reserved": item.get("repair_reserved") or 0,
        "quality_attempts": item.get("quality_attempts") or 0,
        "image_state": images.get("state"),
        "image_reason": images.get("reason"),
        "issue": issue_head(reason) if reason else None,
        "failure_stage": fstage,
        "failure_stage_raw": (status or {}).get("failure_stage"),
        "conveyor_class": klass,
        "calls": row.get("calls") if row else None,
        "failure_class": normalize_reason(reason) if (reason and fstage) else None,
        "reason_class": normalize_reason(reason) if reason else None,
        "terminal_disposition": (status or {}).get("terminal_disposition"),
        "transitions": (status or {}).get("transitions") if isinstance((status or {}).get("transitions"), list) else None,
        "acceptance_resumed": bool((status or {}).get("acceptance_resumed")),
        "sessions": item.get("sessions", 0),
        "anomalies": [],
    }


def _stuck_threshold_s(record: dict[str, Any], thresholds: dict[str, float]) -> float:
    act = record.get("activity") or ""
    if act.startswith("repair"):
        return thresholds.get("repair", thresholds.get("unknown", 6.0)) * 3600
    return thresholds.get(record["stage"], thresholds.get("unknown", 6.0)) * 3600


def current_job_start(snap: dict[str, Any], live_jobs: list[dict[str, Any]]) -> tuple[datetime | None, str | None]:
    """Start of the base's current job: its own submission.json when synced, else Iris submit time."""
    if not live_jobs:
        return None, None
    names = {j["name"].rsplit("/", 1)[-1] for j in live_jobs}
    submission = snap.get("submission") or {}
    started = parse_time(submission.get("started_utc"))
    if started and submission.get("run_name") in names:
        return started, "submission.json"
    submitted = [parse_time(j.get("submitted")) for j in live_jobs]
    submitted = [t for t in submitted if t]
    return (max(submitted), "iris submitted") if submitted else (None, None)


def build_model(collected: Collected, *, now: datetime | None = None,
                thresholds: dict[str, float] | None = None,
                stale_min: float = STALE_SNAPSHOT_MIN,
                stall_hours: float = 2.0,
                history: "History | None" = None) -> dict[str, Any]:
    """Pure function of the collected snapshots + jobs (+ optional history); the full conveyor model."""
    now = now or datetime.now(UTC)
    thresholds = {**DEFAULT_STUCK_HOURS, **(thresholds or {})}
    jobs = collected.jobs
    jobs_known = jobs is not None
    live = live_jobs_by_base(jobs or [])
    items: list[dict[str, Any]] = []
    bases: dict[str, dict[str, Any]] = {}
    anomalies: list[dict[str, Any]] = []

    for base in sorted(set(collected.snapshots) | set(live)):
        snap = collected.snapshots.get(base) or {"base": base, "manifest": None, "items": {}, "error": None}
        base_live = live.get(base, [])
        is_live = bool(base_live) if jobs_known else True  # unknown job state: never claim PARKED
        job_start, job_start_source = current_job_start(snap, base_live)
        manifest = snap.get("manifest") or None
        created = parse_time((manifest or {}).get("created_utc"))
        observed = created or now
        submission = snap.get("submission") or {}
        terminal = snap.get("terminal") or {}
        sub_started = parse_time(submission.get("started_utc"))
        term_finished = parse_time(terminal.get("finished_utc"))
        last_job_ended = bool(term_finished and (not sub_started or term_finished >= sub_started))
        conveyor = snap.get("conveyor") if isinstance(snap.get("conveyor"), dict) else None
        conveyor_items = conveyor.get("items") if conveyor and isinstance(conveyor.get("items"), dict) else {}
        newest_submit = max((t for t in (parse_time(j.get("submitted")) for j in base_live) if t), default=None)
        base_rec: dict[str, Any] = {
            "run": base,
            "live": is_live if jobs_known else None,
            "live_jobs": [j["name"] for j in base_live],
            "live_job_states": sorted({j["state"] for j in base_live}),
            "job_start": iso(job_start),
            "job_start_source": job_start_source,
            "manifest_created": iso(created),
            "manifest_age_s": (now - created).total_seconds() if created else None,
            "manifest_final": (manifest or {}).get("final"),
            "manifest_error": snap.get("error"),
            "submission": {k: submission.get(k) for k in ("run_name", "concurrency", "started_utc", "tier")} if submission else None,
            "terminal": terminal or None,
            "last_job_ended": last_job_ended,
            "accepted_count": snap.get("accepted_count"),
            "conveyor": {k: v for k, v in conveyor.items() if k != "items"} if conveyor else None,
            "source": snap.get("source"),
            "reused_from": snap.get("reused_from"),
            "live_fallback": snap.get("live_fallback"),
            "report_state": (snap.get("report") or {}).get("state") if isinstance(snap.get("report"), dict) else None,
            "started": 0,
            "queued": 0,
            "counts": Counter(),
            "stages": Counter(),
            "pass": Counter(),
            "last_change": None,
            "fingerprint": None,
            "anomalies": [],
        }
        if snap.get("error"):
            anomalies.append({"kind": "ERROR", "run": base, "detail": f"manifest unreadable: {snap['error']}"})
        if manifest is None and not snap.get("error"):
            if base_live:
                waited = (now - newest_submit).total_seconds() if newest_submit else None
                if waited is None or waited > stale_min * 60:
                    anomalies.append({"kind": "NO_SNAPSHOT", "run": base,
                                      "detail": f"live job {base_live[0]['name'].rsplit('/', 1)[-1]} has no manifest after {fmt_age(waited)}"})
            else:
                anomalies.append({"kind": "NO_SNAPSHOT", "run": base, "detail": "no manifest and no live job"})
        if manifest is not None and base_live and created:
            # A snapshot reused from an earlier cycle is judged as of when it was read (no flapping).
            ref_now = parse_time(snap.get("reused_from")) or now
            age_min = (ref_now - created).total_seconds() / 60
            fresh_job = newest_submit is not None and (ref_now - newest_submit).total_seconds() / 60 < stale_min
            if age_min > stale_min and not fresh_job:
                anomalies.append({"kind": "STALE_SNAPSHOT", "run": base,
                                  "detail": f"manifest {age_min:.0f}m old (created {iso(created)}) with live job {base_live[0]['name'].rsplit('/', 1)[-1]}"})
        fingerprint = hashlib.sha256()
        base_items: list[dict[str, Any]] = []
        last_change_base: datetime | None = None
        for name, item in sorted((snap.get("items") or {}).items()):
            if not item.get("has_contract", True) and item.get("status") is None:
                continue
            if name in conveyor_items and isinstance(conveyor_items[name], dict) and isinstance(item.get("status"), dict):
                # Run-level conveyor.json entries refine (never replace) the per-item status.
                merged = dict(item["status"])
                for key in ("activity", "wait", "state_since", "updated_at", "failure_stage", "transitions"):
                    if key in conveyor_items[name] and key not in merged:
                        merged[key] = conveyor_items[name][key]
                item = {**item, "status": merged}
            record = classify_item(base, name, item, live=is_live, job_start=job_start)
            fingerprint.update(f"{name}={item.get('status_sha')}\n".encode())
            if history is not None:
                seen_at = history.touch(f"item:{base}/{name}", f"{record['state']}|{record['activity']}", now)
                record["first_seen"] = iso(seen_at)
                if record["since"] is None:
                    record["since"], record["since_source"] = iso(seen_at), "first_seen"
            since = parse_time(record["since"])
            activity_since = parse_time(record["activity_since"])
            last_change = max((t for t in (since, activity_since, parse_time(record["status_written"])) if t), default=None)
            if last_change and (last_change_base is None or last_change > last_change_base):
                last_change_base = last_change
            record["unchanged_s"] = (observed - last_change).total_seconds() if last_change else None
            record["age_s"] = (now - since).total_seconds() if since else None
            if record["disposition"] == "unknown":
                record["anomalies"].append("UNKNOWN_STATE")
                anomalies.append({"kind": "UNKNOWN_STATE", "run": base, "item": name, "detail": f"state={record['state']!r}"})
            if not is_live and record["disposition"] in NON_TERMINAL:
                record["anomalies"].append("PARKED")
            # STUCK: the current job has touched the item (activity this pass) but nothing moved
            # for longer than the stage's threshold.  Items that already returned this pass are
            # 'waiting' (they only move on the next pass), and untouched ones are 'pending'.
            if is_live and jobs_known and record["disposition"] == "active" and record["pass_state"] == "touched":
                act_since = activity_since or since
                idle = (observed - act_since).total_seconds() if act_since else None
                limit = _stuck_threshold_s(record, thresholds)
                if idle is not None and idle > limit:
                    record["anomalies"].append("STUCK")
                    anomalies.append({"kind": "STUCK", "run": base, "item": name, "stage": record["stage"],
                                      "detail": f"{record['activity'] or record['state']} idle {fmt_age(idle)} > {fmt_age(limit)}"})
            nxt = parse_time(record.get("next_attempt_at"))
            if is_live and nxt and (observed - nxt).total_seconds() > _stuck_threshold_s(record, thresholds):
                if "STUCK" not in record["anomalies"]:
                    record["anomalies"].append("STUCK")
                    anomalies.append({"kind": "STUCK", "run": base, "item": name, "stage": record["stage"],
                                      "detail": f"wait {record.get('wait_kind')} next attempt overdue since {iso(nxt)}"})
            items.append(record)
            base_items.append(record)
            base_rec["started"] += 1
            base_rec["counts"][record["disposition"]] += 1
            base_rec["stages"][record["stage"]] += 1
            if record["pass_state"]:
                base_rec["pass"][record["pass_state"]] += 1
        base_rec["fingerprint"] = fingerprint.hexdigest()[:16]
        base_rec["last_change"] = iso(last_change_base)
        total = snap.get("accepted_count")
        if isinstance(total, int):
            base_rec["queued"] = max(0, total - base_rec["started"])
            if base_rec["queued"]:
                base_rec["counts"]["queued"] += base_rec["queued"]
                base_rec["stages"]["queued"] += base_rec["queued"]
        non_terminal = sum(base_rec["counts"][d] for d in NON_TERMINAL)
        base_rec["non_terminal"] = non_terminal
        base_rec["progressable"] = sum(base_rec["counts"][d] for d in ("queued", "active", "waiting", "parked", "unknown"))
        base_rec["all_terminal"] = manifest is not None and non_terminal == 0 and isinstance(total, int)
        if jobs_known and not is_live and non_terminal and manifest is not None:
            parked_items = sorted(r["item"] for r in base_items if "PARKED" in r["anomalies"])
            parked_since = history.touch(f"parked:{base}", "parked", now) if history is not None else None
            anomalies.append({"kind": "PARKED", "run": base, "items": parked_items, "since": iso(parked_since),
                              "detail": f"{non_terminal} non-terminal (started {len(parked_items)}, queued {base_rec['queued']}, "
                                        f"held {base_rec['counts']['held']}) and no live job"
                                        + (f"; first seen parked {iso(parked_since)}" if parked_since else "")})
        # STALLED: a live job past its warm-up whose run shows no item change since the job began.
        if jobs_known and is_live and job_start and manifest is not None and non_terminal:
            warm = (observed - job_start).total_seconds()
            if warm > stall_hours * 3600 and (last_change_base is None or last_change_base < observed - timedelta(hours=stall_hours)):
                anomalies.append({"kind": "STALLED", "run": base,
                                  "detail": f"no item status/activity change for >{stall_hours:g}h (last {iso(last_change_base) or 'never'}; "
                                            f"job started {iso(job_start)})"})
        base_rec["counts"] = dict(base_rec["counts"])
        base_rec["stages"] = dict(base_rec["stages"])
        base_rec["pass"] = dict(base_rec["pass"])
        bases[base] = base_rec
    for anomaly in anomalies:
        if anomaly.get("run") in bases:
            bases[anomaly["run"]]["anomalies"].append(anomaly["kind"])

    funnel = {key: Counter() for key in STAGE_KEYS}
    for base_rec in bases.values():
        if base_rec["queued"]:
            funnel["queued"]["queued"] += base_rec["queued"]
    for record in items:
        stage = record["stage"] if record["stage"] in funnel else "unknown"
        funnel[stage][record["disposition"]] += 1
    funnel_rows = []
    for key, label in STAGES:
        counts = funnel[key]
        funnel_rows.append({"stage": key, "label": label, "total": sum(counts.values()), **{d: counts.get(d, 0) for d in DISPOSITIONS}})

    failures_by_stage: dict[str, Counter] = defaultdict(Counter)
    histogram: dict[str, Counter] = defaultdict(Counter)
    for record in items:
        if record["disposition"] in ("accepted", "queued"):
            continue
        if record["failure_stage"]:
            if record["disposition"] == "rejected":
                bucket = "rejected"
            elif record["state"] == "failed":
                bucket = "failed_held" if record["disposition"] == "held" else "failed_repairable"
            else:
                bucket = "held"
            failures_by_stage[record["failure_stage"]][bucket] += 1
        if record["reason_class"] and (record["disposition"] in ("rejected", "held") or record["state"] == "failed"):
            histogram[record["failure_stage"] or record["stage"]][record["reason_class"]] += 1

    throughput = compute_throughput(items, now=now, history=history)
    if history is not None:
        history.save_first_seen()
    disposition_totals = Counter()
    for base_rec in bases.values():
        disposition_totals.update(base_rec["counts"])
    kinds = Counter(a["kind"] for a in anomalies)
    return {
        "run": RUN,
        "generated_at": iso(now),
        "collected_at": iso(collected.collected_at),
        "timings": collected.timings,
        "errors": list(collected.errors),
        "jobs_known": jobs_known,
        "thresholds_h": thresholds,
        "totals": {
            "bases": len(bases),
            "live_bases": sum(1 for b in bases.values() if b["live"]),
            "catalog": sum(b["accepted_count"] or 0 for b in bases.values()),
            "started": sum(b["started"] for b in bases.values()),
            **{d: disposition_totals.get(d, 0) for d in DISPOSITIONS},
            "pass": dict(sum((Counter(b["pass"]) for b in bases.values()), Counter())),
        },
        "funnel": funnel_rows,
        "failures_by_stage": {stage: dict(c) for stage, c in sorted(failures_by_stage.items(), key=lambda kv: STAGE_INDEX.get(kv[0], 99))},
        "failure_histogram": {stage: dict(c.most_common()) for stage, c in sorted(histogram.items(), key=lambda kv: STAGE_INDEX.get(kv[0], 99))},
        "throughput": throughput,
        "anomaly_counts": dict(kinds),
        "anomalies": anomalies,
        "bases": bases,
        "items": items,
    }


# --------------------------------------------------------------------------------------------
# Throughput: transitions when present, else diff against the saved previous snapshot
# --------------------------------------------------------------------------------------------


class History:
    """Previous-snapshot diffing + event log in ops/state (atomic writes; last writer wins)."""

    def __init__(self, state_dir: Path = STATE_DIR, *, persist: bool = True) -> None:
        self.state_dir = state_dir
        self.persist = persist
        self.prev_path = state_dir / "prev-items.json"
        self.events_path = state_dir / "events.jsonl"
        self.first_seen_path = state_dir / "first-seen.json"
        self.prev = _read_json_file(self.prev_path)
        self.first_seen: dict[str, dict[str, str]] = _read_json_file(self.first_seen_path, {}) or {}
        self._touched: set[str] = set()
        self.events: list[dict[str, Any]] = []
        try:
            for line in self.events_path.read_text().splitlines():
                try:
                    self.events.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            pass

    def touch(self, key: str, signature: str, now: datetime) -> datetime:
        """First time this tool saw ``key`` with ``signature`` (reset when the signature changes)."""
        self._touched.add(key)
        rec = self.first_seen.get(key)
        if not rec or rec.get("sig") != signature:
            rec = {"sig": signature, "at": iso(now)}
            self.first_seen[key] = rec
        return parse_time(rec["at"]) or now

    def save_first_seen(self) -> None:
        # Forget keys not seen this cycle so a returning condition starts a fresh clock.
        self.first_seen = {k: v for k, v in self.first_seen.items() if k in self._touched}
        self._touched = set()
        if self.persist:
            _atomic_write(self.first_seen_path, json.dumps(self.first_seen))

    def update(self, items: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
        """Record accepted/failed/rejected transitions since the previous snapshot; return new events."""
        seen_keys = {(e.get("key"), e.get("kind"), e.get("sha")) for e in self.events}
        new: list[dict[str, Any]] = []
        prev_items = (self.prev or {}).get("items") or {}
        prev_at = parse_time((self.prev or {}).get("at"))
        prev_bases = set((self.prev or {}).get("bases") or {k.split("/", 1)[0] for k in prev_items})
        for record in items:
            key = f"{record['run']}/{record['item']}"
            old = prev_items.get(key) or {}
            # A base this history has never seen is seeded from status write times, never diffed.
            seeded = record["run"] not in prev_bases
            kinds = []
            if record["disposition"] == "accepted" and old.get("disposition") != "accepted":
                kinds.append("accepted")
            if record["disposition"] == "rejected" and old.get("disposition") != "rejected":
                kinds.append("rejected")
            if old.get("disposition") == "accepted" and record["disposition"] != "accepted" and not seeded:
                kinds.append("unaccepted")  # an acceptance lost (e.g. not preserved across a relaunch)
            if record["state"] == "failed" and (old.get("state") != "failed" or old.get("written") != record.get("status_written")):
                kinds.append("failed")
            for kind in kinds:
                written = parse_time(record.get("status_written"))
                if seeded:
                    # First run: only statuses written in the last 6h, and never resumed acceptances.
                    if not written or (now - written) > timedelta(hours=6) or (kind == "accepted" and record.get("acceptance_resumed")):
                        continue
                    at, source = written, "seed:status_written"
                else:
                    at = written if (written and prev_at and written >= prev_at - timedelta(minutes=5)) else now
                    source = "diff"
                if kind == "unaccepted":
                    at, source = now, "diff"
                sig = (key, kind, record.get("status_written"))
                if sig in seen_keys:
                    continue
                seen_keys.add(sig)
                event = {"at": iso(at), "key": key, "kind": kind, "stage": record.get("failure_stage") or record["stage"],
                         "sha": record.get("status_written"), "source": source}  # sha: status write time (dedupe key)
                if kind == "unaccepted":
                    event["now_state"] = record["state"]
                new.append(event)
        self.events.extend(new)
        cutoff = now - timedelta(days=7)
        self.events = [e for e in self.events if (parse_time(e.get("at")) or now) >= cutoff]
        current = {f"{r['run']}/{r['item']}": {"disposition": r["disposition"], "state": r["state"], "written": r.get("status_written")}
                   for r in items}
        seen_bases = {r["run"] for r in items}
        # Keep entries of bases not collected this time (a filtered run must not erase them).
        kept = {k: v for k, v in prev_items.items() if k.split("/", 1)[0] not in seen_bases}
        self.prev = {"at": iso(now), "bases": sorted(prev_bases | seen_bases), "items": {**kept, **current}}
        if self.persist:
            _atomic_write(self.prev_path, json.dumps(self.prev))
            _atomic_write(self.events_path, "".join(json.dumps(e) + "\n" for e in self.events))
        return new

    def since(self) -> datetime | None:
        times = [parse_time(e.get("at")) for e in self.events]
        times = [t for t in times if t]
        return min(times) if times else None


def compute_throughput(items: list[dict[str, Any]], *, now: datetime, history: History | None) -> dict[str, Any]:
    windows = {"1h": timedelta(hours=1), "6h": timedelta(hours=6)}
    out: dict[str, Any] = {}
    have_transitions = any(r.get("transitions") for r in items)
    events: list[tuple[datetime, str]] = []
    if have_transitions:
        for r in items:
            for t in r.get("transitions") or []:
                if not isinstance(t, dict):
                    continue
                at = parse_time(t.get("at"))
                state = t.get("state")
                if not at:
                    continue
                if state == "quality_accepted":
                    events.append((at, "accepted"))
                elif state == "failed":
                    events.append((at, "failed"))
                elif state in ("rejected", "terminal_rejected"):
                    events.append((at, "rejected"))
            if r["disposition"] == "rejected" and not any(
                isinstance(t, dict) and t.get("state") in ("rejected", "terminal_rejected") for t in r.get("transitions") or []
            ):
                at = parse_time(r.get("since"))
                if at:
                    events.append((at, "rejected"))
        out["source"] = "transitions"
    elif history is not None:
        history.update(items, now)
        events = [(parse_time(e["at"]), e["kind"]) for e in history.events if parse_time(e.get("at"))]
        out["source"] = "diff" if history.prev and history.events else "diff (no history yet)"
        out["history_since"] = iso(history.since())
    else:
        out["source"] = "none"
    for label, delta in windows.items():
        for kind in ("accepted", "failed", "rejected", "unaccepted"):
            out[f"{kind}_{label}"] = sum(1 for at, k in events if k == kind and now - at <= delta)
    last_accept = max((at for at, k in events if k == "accepted"), default=None)
    out["last_accepted_at"] = iso(last_accept)
    return out


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def render_summary(model: dict[str, Any], *, width: int = 118) -> str:
    t = model["totals"]
    lines = []
    lines.append(
        f"CONVEYOR {model['run']}  {model['generated_at']}  bases={t['bases']} live={t['live_bases']}  "
        f"catalog={t['catalog']} started={t['started']} queued={t['queued']}  accepted={t['accepted']} rejected={t['rejected']}"
    )
    tm = model.get("timings") or {}
    if tm:
        lines.append(
            f"reads: jobs {tm.get('jobs_s')}s | _live views {tm.get('live_views', 0)} ({tm.get('live_fetched', 0)} fetched, "
            f"{tm.get('live_kb', 0)} KB) | reused {tm.get('manifest_reused', 0)} | manifests {tm.get('manifests')} "
            f"({tm.get('manifest_fetched')} fetched {tm.get('manifest_mb')} MB, {tm.get('manifest_cached')} cached) "
            f"{tm.get('manifest_s')}s | objects {tm.get('objects')} ({tm.get('object_fetched')} new) {tm.get('object_s')}s"
        )
    if not model.get("jobs_known"):
        lines.append("!! Iris job list unavailable: live/parked split and PARKED/STUCK detection are OFF")
    for error in model.get("errors", [])[:5]:
        lines.append(f"ERROR {error}")
    lines.append("")
    passes = t.get("pass") or {}
    lines.append(f"{'FUNNEL (stage now)':22s} {'total':>6s} {'active':>7s} {'waiting':>8s} {'parked':>7s} {'held':>6s} "
                 f"{'rejected':>9s} {'unknown':>8s}")
    for row in model["funnel"]:
        if row["stage"] in ("queued", "accepted"):
            value = row["queued"] if row["stage"] == "queued" else row["accepted"]
            lines.append(f"  {row['stage']:20s} {value:6d}")
            continue
        if not row["total"]:
            continue
        lines.append(
            f"  {row['stage']:20s} {row['total']:6d} {row['active']:7d} {row['waiting']:8d} {row['parked']:7d} "
            f"{row['held']:6d} {row['rejected']:9d} {row['unknown']:8d}"
        )
    lines.append(f"  (live-run items this pass: pending={passes.get('pending', 0)} in-progress={passes.get('touched', 0)} "
                 f"returned={passes.get('returned', 0)}; waiting = returned non-terminal, moves on the next pass; "
                 f"held = frozen under plain --resume)")
    tp = model["throughput"]
    lines.append("")
    lines.append(
        f"THROUGHPUT accepted 1h={tp.get('accepted_1h', 0)} 6h={tp.get('accepted_6h', 0)} | failed-entries "
        f"1h={tp.get('failed_1h', 0)} 6h={tp.get('failed_6h', 0)} | rejected 1h={tp.get('rejected_1h', 0)} "
        f"6h={tp.get('rejected_6h', 0)} | ACCEPTANCES LOST 1h={tp.get('unaccepted_1h', 0)} 6h={tp.get('unaccepted_6h', 0)}  "
        f"(source: {tp.get('source')}; last accept {tp.get('last_accepted_at') or '-'})"
    )
    fbs = model["failures_by_stage"]
    if fbs:
        parts = []
        for stage, c in fbs.items():
            bits = [f"{k.replace('failed_', '')}={v}" for k, v in sorted(c.items())]
            parts.append(f"{stage}[{' '.join(bits)}]")
        text = "FAILURES by stage: " + "  ".join(parts)
        lines.extend(_wrap(text, width))
    hist = []
    for stage, classes in model["failure_histogram"].items():
        for cls, n in classes.items():
            hist.append((n, stage, cls))
    hist.sort(reverse=True)
    if hist:
        lines.append("TOP FAILURE/HOLD CLASSES")
        for n, stage, cls in hist[:8]:
            lines.append(f"  {n:5d}  {stage:20s} {cls[:width - 30]}")
    kinds = model["anomaly_counts"]
    lines.append("")
    lines.append("ANOMALIES " + ("  ".join(f"{k}={kinds.get(k, 0)}" for k in ("PARKED", "STUCK", "STALLED", "UNKNOWN_STATE", "STALE_SNAPSHOT", "NO_SNAPSHOT", "ERROR"))))
    parked = [a for a in model["anomalies"] if a["kind"] == "PARKED"]
    if parked:
        text = "  PARKED runs: " + ", ".join(
            f"{a['run']}({model['bases'][a['run']]['non_terminal']})" for a in parked)
        lines.extend(_wrap(text, width))
    for kind in ("STALLED", "STALE_SNAPSHOT", "NO_SNAPSHOT", "UNKNOWN_STATE", "ERROR"):
        for a in [a for a in model["anomalies"] if a["kind"] == kind][:4]:
            lines.append(f"  {kind} {a['run']}{'/' + a['item'] if a.get('item') else ''}: {a['detail']}"[:width])
    stuck = [a for a in model["anomalies"] if a["kind"] == "STUCK"]
    by_stage = Counter(a["stage"] for a in stuck)
    if stuck:
        lines.append("  STUCK by stage: " + ", ".join(f"{s}={n}" for s, n in by_stage.most_common()))
        for a in stuck[:5]:
            lines.append(f"  STUCK {a['run']}/{a['item'][:48]}: {a['detail']}"[:width])
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for word in text.split(" "):
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = "    " + word
        else:
            line = f"{line} {word}" if line else word
    if line:
        out.append(line)
    return out


def filter_items(items: list[dict[str, Any]], *, run: str | None = None, stage: str | None = None,
                 state: str | None = None, disposition: str | None = None, anomaly: str | None = None,
                 item: str | None = None, pass_state: str | None = None) -> list[dict[str, Any]]:
    def match(value: str | None, pattern: str | None) -> bool:
        if pattern is None:
            return True
        return any(fnmatch.fnmatch(str(value), p.strip()) for p in pattern.split(","))

    out = []
    for r in items:
        if not match(r["run"], run) or not match(r["stage"], stage) or not match(r["state"], state):
            continue
        if not match(r["disposition"], disposition) or not match(r["item"], item) or not match(r.get("pass_state"), pass_state):
            continue
        if anomaly and not any(match(a, anomaly) for a in r["anomalies"]):
            continue
        out.append(r)
    return out


def render_items(items: list[dict[str, Any]], *, now: datetime | None = None, limit: int | None = None) -> str:
    now = now or datetime.now(UTC)
    rows = sorted(items, key=lambda r: (r["run"], STAGE_INDEX.get(r["stage"], 99), r["item"]))
    header = (f"{'RUN':10s} {'ITEM':52s} {'STATE':28s} {'DISP':9s} {'SINCE':>6s} {'ACTIVITY':24s} "
              f"{'WAIT':9s} {'REP':5s} {'FLAGS':8s} ISSUE")
    lines = [header]
    for r in rows[: limit or len(rows)]:
        since = parse_time(r.get("since"))
        age = fmt_age((now - since).total_seconds()) if since else "-"
        act = r.get("activity") or "-"
        if r.get("activity_since"):
            act += f" {fmt_age((now - parse_time(r['activity_since'])).total_seconds())}"
        wait = "-"
        if r.get("wait_attempts") is not None or r.get("next_attempt_at"):
            nxt = parse_time(r.get("next_attempt_at"))
            wait = f"{r.get('wait_attempts') or 0}/" + (fmt_age((nxt - now).total_seconds()) if nxt and nxt > now else ("due" if nxt else "-"))
        rep = f"{r['repairs_used'] if r.get('repairs_used') is not None else '-'}/{r['repairs_max'] if r.get('repairs_max') is not None else '-'}"
        flags = ",".join(a[:5] for a in r["anomalies"]) or "-"
        state = r["state"] if r["state"] != "failed" else f"failed@{r.get('failure_stage')}"
        disp = r["disposition"] + (f":{r['pass_state'][0]}" if r.get("pass_state") and r["disposition"] in ("active", "waiting") else "")
        lines.append(
            f"{r['run']:10s} {r['item'][:52]:52s} {state[:28]:28s} {disp[:9]:9s} {age:>6s} {act[:24]:24s} "
            f"{wait[:9]:9s} {rep:5s} {flags[:8]:8s} {(r.get('issue') or '')[:90]}"
        )
    if limit and len(rows) > limit:
        lines.append(f"... {len(rows) - limit} more (raise --limit)")
    lines.append(f"{len(rows)} item(s)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Watch: edge-triggered lines only
# --------------------------------------------------------------------------------------------


class Watcher:
    """Turns successive models into edge-triggered lines.  First cycle establishes the baseline."""

    def __init__(self, *, stage_step: int = 5, zero_minutes: float = 60.0, status_every_s: float = 900.0) -> None:
        self.stage_step = stage_step
        self.zero_minutes = zero_minutes
        self.status_every_s = status_every_s
        self.cycle = 0
        self.parked: dict[str, set[str]] = {}
        self.stuck: set[tuple[str, str]] = set()
        self.flags: set[tuple[str, str]] = set()
        self.classes: set[tuple[str, str]] = set()
        self.stage_reported: dict[str, int] = {}
        self.accepted_keys: set[str] | None = None
        self.zero_alarm = False
        self.last_status_at = 0.0
        self.baselined = False

    def status_line(self, model: dict[str, Any], now: datetime, read_s: float) -> str:
        t = model["totals"]
        tp = model["throughput"]
        k = model["anomaly_counts"]
        tm = model.get("timings") or {}
        return (f"STATUS {hhmm(now)} cycle={self.cycle} read={read_s:.1f}s bases={t['bases']} live={t['live_bases']} "
                f"started={t['started']} queued={t['queued']} accepted={t['accepted']} (1h +{tp.get('accepted_1h', 0)}, "
                f"6h +{tp.get('accepted_6h', 0)}) active={t['active']} waiting={t['waiting']} parked={t['parked']} held={t['held']} "
                f"rejected={t['rejected']} stuck={k.get('STUCK', 0)} stalled={k.get('STALLED', 0)} stale={k.get('STALE_SNAPSHOT', 0)} "
                f"errors={len(model.get('errors', []))} sources(live={tm.get('live_views', 0)} reused={tm.get('manifest_reused', 0)} "
                f"manifest={tm.get('manifests', 0)} {tm.get('manifest_mb', 0)}MB)")

    def diff(self, model: dict[str, Any], now: datetime, read_s: float, wall: float | None = None) -> list[str]:
        self.cycle += 1
        wall = time.time() if wall is None else wall
        out: list[str] = []
        ts = hhmm(now)
        first = not self.baselined
        self.baselined = True
        # Low-side alarm first: no acceptances for N minutes.
        tp = model["throughput"]
        last = parse_time(tp.get("last_accepted_at"))
        zero = last is None or (now - last) > timedelta(minutes=self.zero_minutes)
        if zero and not self.zero_alarm:
            out.append(f"THROUGHPUT_ZERO {ts} no quality_accepted for >{self.zero_minutes:.0f}m (last {tp.get('last_accepted_at') or 'never seen'}; source {tp.get('source')})")
        elif not zero and self.zero_alarm:
            out.append(f"THROUGHPUT_OK {ts} acceptances resumed (last {tp.get('last_accepted_at')})")
        self.zero_alarm = zero
        for error in model.get("errors", []):
            out.append(f"ERROR {ts} {error}")
        # PARKED runs (edge: new run parked, or new items within a parked run)
        parked_now = {a["run"]: set(a.get("items") or []) for a in model["anomalies"] if a["kind"] == "PARKED"}
        for run, names in sorted(parked_now.items()):
            before = self.parked.get(run)
            detail = next(a["detail"] for a in model["anomalies"] if a["kind"] == "PARKED" and a["run"] == run)
            if before is None:
                out.append(f"PARKED {ts} {run}: {detail}")
            elif names - before:
                extra = sorted(names - before)
                out.append(f"PARKED {ts} {run}: +{len(extra)} item(s) {', '.join(x[:40] for x in extra[:3])}")
        for run in sorted(set(self.parked) - set(parked_now)):
            out.append(f"UNPARKED {ts} {run}")
        self.parked = parked_now
        stuck_now = {(a["run"], a["item"]): a for a in model["anomalies"] if a["kind"] == "STUCK"}
        for key in sorted(set(stuck_now) - self.stuck):
            a = stuck_now[key]
            out.append(f"STUCK {ts} {a['run']}/{a['item']}: {a['detail']}")
        self.stuck = set(stuck_now)
        flag_now = {(a["kind"], f"{a['run']}/{a.get('item') or ''}"): a for a in model["anomalies"]
                    if a["kind"] in ("STALE_SNAPSHOT", "NO_SNAPSHOT", "UNKNOWN_STATE", "STALLED")}
        for key in sorted(set(flag_now) - self.flags):
            a = flag_now[key]
            out.append(f"{a['kind']} {ts} {key[1].rstrip('/')}: {a['detail']}")
        for key in sorted(self.flags - set(flag_now)):
            out.append(f"CLEARED {ts} {key[0]} {key[1].rstrip('/')}")
        self.flags = set(flag_now)
        classes_now = {(stage, cls) for stage, classes in model["failure_histogram"].items() for cls in classes}
        if not first:
            for stage, cls in sorted(classes_now - self.classes):
                n = model["failure_histogram"][stage][cls]
                out.append(f"FAILCLASS {ts} new at {stage}: {cls} (n={n})")
        self.classes |= classes_now
        records = {f"{r['run']}/{r['item']}": r for r in model["items"]}
        accepted_now = {key for key, r in records.items() if r["disposition"] == "accepted"}
        if self.accepted_keys is not None:
            for key in sorted(accepted_now - self.accepted_keys):
                out.append(f"ACCEPTED {ts} {key}")
            for key in sorted(self.accepted_keys - accepted_now):
                r = records.get(key)
                now_state = f"{r['state']} ({r['disposition']}): {r.get('issue') or '-'}" if r else "item missing from snapshot"
                out.append(f"UNACCEPTED {ts} {key}: acceptance lost; now {now_state}"[:400])
        self.accepted_keys = accepted_now
        for row in model["funnel"]:
            value = row["total"] if row["stage"] not in ("queued", "accepted") else row[row["stage"]]
            before = self.stage_reported.get(row["stage"])
            if before is None:
                self.stage_reported[row["stage"]] = value
            elif abs(value - before) >= self.stage_step:
                out.append(f"STAGE {ts} {row['stage']} {before} -> {value}")
                self.stage_reported[row["stage"]] = value
        if first or wall - self.last_status_at >= self.status_every_s:
            out.append(self.status_line(model, now, read_s))
            self.last_status_at = wall
        return out


def watch(args: argparse.Namespace, thresholds: dict[str, float], *, store: "S3Store | None" = None,
          collect_fn: Callable[..., Collected] | None = None, emit: Callable[[str], None] | None = None,
          clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> int:
    """Line-oriented monitor: edge-triggered lines, an ERROR per failed read, GAP on overrun, STATUS heartbeat."""
    emit = emit or (lambda line: print(line, flush=True))
    store = store or S3Store()
    collect_fn = collect_fn or collect
    watcher = Watcher(stage_step=args.stage_step, zero_minutes=args.zero_minutes, status_every_s=args.status_every)
    history = History(persist=not args.no_state)
    last_start = None
    previous: Collected | None = None
    attempt = 0
    every = max(1, int(getattr(args, "manifest_every", 6) or 1))
    while True:
        cycle_start = clock()
        if last_start is not None and cycle_start - last_start > args.interval * 1.5:
            emit(f"GAP {hhmm()} {cycle_start - last_start:.0f}s since previous cycle start (interval {args.interval}s)")
        last_start = cycle_start
        try:
            if getattr(store, "fs", True) is None:
                store.connect()
            if hasattr(store, "keys_fetched_at") and clock() - store.keys_fetched_at > args.key_refresh_min * 60:
                try:
                    store.refresh_keys()
                except Exception as error:  # noqa: BLE001
                    emit(f"ERROR {hhmm()} {error}")
            # Bases with _live/conveyor.json are read every cycle (KBs); manifest-only bases are
            # re-read every ``every``-th cycle and otherwise reuse the previous snapshot.
            refresh = previous is None or attempt % every == 0
            attempt += 1
            collected = collect_fn(store, bases=args.run_list, previous=previous, refresh_manifests=refresh)
            previous = collected
            model = build_model(collected, thresholds=thresholds, history=history)
            read_s = clock() - cycle_start
            for line in watcher.diff(model, datetime.now(UTC), read_s, wall=clock()):
                emit(line)
            emit_duration = model.get("timings") or {}
            if emit_duration and args.verbose:
                emit(f"READ {hhmm()} {json.dumps(emit_duration)}")
        except Exception as error:  # noqa: BLE001 - a monitor degrades into ERROR, never silence
            watcher.cycle += 1
            emit(f"ERROR {hhmm()} cycle {watcher.cycle} failed: {type(error).__name__}: {str(error)[:300]}")
        duration = clock() - cycle_start
        if duration > args.interval:
            emit(f"GAP {hhmm()} cycle took {duration:.0f}s > interval {args.interval}s")
        if args.max_cycles and watcher.cycle >= args.max_cycles:
            return 0
        sleep(max(1.0, args.interval - duration))


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def parse_thresholds(text: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in (text or "").split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = float(value)
    return out


def load_model(args: argparse.Namespace, thresholds: dict[str, float]) -> dict[str, Any]:
    store = S3Store(log=lambda m: print(m, file=sys.stderr, flush=True) if args.verbose else None)
    store.connect()
    jobs = None
    if args.jobs_file:
        jobs = parse_job_list(Path(args.jobs_file).read_text())
    collected = collect(store, jobs=jobs, fetch_job_list=not args.no_jobs, bases=args.run_list,
                        log=(lambda m: print(m, file=sys.stderr, flush=True)) if args.verbose else None)
    history = History(persist=not args.no_state)
    return build_model(collected, thresholds=thresholds, history=history)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print the full model as JSON")
    parser.add_argument("--items", action="store_true", help="print the per-item table (use filters)")
    parser.add_argument("--run", help="filter: run base glob(s), comma-separated (e.g. 'hc*,shard-07?')")
    parser.add_argument("--stage", help="filter: stage glob(s)")
    parser.add_argument("--state", help="filter: state glob(s)")
    parser.add_argument("--disp", help="filter: disposition(s): active,waiting,parked,held,rejected,accepted,unknown")
    parser.add_argument("--anomaly", help="filter: PARKED,STUCK,UNKNOWN_STATE")
    parser.add_argument("--pass-state", dest="pass_state", help="filter: pending,touched,returned (live runs)")
    parser.add_argument("--item", help="filter: item name glob(s)")
    parser.add_argument("--limit", type=int, default=None, help="max rows for --items")
    parser.add_argument("--only-runs", dest="run_list", type=lambda s: [x for x in s.split(",") if x],
                        help="read only these bases (glob ok); default all")
    parser.add_argument("--watch", action="store_true", help="edge-triggered monitor lines")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--max-cycles", type=int, default=0, help="watch: stop after N cycles (0 = forever)")
    parser.add_argument("--stage-step", type=int, default=5, help="watch: report a stage count change of >= N")
    parser.add_argument("--zero-minutes", type=float, default=60.0, help="watch: alarm when no acceptance for N minutes")
    parser.add_argument("--status-every", type=float, default=900.0, help="watch: STATUS heartbeat period (s)")
    parser.add_argument("--key-refresh-min", type=float, default=50.0)
    parser.add_argument("--manifest-every", type=int, default=6,
                        help="watch: re-read manifests of bases without _live/conveyor.json every N cycles (ETag-unchanged never)")
    parser.add_argument("--stuck-hours", help="override thresholds, e.g. 'build=10,repair=3,quality_review=4'")
    parser.add_argument("--jobs-file", help="use a saved 'iris job list' output instead of calling Iris")
    parser.add_argument("--no-jobs", action="store_true", help="skip the Iris job list (disables PARKED/STUCK)")
    parser.add_argument("--no-state", action="store_true", help="do not write ops/state history")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    thresholds = parse_thresholds(args.stuck_hours)
    if args.watch:
        return watch(args, thresholds)
    try:
        model = load_model(args, thresholds)
    except ReadError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(model, sys.stdout, indent=1, default=str)
        print()
    elif args.items:
        rows = filter_items(model["items"], run=args.run, stage=args.stage, state=args.state,
                            disposition=args.disp, anomaly=args.anomaly, item=args.item, pass_state=args.pass_state)
        print(render_items(rows, limit=args.limit))
    else:
        print(render_summary(model))
    return 1 if model.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())

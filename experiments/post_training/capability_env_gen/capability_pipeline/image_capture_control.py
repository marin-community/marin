"""Capture-attempt control for the image controller: evidence, budget, explicit outcomes.

One reviewed role is captured by ``scripts/capture_generic_task_image.py``.  This
module decides what each outcome means for the item, and never leaves it parked:

* a verified capture                         -> ``None`` (the controller moves on);
* a transient failure within the budget      -> ``pending_capture``, ``retryable: True``,
  with ``attempts``, ``max_attempts`` and ``backoff_seconds``;
* the budget exhausted                       -> ``failed_terminal`` (``failure_stage:
  image_capture``, ``reason: image_capture_failed: <stage>:<error>``);
* deterministic task content                 -> ``repairable`` with builder-facing issues;
* our own harness broke                      -> ``pending_capture``, ``retryable: False``,
  ``reason: capture_harness_error``; blocked for the rest of this job (a relaunch
  retries, still within the budget), never a silent loop.

Durable state lives beside the capture, in the attempt directory the sync
uploads, so it survives relaunches:

* ``capture-<role>.attempt-<n>.json``   one record per attempt, written BEFORE the
  command runs (a job killed mid-capture still consumed the attempt), finished after;
* ``capture-<role>.attempt-<n>.log``    bounded, credential-redacted stdout/stderr tail;
* ``capture-<role>.attempt-<n>.receipt.json``  a retired failed receipt of attempt n;
* ``capture-<role>.infrastructure-<k>.json``   a retired receipt written by the
  pre-ledger controller (the old 3-strike retirement), counted as one attempt each.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .generic_image_capture import CONTENT, HARNESS, TRANSIENT, snapshot_identity_matches

DEFAULT_MAX_ATTEMPTS = 6
MAX_ATTEMPTS_ENV = "CAPABILITY_IMAGE_CAPTURE_MAX_ATTEMPTS"
BACKOFF_BASE_SECONDS = 120
BACKOFF_CAP_SECONDS = 3600
CAPACITY_BACKOFF_FLOOR_SECONDS = 300
EVIDENCE_TAIL_BYTES = 32 * 1024  # per stream; one log stays under 64 KiB of output
_REDACT_SCAN_BYTES = 4 * 1024 * 1024
CAPTURED = "captured_pending_privacy_and_publication"
RECEIPT_SCHEMA = "capability-rootfs-capture-v1"
LEDGER_SCHEMA = "capability-capture-attempt-v1"
# Content verdicts that a single slow or overloaded sandbox could fake are
# confirmed by a second attempt before the builder is asked to repair.
_CONFIRM_STEPS = frozenset({"readiness", "quiesce"})
_CODE_FILES = (
    "generic_image_capture.py", "silo_rootfs_capture.py", "silo_rootfs_stream.py",
    "image_capture_control.py", "image_pipeline.py",
)


def max_attempts() -> int:
    """The per-role capture budget; an unusable override falls back to the default."""
    raw = os.environ.get(MAX_ATTEMPTS_ENV)
    try:
        value = int(raw) if raw is not None else DEFAULT_MAX_ATTEMPTS
    except ValueError:
        return DEFAULT_MAX_ATTEMPTS
    return value if 1 <= value <= 50 else DEFAULT_MAX_ATTEMPTS


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_identity() -> str:
    digest = hashlib.sha256()
    here = Path(__file__).parent
    for name in _CODE_FILES:
        path = here / name
        digest.update(name.encode() + b"\0" + (path.read_bytes() if path.is_file() else b"<absent>"))
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Evidence: bounded, redacted command output.
# --------------------------------------------------------------------------- #

_SECRET_ENV_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION", re.IGNORECASE)
_SECRET_PATTERNS = (
    (re.compile(r"(?i)\b(authorization|x-silo-sandbox-token|x-amz-security-token)(\s*[:=]\s*)[^\r\n]+"), r"\1\2<redacted>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer <redacted>"),
    (re.compile(r"(?i)\b(X-Amz-(?:Signature|Credential|Security-Token))=[^&\s\"']+"), r"\1=<redacted>"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "<redacted-access-key-id>"),
    (re.compile(r"(?i)((?:api[_-]?key|access[_-]?key|secret|token|password|passwd)[\"']?\s*[:=]\s*[\"']?)[^\s\"'&,;]{4,}"),
     r"\1<redacted>"),
)


def redact(text: str) -> str:
    """Remove every credential-looking value: env secrets by value, then patterns."""
    secrets = sorted(
        ((name, value) for name, value in os.environ.items() if _SECRET_ENV_NAME.search(name) and len(value) >= 8),
        key=lambda item: -len(item[1]),
    )
    for name, value in secrets:
        text = text.replace(value, f"<redacted:{name}>")
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _tail(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    text = redact(str(value)[-_REDACT_SCAN_BYTES:])
    encoded = text.encode("utf-8")
    if len(encoded) <= EVIDENCE_TAIL_BYTES:
        return text
    return "[... earlier output trimmed ...]\n" + encoded[-EVIDENCE_TAIL_BYTES:].decode("utf-8", "replace")


def _outcome_parts(outcome: object) -> tuple[int | None, str, str]:
    if isinstance(outcome, int):
        return outcome, "", ""
    code = getattr(outcome, "returncode", None)
    return (code if isinstance(code, int) else None,
            _as_text(getattr(outcome, "stdout", None)), _as_text(getattr(outcome, "stderr", None)))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _last_json_line(stdout: str) -> dict:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    return {}


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


# --------------------------------------------------------------------------- #
# The per-role ledger.
# --------------------------------------------------------------------------- #


class _Ledger:
    def __init__(self, attempt_dir: Path, role: str) -> None:
        self.dir = attempt_dir
        self.role = role

    def record_path(self, n: int) -> Path:
        return self.dir / f"capture-{self.role}.attempt-{n}.json"

    def records(self) -> list[dict]:
        rows = []
        pattern = re.compile(rf"capture-{re.escape(self.role)}\.attempt-(\d+)\.json\Z")
        for path in self.dir.glob(f"capture-{self.role}.attempt-*.json"):
            match = pattern.match(path.name)
            if not match or path.is_symlink():
                continue
            try:
                value = json.loads(path.read_text())
            except (OSError, ValueError):
                # An unreadable record still consumed an attempt.
                value = {"state": "unreadable"}
            if not isinstance(value, dict):
                value = {"state": "unreadable"}
            value["n"] = int(match.group(1))
            rows.append(value)
        return sorted(rows, key=lambda row: row["n"])

    def legacy_receipts(self) -> list[Path]:
        return sorted(self.dir.glob(f"capture-{self.role}.infrastructure-*.json"))

    def used(self) -> int:
        return len(self.records()) + len(self.legacy_receipts())

    def start(self, n: int, item_root: Path) -> dict:
        record = {"schema_version": LEDGER_SCHEMA, "role": self.role, "attempt": n, "state": "started",
                  "started_utc": _utc(), "results_root": str(Path(item_root).resolve()),
                  "code_identity": code_identity()}
        _write_atomic(self.record_path(n), json.dumps(record, sort_keys=True, indent=2) + "\n")
        return record

    def finish(self, record: dict, **fields: Any) -> dict:
        # A record read back from disk carries ``n`` (its file number); one from
        # ``start`` carries ``attempt``.  Either names the file to finish.
        number = record.get("n", record.get("attempt"))
        record = {**record, **fields, "state": "finished", "finished_utc": _utc()}
        record.pop("n", None)
        record.setdefault("attempt", number)
        _write_atomic(self.record_path(number), json.dumps(record, sort_keys=True, indent=2) + "\n")
        return record

    def retire(self, capture: Path, n: int | None) -> Path:
        if n is not None:
            target = self.dir / f"capture-{self.role}.attempt-{n}.receipt.json"
        else:
            target = self.dir / f"capture-{self.role}.infrastructure-{len(self.legacy_receipts()) + 1}.json"
        capture.rename(target)
        return target

    def retired_receipts(self) -> list[Path]:
        return self.legacy_receipts() + sorted(self.dir.glob(f"capture-{self.role}.attempt-*.receipt.json"))

    def unverified_sandbox_ids(self) -> list[str]:
        """Capture sandboxes whose deletion no receipt ever verified."""
        pending: list[str] = []
        verified: set[str] = set()
        for path in self.retired_receipts():
            try:
                receipt = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(receipt, dict):
                continue
            for row in receipt.get("prior_sandbox_cleanup") or []:
                if isinstance(row, dict) and row.get("absence_verified") is True:
                    verified.add(str(row.get("sandbox_id")))
            sandbox_id = receipt.get("sandbox_id")
            if isinstance(sandbox_id, str) and (receipt.get("cleanup") or {}).get("absence_verified") is not True:
                pending.append(sandbox_id)
        return [sandbox_id for sandbox_id in pending if sandbox_id not in verified][-5:]


# --------------------------------------------------------------------------- #
# Verdicts.
# --------------------------------------------------------------------------- #


def _read_receipt(path: Path) -> dict | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _env_issue(role: str, receipt: dict) -> str:
    rows = receipt.get("environment_mismatch") or []
    parts = []
    for row in rows[:8]:
        if isinstance(row, dict):
            parts.append(f"{row.get('name')}: reviewed {row.get('reviewed')!r}, sandbox {row.get('observed')!r}"
                         + (f" ({row['hint']})" if row.get("hint") else ""))
    snapshot = (receipt.get("source_snapshot") or {}).get("name")
    return (f"{role} image: the reviewed image_config.Env is not what a fresh sandbox from snapshot "
            f"{snapshot!r} runs with -- " + "; ".join(parts or ["values differ"])
            + ". Correct image_config.Env (or the Dockerfile ENV lines) and provide an updated "
            "image-capture-request.json for fresh review.")


def _check_issue(role: str, receipt: dict) -> str:
    detail = receipt.get("failure_detail") or {}
    extra = ", ".join(f"{key}={detail[key]!r}" for key in ("path", "exit_code", "exit_codes") if key in detail)
    return (f"{role} image failed the capture check '{receipt.get('failed_step')}': "
            f"{receipt.get('error_message') or receipt.get('error_type')}" + (f" ({extra})" if extra else "")
            + ". Fix the snapshot or the capture_lifecycle in image-capture-request.json and provide an "
            "updated request for fresh review.")


def _judge(receipt: dict | None, role: str, image: dict, plan_sha256: str, prior: list[dict]) -> dict:
    """Classify an on-disk receipt: captured / retry / repairable / harness."""
    if receipt is None:
        return {"kind": "harness", "reason": "capture_receipt_unreadable", "error_type": "UnreadableReceipt"}
    if (receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("role") != role
            or receipt.get("plan_sha256") != plan_sha256 or not snapshot_identity_matches(receipt, image)):
        # Receipts live in a per-plan attempt directory; a mismatch is our bug.
        return {"kind": "harness", "reason": "capture_receipt_mismatch", "error_type": "ReceiptPlanMismatch"}
    state = receipt.get("state")
    verified = (receipt.get("cleanup") or {}).get("absence_verified") is True
    failure_class = receipt.get("failure_class")
    underlying = receipt.get("state_before_cleanup") if state == "cleanup_unverified" else state
    base = {"error_type": receipt.get("error_type"), "stage": receipt.get("failed_step"),
            "receipt_state": state, "failure_class": failure_class}
    if state == CAPTURED and verified:
        return {"kind": "captured"}
    if underlying == "privacy_review_required":
        return {**base, "kind": "repairable", "reason": "image_sensitive_paths_present", "failure_class": CONTENT,
                "issues": [(f"{role} image failed the sensitive-path scan; remove credentials and "
                            "unintended private assets from the source snapshot, then provide an "
                            "updated image-capture-request.json for fresh review.")]}
    if underlying == "image_config_mismatch":
        return {**base, "kind": "repairable", "reason": "image_config_env_mismatch", "failure_class": CONTENT,
                "issues": [_env_issue(role, receipt)]}
    if underlying == "capture_content_error" or (failure_class == CONTENT and state == "cleanup_unverified"):
        step = receipt.get("failed_step")
        seen = sum(1 for row in prior
                   if row.get("reason") == "capture_content_check_unconfirmed" and row.get("stage") == step)
        if step in _CONFIRM_STEPS and seen < 1:
            return {**base, "kind": "retry", "reason": "capture_content_check_unconfirmed", "failure_class": TRANSIENT}
        return {**base, "kind": "repairable", "reason": "image_capture_content_check_failed",
                "failure_class": CONTENT, "issues": [_check_issue(role, receipt)]}
    if underlying == "capture_harness_error" or failure_class == HARNESS:
        return {**base, "kind": "harness", "reason": "capture_harness_error", "failure_class": HARNESS}
    if state == "cleanup_unverified":
        return {**base, "kind": "retry", "reason": "capture_cleanup_unverified", "failure_class": TRANSIENT}
    # infrastructure_error, and every receipt the pre-classification capture wrote
    # (its RuntimeError environment "mismatches" were our own comparison bug).
    return {**base, "kind": "retry", "reason": "capture_infrastructure_error", "failure_class": TRANSIENT}


def _line_verdict(line: dict, exit_code: int | None) -> dict:
    """Classify a command that left no receipt, from its structured failure line."""
    failure_class = line.get("failure_class")
    if failure_class not in (TRANSIENT, CONTENT, HARNESS):
        # No parseable line: uv failed to start, the process was killed (124 is our
        # timeout), or it predates structured output.  Retry within the budget.
        failure_class = TRANSIENT
    verdict = {"failure_class": failure_class, "error_type": line.get("error_type"),
               "stage": line.get("stage"), "exit_code": exit_code}
    for key in ("status_code", "error_message", "detail"):
        if line.get(key) is not None:
            verdict[key] = line[key]
    if not line:
        verdict["error_type"] = "timeout" if exit_code == 124 else "unparsed_output"
    return verdict


def _backoff(records: list[dict], latest: dict) -> int:
    streak = 0
    for row in reversed(records):
        if row.get("failure_class") != TRANSIENT:
            break
        streak += 1
    seconds = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2 ** max(0, streak - 1))
    capacity = latest.get("status_code") == 429 or latest.get("error_type") == "SiloRateLimitError" or any(
        isinstance(row, dict) and row.get("status_code") == 429
        for row in ((latest.get("detail") or {}).get("attempts") or []))
    return max(seconds, CAPACITY_BACKOFF_FLOOR_SECONDS) if capacity else seconds


# --------------------------------------------------------------------------- #
# The entry point the controller calls per role.
# --------------------------------------------------------------------------- #


def run_role_capture(
    *, role: str, attempt: Path, plan: dict, plan_path: Path, plan_sha256: str,
    workspace: Path, capture_tools: Path, approval_path: Path, scripts_root: Path,
    builder_session_ids: set[str], command_runner: Callable[[list[str]], object], item_root: Path,
) -> dict | None:
    """Advance one role's capture by at most one attempt; None once it is captured."""
    capture = attempt / f"capture-{role}.json"
    image = next(row for row in plan["images"] if row["role"] == role)
    ledger = _Ledger(attempt, role)
    budget = max_attempts()
    records = ledger.records()
    retired: Path | None = None

    if capture.exists() or capture.is_symlink():
        verdict = _judge(_read_receipt(capture), role, image, plan_sha256, records)
        if verdict["kind"] == "captured":
            return None
        if verdict["kind"] == "repairable":
            return _repairable(role, capture, verdict)
        # A live failed receipt from an earlier call: an unfinished record owns it
        # (the controller died after the command), otherwise the old controller wrote it.
        owner = records[-1] if records and records[-1].get("state") == "started" else None
        if owner is not None:
            ledger.finish(owner, outcome="receipt", **_record_fields(verdict))
        retired = ledger.retire(capture, owner["n"] if owner else None)
        records = ledger.records()

    used = ledger.used()
    latest = records[-1] if records else {}
    if (latest.get("failure_class") == HARNESS
            and latest.get("results_root") == str(Path(item_root).resolve())):
        return _harness_hold(role, latest, used, budget)
    if used >= budget:
        return _terminal(role, latest, used, budget)

    # ``n`` (from the record's file name) is always present; an unreadable
    # record has no ``attempt`` field but still consumed its number.
    n = (records[-1]["n"] + 1) if records else 1
    record = ledger.start(n, item_root)
    command = [str(scripts_root / "capture_generic_task_image.py"),
               "--plan", str(plan_path), "--workspace", str(workspace),
               "--capture-tools", str(capture_tools), "--role", role,
               "--approval", str(approval_path), "--output", str(capture),
               "--execute"]
    for session in sorted(builder_session_ids):
        command += ["--builder-session-id", session]
    for sandbox_id in ledger.unverified_sandbox_ids():
        command += ["--prior-sandbox-id", sandbox_id]
    started = time.monotonic()
    try:
        outcome = command_runner(command)
    except (OSError, RuntimeError) as error:
        # MARIN_PROJECT missing, uv absent: the job is misconfigured.
        record = ledger.finish(record, outcome="runner_error", failure_class=HARNESS,
                               reason="capture_transport_failed", error_type=type(error).__name__,
                               stage="command_runner")
        return _harness_hold(role, record, ledger.used(), budget)
    exit_code, stdout, stderr = _outcome_parts(outcome)
    evidence = None
    if not (exit_code == 0 and capture.is_file()):
        evidence = attempt / f"capture-{role}.attempt-{n}.log"
        header = (f"capture attempt {n} of {budget} for role {role}\nexit_code: {exit_code}\n"
                  f"started_utc: {record['started_utc']}\nfinished_utc: {_utc()}\n"
                  f"wall_seconds: {round(time.monotonic() - started, 1)}\n"
                  f"command: {' '.join(command)}\n")
        _write_atomic(evidence, header + f"--- stdout (tail, redacted) ---\n{_tail(stdout)}\n"
                      f"--- stderr (tail, redacted) ---\n{_tail(stderr)}\n")
    evidence_name = evidence.name if evidence else None

    if capture.exists() or capture.is_symlink():
        verdict = _judge(_read_receipt(capture), role, image, plan_sha256, records)
        record = ledger.finish(record, outcome="receipt", exit_code=exit_code, evidence=evidence_name,
                               **_record_fields(verdict))
        if verdict["kind"] == "captured":
            return None
        if verdict["kind"] == "repairable":
            return _repairable(role, capture, verdict, evidence)
        retired = ledger.retire(capture, n)
        used = ledger.used()
        if verdict["kind"] == "harness":
            return _harness_hold(role, record, used, budget, evidence=evidence, retired=retired)
        return _retry_or_terminal(role, record, ledger.records(), used, budget, evidence, retired)

    verdict = _line_verdict(_last_json_line(stdout), exit_code)
    reason = "capture_command_failed"
    record = ledger.finish(record, outcome="no_receipt", reason=reason, evidence=evidence_name,
                           **{key: value for key, value in verdict.items() if key != "detail"},
                           **({"detail": verdict["detail"]} if "detail" in verdict else {}))
    used = ledger.used()
    if verdict["failure_class"] == CONTENT:
        issue = (f"{role} image could not be captured: {verdict.get('error_message') or verdict.get('error_type')}"
                 + (f" ({json.dumps(verdict['detail'], sort_keys=True)[:400]})" if verdict.get("detail") else "")
                 + ". Provide an updated image-capture-request.json (recipe or snapshot) for fresh review.")
        return {"state": "repairable", "reason": "image_capture_input_unusable", "role": role,
                "failure_class": CONTENT, "stage": verdict.get("stage"), "error_type": verdict.get("error_type"),
                "evidence_path": str(evidence) if evidence else None, "issues": [issue]}
    if verdict["failure_class"] == HARNESS:
        return _harness_hold(role, record, used, budget, evidence=evidence)
    return _retry_or_terminal(role, record, ledger.records(), used, budget, evidence, retired)


def _record_fields(verdict: dict) -> dict:
    return {key: verdict.get(key) for key in ("reason", "failure_class", "error_type", "stage", "receipt_state")
            if verdict.get(key) is not None}


def _summary(record: dict) -> dict:
    return {key: record[key] for key in ("reason", "failure_class", "error_type", "stage", "exit_code",
                                          "status_code", "error_message", "receipt_state", "evidence")
            if record.get(key) is not None}


def _repairable(role: str, capture: Path, verdict: dict, evidence: Path | None = None) -> dict:
    result = {"state": "repairable", "reason": verdict["reason"], "role": role, "capture_path": str(capture),
              "failure_class": CONTENT, "issues": verdict["issues"]}
    for key in ("error_type", "stage"):
        if verdict.get(key):
            result[key] = verdict[key]
    if evidence is not None:
        result["evidence_path"] = str(evidence)
    return result


def _retry_or_terminal(role: str, record: dict, records: list[dict], used: int, budget: int,
                       evidence: Path | None, retired: Path | None) -> dict:
    if used >= budget:
        return _terminal(role, record, used, budget, evidence=evidence)
    result = {**_summary(record), "state": "pending_capture", "role": role, "retryable": True,
              "reason": record.get("reason") or "capture_command_failed",
              "attempts": used, "max_attempts": budget, "backoff_seconds": _backoff(records, record)}
    result.pop("evidence", None)
    if evidence is not None:
        result["evidence_path"] = str(evidence)
    if retired is not None:
        result["retired_receipt"] = str(retired)
    return result


def _terminal(role: str, latest: dict, used: int, budget: int, *, evidence: Path | None = None) -> dict:
    cause = latest.get("reason") or "capture_attempts_exhausted"
    if latest.get("stage") or latest.get("error_type"):
        cause = f"{latest.get('stage') or cause}:{latest.get('error_type') or latest.get('reason')}"
    last = _summary(latest)
    if latest.get("attempt") is not None:
        last["attempt"] = latest["attempt"]
    result = {"state": "failed_terminal", "failure_stage": "image_capture",
              "reason": f"image_capture_failed: {cause}", "role": role, "retryable": False,
              "attempts": used, "max_attempts": budget, "last_failure": last}
    if evidence is not None:
        result["evidence_path"] = str(evidence)
    return result


def _harness_hold(role: str, record: dict, used: int, budget: int, *, evidence: Path | None = None,
                  retired: Path | None = None) -> dict:
    """Our own bug or misconfiguration: loud, blocked for this job, budgeted across jobs."""
    if used >= budget:
        return _terminal(role, record, used, budget, evidence=evidence)
    result = {**_summary(record), "state": "pending_capture", "reason": "capture_harness_error", "role": role,
              "retryable": False, "blocked_until": "job_relaunch", "attempts": used,
              "max_attempts": budget, "failure_class": HARNESS}
    evidence_name = result.pop("evidence", None)
    if record.get("reason") and record["reason"] != "capture_harness_error":
        result["detail_reason"] = record["reason"]
    if evidence is not None:
        result["evidence_path"] = str(evidence)
    elif evidence_name:
        result["evidence_file"] = evidence_name  # beside the capture receipt, in the attempt directory
    if retired is not None:
        result["retired_receipt"] = str(retired)
    return result

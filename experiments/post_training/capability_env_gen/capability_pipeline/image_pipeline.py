"""Resumable controller for GLM-reviewed custom task image construction."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
from collections.abc import Callable
from pathlib import Path

from .generic_image_construction import prepare_construction_capture, request_needed
from .image_capture_control import DEFAULT_MAX_ATTEMPTS, run_role_capture
from .generic_image_migration import migrate_image_pointers
from .generic_image_publication import validate_review
from .image_plan_review import run_review

Runner = Callable[[list[str]], object]


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


COMMAND_TIMEOUT_ENV = "CAPABILITY_IMAGE_COMMAND_TIMEOUT_SECONDS"
# A capture can legitimately take a snapshot rebuild (<=1 h), several capacity
# waits for a sandbox (~10 min each) and a rootfs stream (<=70 min); this bound
# only turns a hung command into a visible failure instead of a stuck item.
DEFAULT_COMMAND_TIMEOUT_SECONDS = 5 * 3600


def _command_timeout() -> float:
    try:
        value = float(os.environ.get(COMMAND_TIMEOUT_ENV, DEFAULT_COMMAND_TIMEOUT_SECONDS))
    except ValueError:
        return DEFAULT_COMMAND_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_COMMAND_TIMEOUT_SECONDS


def default_cli_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run staged trusted scripts under the pinned Marin project.

    stdout and stderr are returned (the capture controller persists a redacted
    tail as evidence).  A command past its deadline is killed with its whole
    process group -- ``uv run`` leaves the script as a grandchild -- and reported
    as exit 124 with the output it had produced.
    """
    project = os.environ.get("MARIN_PROJECT")
    if not project or not Path(project).is_dir():
        raise RuntimeError("MARIN_PROJECT is required for image controller commands")
    argv = ["uv", "run", "--project", project, "--frozen", "--prerelease=allow",
            "--with", "daytona==0.200.2", *command]
    timeout = _command_timeout()
    process = subprocess.Popen(argv, cwd=project, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        stderr = (stderr or "") + f"\n[image controller] command exceeded {timeout:.0f}s and was killed\n"
        return subprocess.CompletedProcess(argv, 124, stdout, stderr)
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _command_result(value: object) -> bool:
    if isinstance(value, int):
        return value == 0
    return getattr(value, "returncode", None) == 0


def _receipt(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("image pipeline receipt is absent or linked")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError("image pipeline receipt must be an object")
    return value


# A capture receipt is written exactly once per path.  Capture attempts are now
# budgeted per role (image_capture_control: default 6, env
# CAPABILITY_IMAGE_CAPTURE_MAX_ATTEMPTS) and every failure ends in an explicit
# outcome -- retryable, repairable, harness hold or terminal.  The old cap of
# three retired infrastructure receipts parked the item forever after the third;
# those retired receipts now count as attempts against the larger budget, so a
# parked item captures again after rollout with no manual file surgery.
MAX_CAPTURE_INFRASTRUCTURE_RETRIES = DEFAULT_MAX_ATTEMPTS  # legacy name for the per-role budget

# An image-plan review without a decision (reviewer interrupted, or its
# decision failed validation), or whose approval no longer binds the plan
# and task, used to hold the item at pending_image_review forever.  It is now
# moved aside as <attempt>.incomplete-N / <attempt>.stale-N (a review packet
# cannot be re-prepared in place) and re-run, at most this many times; then the
# item fails terminally with failure_stage image_review.
MAX_REVIEW_RETIREMENTS = 3
REVIEW_BACKOFF_SECONDS = 300
# A review whose reviewer process never completed (omp exited non-zero or timed
# out: a GLM relay/router outage) is not a review outcome.  It is retired as
# <attempt>.transport-N, never counted against MAX_REVIEW_RETIREMENTS, and its
# retry is budgeted by the conveyor's image_review_transport wait (12 attempts,
# 900 s backoff by default, like builder_process).
REVIEW_TRANSPORT_MARKER = "transport-failure.json"


# A cold-pull receipt is written even when the cold boot failed (state
# "pending" with an error_type: provider transport errors, rate limits, a
# snapshot not yet active).  Such a receipt used to be skipped as "done" on
# the next pass, so migration rejected it forever
# ("exact_migration_evidence_failed") until the wait budget ran out.  A
# non-passing receipt is now retired as cold-pull-<role>.failed-N.json and the
# cold pull re-run, at most this many times per role; then the item fails
# terminally with failure_stage image_cold_pull.
COLD_PULL_PASSED = "passed_pending_task_gates"
MAX_COLD_PULL_RETIREMENTS = 5


def _retired_cold_pulls(path: Path) -> list[Path]:
    pattern = re.compile(re.escape(path.stem) + r"\.failed-\d+\.json\Z")
    return sorted(candidate for candidate in path.parent.glob(path.stem + ".failed-*.json")
                  if pattern.match(candidate.name))


def _cold_pull_state(path: Path) -> tuple[str | None, str | None]:
    """(state, error_type) of an existing cold-pull receipt; (None, None) if unreadable."""
    try:
        value = _receipt(path)
    except (OSError, ValueError, TypeError):
        return None, None
    state, error = value.get("state"), value.get("error_type")
    return (state if isinstance(state, str) else None), (error if isinstance(error, str) else None)


def _migrated_pointers(item_root: Path) -> set[str]:
    """Authored pointers replaced by applied migration receipts (see request_needed)."""
    replaced: set[str] = set()
    for receipt_path in (item_root / "diagnostics/image-capture").glob("attempt-*/migration/migration.json"):
        try:
            receipt = _receipt(receipt_path)
        except (OSError, ValueError, TypeError):
            continue
        changes = receipt.get("changes")
        if receipt.get("state") != "applied" or not isinstance(changes, list):
            continue
        replaced.update(row["image"] for row in changes
                        if isinstance(row, dict) and isinstance(row.get("image"), str))
    return replaced


def controller_io_hold(error: OSError) -> dict:
    """A local filesystem error inside the image controller: a transient wait, never repair input."""
    return {"state": "pending_infrastructure", "reason": "image_controller_io_error", "retryable": True,
            "error_type": type(error).__name__, "issues": [str(error)[:500]]}


def _retired_reviews(review_root: Path) -> list[Path]:
    pattern = re.compile(re.escape(review_root.name) + r"\.(?:incomplete|stale)-\d+\Z")
    return sorted(path for path in review_root.parent.glob(review_root.name + ".*") if pattern.match(path.name))


def _transport_retirements(review_root: Path) -> list[Path]:
    pattern = re.compile(re.escape(review_root.name) + r"\.transport-\d+\Z")
    return sorted(path for path in review_root.parent.glob(review_root.name + ".*") if pattern.match(path.name))


def _step_timeout(agent: object) -> dict:
    """The reviewer session bound, so the conveyor window fits a whole review."""
    value = getattr(agent, "session_time", None)
    return {"step_timeout_seconds": value} if type(value) in (int, float) and value > 0 else {}


def _review_retry(reason: str, review_root: Path, retired: list[Path], **extra: object) -> dict:
    return {"state": "pending_review", "reason": reason, "retryable": True,
            "attempts": len(retired) + 1, "max_attempts": MAX_REVIEW_RETIREMENTS + 1,
            "backoff_seconds": REVIEW_BACKOFF_SECONDS, "review_root": str(review_root), **extra}


def _review_transport_failure(review_root: Path) -> dict | None:
    """Evidence that the reviewer process never completed (``None``: it did, or no evidence)."""
    marker = review_root / REVIEW_TRANSPORT_MARKER
    if marker.is_file() and not marker.is_symlink():
        try:
            value = json.loads(marker.read_text())
        except (OSError, ValueError):
            value = {}
        return {"cause": "transport_exception", **(value if isinstance(value, dict) else {})}
    execution = review_root / "execution.json"
    if execution.is_symlink() or not execution.is_file():
        return None
    try:
        value = json.loads(execution.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    returncode, timed_out = value.get("returncode"), value.get("timed_out")
    if timed_out is True:
        return {"cause": "reviewer_timed_out", "returncode": returncode}
    if type(returncode) is int and returncode != 0:
        return {"cause": "reviewer_exited_nonzero", "returncode": returncode}
    return None


def _review_transport_retry(review_root: Path, evidence: dict, **extra: object) -> dict:
    """Retry after a reviewer transport failure; the conveyor owns the attempt budget and backoff."""
    return {"state": "pending_review", "reason": "image_review_transport_failed", "retryable": True,
            "failure_class": "transport", "transport_attempts": len(_transport_retirements(review_root)) + 1,
            "transport_evidence": evidence, "review_root": str(review_root), **extra}


def _review_matches_task(approval_path: Path, task: Path) -> None:
    from .image_review_contract import validate_retained_packet

    approval = _receipt(approval_path)
    manifest, _ = validate_retained_packet(
        approval_path.parent, plan_sha256=approval["plan_sha256"],
        snapshot_hash=approval["snapshot_hash"],
        manifest_sha256=approval["input_manifest_sha256"],
    )
    for name in (
        "specification.json",
        "binding.json",
        "composite-verifier.json",
        "judge-calibration.json",
    ):
        path = task / name
        expected = manifest["files"].get("workspace/task/" + name)
        if (
            name not in {"composite-verifier.json", "judge-calibration.json"}
            and expected is None
        ) or path.is_symlink():
            raise ValueError("image review task document binding is incomplete")
        if (expected is None) != (not path.exists()):
            raise ValueError("image review task document presence changed")
        if expected is not None and (not path.is_file() or _sha(path) != expected):
            raise ValueError("image review task document changed")


def process_image_construction(
    *, item_root: Path, capture_tools: Path, scripts_root: Path,
    agent: object, builder_session_ids: set[str],
    command_runner: Runner = default_cli_runner,
    review_base: Path | None = None,
    publication_queue: object | None = None,
) -> dict:
    """Advance one item to reviewed OCI pointers without publisher credentials.

    No publication is attempted by this builder-side controller: at pending
    publication it submits the credential-free handoff to the publisher queue
    (``publication_exchange``; ``publication_queue`` overrides the configured
    ``CAPABILITY_PUBLICATION_QUEUE``) and imports the trusted publisher's
    receipts on a later pass. Waiting returns ``retryable: True`` with
    ``backoff_seconds``; see ``docs/image_registry.md`` for the contract.
    """
    item_root = Path(item_root)
    capture_tools = Path(capture_tools)
    scripts_root = Path(scripts_root)
    task = item_root / "workspace/task"
    try:
        migrated = _migrated_pointers(item_root)
        inventory = (request_needed(item_root / "workspace", migrated_pointers=migrated) if migrated
                     else request_needed(item_root / "workspace"))
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return {"state": "repairable", "reason": "invalid_authored_image_documents",
                "error_type": type(error).__name__, "issues": [str(error)]}
    if not inventory["needed"]:
        attempts = sorted((item_root / "diagnostics/image-capture").glob("attempt-*/migration/migration.json"))
        if attempts:
            matching = []
            for receipt_path in attempts:
                try:
                    receipt = _receipt(receipt_path)
                    documents = receipt["documents"]
                    if receipt.get("state") != "applied" or not isinstance(documents, dict):
                        continue
                    expected_names = {"specification.json", "binding.json"}
                    if (task / "composite-verifier.json").exists():
                        expected_names.add("composite-verifier.json")
                    if (task / "judge-calibration.json").exists():
                        expected_names.add("judge-calibration.json")
                    if set(documents) != expected_names:
                        continue
                    if all(_sha(task / name) == documents[name]["after_sha256"] for name in expected_names):
                        matching.append((receipt_path, receipt))
                except (OSError, ValueError, TypeError, KeyError):
                    continue
            if len(matching) != 1:
                return {"state": "pending_migration", "reason": "no_unique_matching_applied_migration",
                        "candidate_receipts": [str(path) for path in attempts]}
            receipt_path, receipt = matching[0]
            return {"state": "ready", "reason": "reviewed_images_migrated",
                    "migration_path": str(receipt_path), "roles": receipt["roles"]}
        return {"state": "ready", "reason": "pinned_public_images", "pointers": inventory["pointers"]}
    try:
        frozen = prepare_construction_capture(item_root, capture_tools)
    except (FileNotFoundError, OSError) as error:
        return {"state": "pending_infrastructure", "reason": "image_capture_inputs_unavailable",
                "error_type": type(error).__name__}
    except (ValueError, TypeError, KeyError) as error:
        return {"state": "repairable", "reason": "invalid_builder_image_request",
                "error_type": type(error).__name__, "issues": [str(error)]}
    attempt = Path(frozen["attempt"])
    plan_path = Path(frozen["plan_path"])
    workspace = Path(frozen["workspace"])
    plan = _receipt(plan_path)
    roles = [row["role"] for row in plan["images"]]
    review_root = (Path(review_base) if review_base is not None
                   else item_root.parent / "image-reviews" / item_root.name) / attempt.name
    approval_path = review_root / "approval.json"
    rerun = not review_root.exists()
    timeout_hint = _step_timeout(agent)
    if review_root.exists():
        retire_as = cause = None
        if not approval_path.is_file():
            result_path = review_root / "result.json"
            try:
                result = _receipt(result_path) if result_path.is_file() else {}
            except (OSError, ValueError, TypeError):
                result = {}
            decision = result.get("state")
            if decision in {"repair", "reject"}:
                return {"state": "repairable", "reason": "image_plan_review_" + decision,
                        "issues": result.get("issues", []), "review_root": str(review_root)}
            if _review_transport_failure(review_root) is not None:
                # The reviewer process never completed (GLM/omp outage): not a
                # review attempt.  Retire it outside the review bound and rerun.
                target = review_root.with_name(
                    f"{review_root.name}.transport-{len(_transport_retirements(review_root)) + 1}")
                review_root.rename(target)
            else:
                # No decision: the reviewer crashed or was interrupted (no
                # result.json), or its decision failed validation (state
                # "pending", e.g. "image plan citation is unbound").
                retire_as = "incomplete"
                cause = {"reason": "image_plan_review_" + str(decision or "incomplete"),
                         "issues": result.get("issues", [])}
            rerun = True
        else:
            try:
                validate_review(approval_path, plan_path, builder_session_ids=builder_session_ids)
                _review_matches_task(approval_path, task)
            except (ValueError, TypeError, KeyError, OSError) as error:
                # The approval no longer binds this plan or task: review afresh.
                retire_as = "stale"
                cause = {"reason": "review_or_task_binding_changed", "error_type": type(error).__name__}
        if retire_as is not None:
            retired = _retired_reviews(review_root)
            if len(retired) >= MAX_REVIEW_RETIREMENTS:
                kind = "incomplete" if retire_as == "incomplete" else "binding_changed"
                return {"state": "failed_terminal", "failure_stage": "image_review",
                        "reason": f"image_plan_review_{kind}_after_{len(retired) + 1}_attempts",
                        "retryable": False, "attempts": len(retired) + 1,
                        "max_attempts": MAX_REVIEW_RETIREMENTS + 1, "last_cause": cause,
                        "review_root": str(review_root)}
            review_root.rename(review_root.with_name(f"{review_root.name}.{retire_as}-{len(retired) + 1}"))
            rerun = True
    if rerun:
        review_root.parent.mkdir(parents=True, exist_ok=True)
        retired = _retired_reviews(review_root)
        try:
            result = run_review(item_root=item_root, plan_path=plan_path,
                                review_root=review_root, agent=agent,
                                builder_session_ids=builder_session_ids)
        except (OSError, RuntimeError) as error:
            evidence = {"cause": "transport_exception", "error_type": type(error).__name__,
                        "error": str(error)[:500]}
            if review_root.is_dir():
                # Lets the next pass retire this directory as transport, not incomplete.
                with contextlib.suppress(OSError):
                    (review_root / REVIEW_TRANSPORT_MARKER).write_text(json.dumps(evidence, sort_keys=True))
            return _review_transport_retry(review_root, evidence, error_type=type(error).__name__,
                                           **timeout_hint)
        if result["state"] != "approve":
            if result["state"] in {"repair", "reject"}:
                return {"state": "repairable", "reason": "image_plan_review_" + result["state"],
                        "issues": result.get("issues", []), "review_root": str(review_root)}
            transport = _review_transport_failure(review_root)
            if transport is not None:
                return _review_transport_retry(review_root, transport, issues=result.get("issues", []),
                                               **timeout_hint)
            return _review_retry("image_plan_review_" + str(result["state"]), review_root, retired,
                                 issues=result.get("issues", []), **timeout_hint)
        try:
            _review_matches_task(approval_path, task)
        except (OSError, ValueError, TypeError, KeyError) as error:
            return _review_retry("review_or_task_binding_changed", review_root, retired,
                                 error_type=type(error).__name__, **timeout_hint)

    plan_sha256 = _sha(plan_path)
    for role in roles:
        # One attempt at most per call; see image_capture_control for the states.
        held = run_role_capture(
            role=role, attempt=attempt, plan=plan, plan_path=plan_path, plan_sha256=plan_sha256,
            workspace=workspace, capture_tools=capture_tools, approval_path=approval_path,
            scripts_root=scripts_root, builder_session_ids=builder_session_ids,
            command_runner=command_runner, item_root=item_root,
        )
        if held is not None:
            if held.get("state") == "pending_capture" and held.get("retryable") is True:
                # One capture command may run for its whole timeout; the conveyor
                # window must fit that plus the backoffs (conveyor._settle_wait).
                held.setdefault("step_timeout_seconds", _command_timeout())
            return held

    publication_paths = {role: attempt / f"publication-{role}.json" for role in roles}
    missing = [role for role, path in publication_paths.items() if not path.is_file()]
    if missing:
        # The controller holds object-store credentials only. It submits the
        # credential-free handoff to the publisher queue, and on a later pass
        # imports the trusted publisher's receipts with the exact handoff
        # checks, then falls through to cold pull in the same call.
        from .publication_exchange import exchange_publication

        pending = {"state": "pending_publication", "roles": missing,
                   "builder_session_ids": sorted(builder_session_ids),
                   "plan_path": str(plan_path), "workspace": str(workspace),
                   "capture_tools": str(capture_tools), "approval_path": str(approval_path),
                   "capture_paths": {role: str(attempt / f"capture-{role}.json") for role in roles},
                   "publication_paths": {role: str(path) for role, path in publication_paths.items()}}
        exchange_root = ((Path(review_base).parent.parent if review_base is not None else item_root.parent)
                         / "image-publication" / item_root.name / attempt.name)
        held = exchange_publication(item_root=item_root, pending=pending,
                                    exchange_root=exchange_root, queue=publication_queue)
        if held is not None:
            return held
        if any(not path.is_file() for path in publication_paths.values()):
            return {**pending, "retryable": False, "reason": "publication_import_incomplete"}
    cold_paths = {role: attempt / f"cold-pull-{role}.json" for role in roles}
    for role in roles:
        if cold_paths[role].exists():
            state, error_type = _cold_pull_state(cold_paths[role])
            if state == COLD_PULL_PASSED:
                continue
            retired = _retired_cold_pulls(cold_paths[role])
            if len(retired) >= MAX_COLD_PULL_RETIREMENTS:
                return {"state": "failed_terminal", "failure_stage": "image_cold_pull",
                        "reason": f"cold_pull_failed_after_{len(retired) + 1}_attempts",
                        "retryable": False, "role": role, "attempts": len(retired) + 1,
                        "max_attempts": MAX_COLD_PULL_RETIREMENTS + 1,
                        "last_error_type": error_type, "cold_pull_path": str(cold_paths[role])}
            cold_paths[role].rename(cold_paths[role].with_name(
                f"{cold_paths[role].stem}.failed-{len(retired) + 1}.json"))
        command = [str(scripts_root / "probe_generic_task_image.py"),
                   "--plan", str(plan_path), "--workspace", str(workspace),
                   "--capture-tools", str(capture_tools), "--approval", str(approval_path),
                   "--publication", str(publication_paths[role]), "--output", str(cold_paths[role])]
        for session in sorted(builder_session_ids):
            command += ["--builder-session-id", session]
        try:
            outcome = command_runner(command)
        except (OSError, RuntimeError) as error:
            return {"state": "pending_cold_pull", "role": role, "reason": "cold_pull_transport_failed",
                    "error_type": type(error).__name__, "step_timeout_seconds": _command_timeout()}
        if not _command_result(outcome):
            return {"state": "pending_cold_pull", "role": role, "reason": "cold_pull_command_failed",
                    "step_timeout_seconds": _command_timeout()}
    migration = attempt / "migration"
    if migration.exists():
        return {"state": "pending_migration", "reason": "existing_migration_requires_validation",
                "migration_path": str(migration)}
    try:
        receipt = migrate_image_pointers(
            task=task, plan_path=plan_path, frozen_workspace=workspace,
            capture_tools=capture_tools, approval_path=approval_path,
            builder_session_ids=builder_session_ids,
            publication_paths=publication_paths, cold_pull_paths=cold_paths,
            output=migration,
        )
    except (ValueError, TypeError, KeyError, OSError) as error:
        return {"state": "pending_migration", "reason": "exact_migration_evidence_failed",
                "error_type": type(error).__name__}
    return {"state": "ready", "reason": "reviewed_images_migrated",
            "migration_path": str(migration / "migration.json"),
            "roles": receipt["roles"]}

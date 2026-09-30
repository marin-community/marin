"""Turn accepted blueprints into TaskCompendium/Harbor task artifacts with OMP.

Agent sessions construct artifacts.  This controller independently imports the
pinned upstream TaskCompendium implementation, lowers the task, executes control
cases where a real grading backend exists, and only exports tasks that pass.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import builder_confinement, sandbox_provider
from .conveyor import (
    REPAIRABLE_STATES,
    ActivityAgent,
    ConveyorConfig,
    ConveyorConfigError,
    ConveyorEntry,
    ConveyorScheduler,
    activity_command_runner,
    classify_result,
    image_failure_stage,
    mark_activity,
    summarize,
    write_status,
)
from .daytona_policy import (
    verifier_bootstrap_sha256,
    verifier_snapshot_recipe,
)
from .daytona_resources import profile_from_receipt, snapshot_name
from .inference import atomic_json, digest
from .validation import AXES, validate_proposal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = PROJECT_ROOT / "vendor" / "task_spec" / "source.lock.json"
REVOCATIONS = PROJECT_ROOT / "data" / "revocations.json"
DRIVER = Path(__file__).with_name("taskcompendium_driver.py")


class SynthesisError(ValueError):
    pass


INCOMPLETE_ADVERSARY_ISSUE = (
    "independent adversary generation is incomplete after bounded output retries"
)
REASON_EVIDENCE = 600
# Gate re-runs per item for runtime infrastructure holds, counted durably by
# infrastructure-history/<item>/revalidation-* so relaunches cannot reset it.
DEFAULT_RUNTIME_INFRA_MAX_RETRIES = 3
UNSUPPORTED_COMPOSITE_FINAL_STATE = (
    "Final-state submission requires executable workspace verification"
)


def _lowering_failure_state(bundle: Path, error: Exception) -> str:
    """Treat a known composite adapter gap as infrastructure, not a task edit."""
    if (
        (bundle / "composite-verifier.json").is_file()
        and UNSUPPORTED_COMPOSITE_FINAL_STATE in str(error)
    ):
        return "pending_schema_validation"
    return "failed"
LEGACY_INCOMPLETE_ADVERSARY_ISSUES = {
    "attestation lacks an independent adversarial attack",
    "independent adversary report needs adjudication or retry",
    "adversary boundary lacks exact step coverage",
}


def _read_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in value
    )
    return cleaned.strip(".-") or "task"


def _build_checklist_path(proposal_hash: str) -> Path:
    exact = PROJECT_ROOT / "docs" / "build_acceptance" / f"{proposal_hash}.md"
    if exact.is_file():
        return exact
    return PROJECT_ROOT / "docs" / "build_acceptance_001.md"


def _normalized_glm_base(value: str) -> str:
    root = value.rstrip("/").removesuffix("/v1")
    if not root:
        raise SynthesisError("GLM judge base URL is empty")
    return root + "/v1"


def _validate_environment_fidelity(item: dict[str, Any], binding: Any) -> None:
    expected = {
        "reasoning": "none",
        "shellsim": "shellsim",
        "container": "docker",
    }.get(item.get("proposal", {}).get("environment"))
    environment = binding.get("environment") if isinstance(binding, dict) else None
    actual = environment.get("kind") if isinstance(environment, dict) else None
    tools = binding.get("tools") if isinstance(binding, dict) else None
    if expected is None or actual != expected:
        raise SynthesisError(
            "final solver environment does not match the admitted proposal: "
            f"expected {expected or 'a known environment'}, got {actual or 'missing'}; "
            "a solver-surface change requires hash-bound readmission"
        )
    expected_backend = {"shellsim": "shellsim", "container": "docker"}.get(
        item["proposal"]["environment"]
    )
    if (
        not isinstance(tools, list)
        or (expected_backend is None and tools)
        or (
            expected_backend is not None
            and (
                not tools
                or any(
                    not isinstance(tool, dict)
                    or tool.get("backend") != expected_backend
                    for tool in tools
                )
            )
        )
    ):
        raise SynthesisError(
            "final public tool bindings do not match the admitted solver environment; "
            "a solver-surface change requires hash-bound readmission"
        )


def _revoked_proposals(path: Path = REVOCATIONS) -> dict[str, dict[str, Any]]:
    try:
        document = _read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        raise SynthesisError(
            f"accepted-input revocation registry is unavailable: {error}"
        )
    if not isinstance(document, dict):
        raise SynthesisError("accepted-input revocation registry has the wrong schema")
    records = document.get("records")
    if document.get("schema_version") != "capability-revocations-v1" or not isinstance(
        records, list
    ):
        raise SynthesisError("accepted-input revocation registry has the wrong schema")
    revoked = {}
    for index, record in enumerate(records):
        evidence = record.get("evidence") if isinstance(record, dict) else None
        proposal_hash = (
            record.get("proposal_hash") if isinstance(record, dict) else None
        )
        if (
            not isinstance(record, dict)
            or not isinstance(proposal_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", proposal_hash)
            or proposal_hash in revoked
            or record.get("state") != "revoked"
            or not isinstance(record.get("capability_id"), str)
            or type(record.get("slot")) is not int
            or not isinstance(record.get("reason_code"), str)
            or not record["reason_code"]
            or not isinstance(record.get("reason"), str)
            or not record["reason"]
            or record.get("allowed_use") != "admission_repair_input_only"
            or not isinstance(evidence, list)
            or not evidence
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not item["path"]
                or Path(item["path"]).is_absolute()
                or ".." in Path(item["path"]).parts
                or not isinstance(item.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                for item in evidence
            )
        ):
            raise SynthesisError(
                f"accepted-input revocation record {index} is malformed"
            )
        revoked[proposal_hash] = record
    return revoked


def load_accepted(
    path: Path,
    limit: int | None = None,
    *,
    allow_pending_admission: bool = False,
    allow_revoked: bool = False,
) -> list[dict[str, Any]]:
    document = _read_json(path)
    if not isinstance(document, list):
        raise SynthesisError("accepted input must be a JSON list")
    accepted = document[:limit] if limit is not None else document
    if not accepted:
        raise SynthesisError("accepted input is empty")
    hashes: set[str] = set()
    keys: set[tuple[str, int]] = set()
    revoked = _revoked_proposals()
    for index, item in enumerate(accepted):
        if not isinstance(item, dict) or not isinstance(item.get("proposal"), dict):
            raise SynthesisError(f"accepted[{index}] lacks a proposal")
        proposal = item["proposal"]
        validate_proposal(proposal, proposal.get("capability_id"), proposal.get("slot"))
        if proposal.get("status") != "proposed":
            raise SynthesisError(f"accepted[{index}] is not proposed")
        if item.get("proposal_hash") != digest(proposal):
            raise SynthesisError(f"accepted[{index}] proposal hash mismatch")
        revocation = revoked.get(item["proposal_hash"])
        if revocation is not None:
            identity_matches = (
                revocation["capability_id"] == proposal["capability_id"]
                and revocation["slot"] == proposal["slot"]
            )
            if not identity_matches:
                raise SynthesisError(
                    f"accepted[{index}] conflicts with its revocation identity"
                )
            if not allow_revoked:
                raise SynthesisError(
                    f"accepted[{index}] proposal is revoked: "
                    f"{revocation['reason_code']}: {revocation['reason']}"
                )
        review = item.get("review")
        if not isinstance(review, dict) or review.get("verdict") != "accept":
            raise SynthesisError(f"accepted[{index}] lacks an accepting review")
        scores = review.get("scores")
        if (
            not isinstance(scores, dict)
            or set(scores) != AXES
            or not all(
                type(score) is int and 4 <= score <= 5 for score in scores.values()
            )
        ):
            raise SynthesisError(
                f"accepted[{index}] review scores do not support acceptance"
            )
        if review.get("critical_failures") or review.get("required_changes"):
            raise SynthesisError(
                f"accepted[{index}] review contains unresolved findings"
            )
        if item.get("construction_context") is not None and not allow_pending_admission:
            admission = item.get("admission")
            history = admission.get("history") if isinstance(admission, dict) else None
            if (
                not isinstance(admission, dict)
                or admission.get("state") != "accepted"
                or admission.get("scope") != "individual_construction"
                or admission.get("portfolio_certified") is not False
                or admission.get("runtime_certified") is not False
                or not isinstance(history, list)
                or not history
                or any(
                    not isinstance(entry, dict)
                    or not isinstance(entry.get("proposal_hash"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", entry["proposal_hash"])
                    or not isinstance(entry.get("review"), dict)
                    for entry in history
                )
                or [entry.get("round") for entry in history]
                != list(range(len(history)))
                or admission.get("source_proposal_hash")
                != history[0].get("proposal_hash")
                or history[-1].get("proposal_hash") != item.get("proposal_hash")
                or history[-1].get("review") != review
            ):
                raise SynthesisError(
                    f"accepted[{index}] lacks a hash-bound individual construction admission"
                )
        provenance = item.get("provenance")
        if not isinstance(provenance, dict):
            raise SynthesisError(f"accepted[{index}] lacks frozen catalog provenance")
        record = provenance.get("capability_record")
        source = provenance.get("catalog_source")
        if not isinstance(record, dict) or provenance.get(
            "capability_record_hash"
        ) != digest(record):
            raise SynthesisError(f"accepted[{index}] capability record hash mismatch")
        if (
            record.get("capability_id") != proposal["capability_id"]
            or record.get("capability", {}).get("id") != proposal["capability_id"]
        ):
            raise SynthesisError(f"accepted[{index}] provenance capability mismatch")
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
        ):
            raise SynthesisError(f"accepted[{index}] catalog source lacks SHA-256")
        progression = provenance.get("learning_progression")
        progression_hash = provenance.get("learning_progression_hash")
        if (progression is not None or progression_hash is not None) and (
            not isinstance(progression, dict)
            or not isinstance(progression_hash, str)
            or progression_hash != digest(progression)
            or progression.get("catalog_version") != source.get("catalog_version")
            or not isinstance(progression.get("edges"), list)
            or any(
                not isinstance(edge, dict)
                or edge.get("dependent_id") != proposal["capability_id"]
                for edge in progression["edges"]
            )
        ):
            raise SynthesisError(
                f"accepted[{index}] learning progression provenance mismatch"
            )
        proposal_hash = item["proposal_hash"]
        if proposal_hash in hashes:
            raise SynthesisError(f"duplicate accepted proposal: {proposal_hash}")
        hashes.add(proposal_hash)
        key = (proposal["capability_id"], proposal["slot"])
        if key in keys:
            raise SynthesisError(
                f"duplicate accepted capability/slot: {key[0]}:{key[1]}"
            )
        keys.add(key)
    return accepted


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _tree_sha256(root: Path) -> str:
    value = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        value.update(path.relative_to(root).as_posix().encode())
        value.update(b"\0")
        value.update(_sha256(path).encode())
        value.update(b"\n")
    return value.hexdigest()


def _workspace_payload_sha256(root: Path) -> str:
    """Hash builder outputs while excluding controller-owned progress plumbing."""
    value = hashlib.sha256()
    ignored = {".capability-progress", "handoffs", "tools"}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in ignored:
            continue
        value.update(relative.as_posix().encode())
        value.update(b"\0")
        value.update(_sha256(path).encode())
        value.update(b"\n")
    return value.hexdigest()


def _omp_transcript_totals(root: Path) -> dict[str, int]:
    """Read credential-free OMP termination and usage counters from JSONL."""
    totals = {
        "assistant_messages": 0,
        "length_stops": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "tool_calls": 0,
        "compactions": 0,
        "session_exits": 0,
    }
    for path in sorted(root.glob("*.jsonl")):
        with path.open(errors="replace") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "compaction":
                    totals["compactions"] += 1
                    continue
                if (
                    event.get("type") == "custom"
                    and event.get("customType") == "session_exit"
                ):
                    totals["session_exits"] += 1
                    continue
                message = event.get("message")
                if event.get("type") != "message" or not isinstance(message, dict):
                    continue
                if message.get("role") != "assistant":
                    continue
                totals["assistant_messages"] += 1
                if message.get("stopReason") == "length":
                    totals["length_stops"] += 1
                usage = message.get("usage")
                if isinstance(usage, dict):
                    for source, target in (
                        ("output", "output_tokens"),
                        ("reasoningTokens", "reasoning_tokens"),
                    ):
                        count = usage.get(source)
                        if type(count) is int and count >= 0:
                            totals[target] += count
                content = message.get("content")
                if isinstance(content, list):
                    totals["tool_calls"] += sum(
                        isinstance(block, dict) and block.get("type") == "toolCall"
                        for block in content
                    )
    return totals


def _counter_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: max(0, after[key] - before.get(key, 0)) for key in after}


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
    umask: int = -1,
) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        umask=umask,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            argv, timeout, output=stdout, stderr=stderr
        ) from None
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


_TOOLCHAIN_HEAL_LOCK = threading.Lock()
_TOOLCHAIN_TREE = ("src", "schema", "shellsim-bridge")
_TOOLCHAIN_FILES = ("pyproject.toml", "uv.lock")


@dataclass(frozen=True)
class OfficialToolchain:
    package_root: Path
    uv: str
    source_package_root: Path | None = None
    # A separate copy for builder and repair agents.  They run uv and edit
    # files freely; the controller's overlay and source must stay pinned
    # (measured 2026-09-29: 31 judge calibrations held on "TaskCompendium source
    # hash mismatch: uv.lock" / "archive file set mismatch" after agents were
    # handed the controller overlay as the "official package root").
    builder_root: Path | None = None

    def builder_copy(self) -> OfficialToolchain:
        """Return this toolchain with a private, disposable copy for agents."""
        target = Path(tempfile.mkdtemp(prefix="taskcompendium-builder-"))
        shutil.copytree(
            self.package_root,
            target,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".venv*", "__pycache__", ".pytest_cache", "target"),
        )
        return dataclasses.replace(self, builder_root=target.resolve())

    @property
    def agent_root(self) -> Path:
        return self.builder_root or self.package_root

    def heal(self) -> dict[str, Any] | None:
        """Restore the controller overlay from the verified source if it drifted.

        Returns None when intact, or a record of the repair.  Raises
        SynthesisError when the pinned source itself no longer verifies.
        """
        if self.source_package_root is None:
            return None
        with _TOOLCHAIN_HEAL_LOCK:
            try:
                self.validate_runtime_overlay()
                return None
            except SynthesisError as error:
                drift = str(error)
            # The pinned source must verify; it is never handed to agents.
            self._verify(self.source_package_root, _read_json(SOURCE_LOCK))
            from .composite_extension import (
                BASE_VERIFIER_SHA256,
                PATCHED_VERIFIER_SHA256,
                apply_taskcompendium_guard,
            )

            for name in _TOOLCHAIN_TREE:
                target = self.package_root / name
                if target.is_symlink() or target.is_file():
                    target.unlink()
                elif target.exists():
                    shutil.rmtree(target)
                source = self.source_package_root / name
                if source.exists():
                    shutil.copytree(
                        source, target,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
                    )
            for name in _TOOLCHAIN_FILES:
                shutil.copy2(self.source_package_root / name, self.package_root / name)
            try:
                patched = apply_taskcompendium_guard(
                    self.package_root, expected_base=BASE_VERIFIER_SHA256
                )
            except ValueError as error:
                raise SynthesisError(f"TaskCompendium overlay heal failed: {error}") from error
            if patched != PATCHED_VERIFIER_SHA256:
                raise SynthesisError("TaskCompendium composite extension overlay hash mismatch")
            self._verify_overlay(self.package_root)
            record = {"healed_at": time.time(), "drift": drift[:REASON_EVIDENCE]}
            print(json.dumps({"event": "toolchain_healed", **record}), flush=True)
            return record

    @staticmethod
    def _expected_overlay_files() -> dict[str, dict[str, str]]:
        """Return the locked base and patched bytes for the runtime overlay."""
        from .composite_extension import (
            BASE_LOWERING_SHA256,
            BASE_RUNNER_SHA256,
            BASE_VERIFIER_SHA256,
            LOWERING_RELATIVE,
            PATCHED_LOWERING_SHA256,
            PATCHED_RUNNER_SHA256,
            PATCHED_VERIFIER_SHA256,
            RUNNER_RELATIVE,
        )

        return {
            "src/taskcompendium/harbor/verifier.py": {
                "base_sha256": BASE_VERIFIER_SHA256,
                "patched_sha256": PATCHED_VERIFIER_SHA256,
            },
            str(RUNNER_RELATIVE): {
                "base_sha256": BASE_RUNNER_SHA256,
                "patched_sha256": PATCHED_RUNNER_SHA256,
            },
            str(LOWERING_RELATIVE): {
                "base_sha256": BASE_LOWERING_SHA256,
                "patched_sha256": PATCHED_LOWERING_SHA256,
            },
        }

    @classmethod
    def _verify_overlay(cls, overlay: Path) -> None:
        """Fail closed unless this is the exact locally locked runtime overlay."""
        source_lock = _read_json(SOURCE_LOCK)
        extension_lock = _read_json(
            PROJECT_ROOT / "vendor" / "task_spec" / "composite_extension.lock.json"
        )
        expected = cls._expected_overlay_files()
        if extension_lock.get("files") != expected:
            raise SynthesisError("TaskCompendium composite extension file lock mismatch")
        patch_path = PROJECT_ROOT / extension_lock["patch"]
        if (
            not patch_path.is_file()
            or _sha256(patch_path) != extension_lock["patch_sha256"]
        ):
            raise SynthesisError("TaskCompendium composite extension patch hash mismatch")
        overlay_files = dict(source_lock["files"])
        for relative, hashes in expected.items():
            if overlay_files.get(relative) != hashes["base_sha256"]:
                raise SynthesisError(
                    "TaskCompendium composite overlay base hash differs from source lock: "
                    + relative
                )
            overlay_files[relative] = hashes["patched_sha256"]
        overlay_lock = {**source_lock, "files": overlay_files}
        cls._verify(overlay, overlay_lock)
        for relative, hashes in expected.items():
            if _sha256(overlay / relative) != hashes["patched_sha256"]:
                raise SynthesisError(
                    "TaskCompendium composite runtime overlay hash mismatch: " + relative
                )

    def validate_runtime_overlay(self) -> None:
        """Recheck both immutable source and applied runtime overlay before reuse."""
        if self.source_package_root is None:
            raise SynthesisError("runtime overlay has no verified source checkout")
        self._verify(self.source_package_root, _read_json(SOURCE_LOCK))
        self._verify_overlay(self.package_root)

    @classmethod
    def resolve(
        cls, output_root: Path, explicit: str | None = None
    ) -> OfficialToolchain:
        del output_root
        lock = _read_json(SOURCE_LOCK)
        candidates: list[Path] = []
        configured = explicit or os.environ.get("TASKCOMPENDIUM_SOURCE")
        if configured:
            candidates.append(Path(configured).expanduser())
        for base in (Path.cwd(), *Path.cwd().parents):
            candidates.extend(
                (base / "lib" / "taskcompendium", base / "vendor" / "taskcompendium")
            )
        diagnostics = []
        seen_candidates: set[Path] = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)
            if not (candidate / "pyproject.toml").is_file():
                diagnostics.append(f"{candidate}: missing pyproject.toml")
                continue
            overlay = None
            try:
                cls._verify(candidate, lock)
                from .composite_extension import (
                    BASE_VERIFIER_SHA256,
                    PATCHED_VERIFIER_SHA256,
                    apply_taskcompendium_guard,
                )

                overlay = Path(
                    tempfile.mkdtemp(prefix="taskcompendium-composite-overlay-")
                )
                shutil.copytree(
                    candidate,
                    overlay,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(
                        ".venv", "__pycache__", ".pytest_cache", "target"
                    ),
                )
                patched = apply_taskcompendium_guard(
                    overlay, expected_base=BASE_VERIFIER_SHA256
                )
                if patched != PATCHED_VERIFIER_SHA256:
                    raise SynthesisError(
                        "TaskCompendium composite extension overlay hash mismatch"
                    )
                cls._verify_overlay(overlay)
                return cls(
                    overlay.resolve(), shutil.which("uv") or "uv",
                    source_package_root=candidate,
                )
            except SynthesisError as error:
                if overlay is not None:
                    shutil.rmtree(overlay, ignore_errors=True)
                diagnostics.append(f"{candidate}: {error}")
                continue
        detail = "; ".join(diagnostics[:8])
        if len(diagnostics) > 8:
            detail += f"; ... {len(diagnostics) - 8} more candidate(s) rejected"
        raise SynthesisError(
            "the exact pinned TaskCompendium checkout was not staged; set --taskcompendium-source "
            f"to {lock['revision']} (network fetch is intentionally disabled on workers). "
            f"Candidate diagnostics: {detail or 'no candidate paths were configured'}"
        )

    @staticmethod
    def _verify(package: Path, lock: dict[str, Any]) -> None:
        for relative, expected in lock["files"].items():
            path = package / relative
            if not path.is_file() or _sha256(path) != expected:
                raise SynthesisError(f"TaskCompendium source hash mismatch: {relative}")
        actual: set[str] = {"pyproject.toml", "uv.lock"}
        for directory in ("src", "schema", "shellsim-bridge"):
            actual.update(
                str(path.relative_to(package))
                for path in (package / directory).rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
        if actual != set(lock["files"]):
            extra = sorted(actual - set(lock["files"]))
            missing = sorted(set(lock["files"]) - actual)
            raise SynthesisError(
                f"TaskCompendium archive file set mismatch; extra={extra}, missing={missing}"
            )

    def _command(self, *arguments: str, runtime: bool = False) -> list[str]:
        frozen = ["--frozen"] if (self.package_root / "uv.lock").is_file() else []
        # Iris and the uploader use the Marin environment. Never let a nested
        # uv sync remove packages from that environment (or vice versa). Keep
        # core and Harbor profiles separate too: concurrent lowering must not
        # remove the extras used by an in-flight runtime trial.
        profile = "runtime" if runtime else "core"
        return [
            "env",
            "-u",
            "VIRTUAL_ENV",
            f"UV_PROJECT_ENVIRONMENT={self.package_root / ('.venv-' + profile)}",
            self.uv,
            "run",
            "--project",
            str(self.package_root),
            *frozen,
            *(
                [
                    "--extra",
                    "harbor",
                    "--prerelease=allow",
                    "--with",
                    "daytona==0.200.2",
                ]
                if runtime
                else []
            ),
            "python",
            str(DRIVER),
            *arguments,
        ]

    def runtime_command(self) -> list[str]:
        command = self._command(runtime=True)
        python_index = command.index("python")
        command[python_index + 1] = str(Path(__file__).with_name("runtime.py"))
        return command

    def validate_and_lower(
        self, bundle: Path, harbor: Path, timeout: int
    ) -> dict[str, Any]:
        completed = _run(
            self._command(
                "validate-and-lower", "--bundle", str(bundle), "--output", str(harbor)
            ),
            timeout=timeout,
        )
        if completed.returncode:
            raise SynthesisError(
                completed.stderr.strip()
                or completed.stdout.strip()
                or "TaskCompendium validation failed"
            )
        return json.loads(completed.stdout)

    def direct_controls(
        self, bundle: Path, controls: Path, timeout: int
    ) -> dict[str, Any]:
        completed = _run(
            self._command(
                "direct-controls", "--bundle", str(bundle), "--controls", str(controls)
            ),
            timeout=timeout,
        )
        if completed.returncode:
            raise SynthesisError(
                completed.stderr.strip()
                or completed.stdout.strip()
                or "control execution failed"
            )
        return json.loads(completed.stdout)


@dataclass(frozen=True)
class OMPAgent:
    executable: str
    model: str | None
    session_time: int
    max_continuations: int
    config: Path | None = None
    max_stagnant_attempts: int = 2

    def invoke(
        self, workspace: Path, session_dir: Path, prompt: Path, attempt: int
    ) -> dict[str, Any]:
        session_dir.mkdir(parents=True, exist_ok=True)
        transcript_before = _omp_transcript_totals(session_dir)
        argv = [
            self.executable,
            "-p",
            "--auto-approve",
            "--no-lsp",
            "--no-prewalk",
            "--thinking",
            "high",
            "--cwd",
            str(workspace),
            "--session-dir",
            str(session_dir),
            "--max-time",
            str(self.session_time),
        ]
        if self.model:
            argv.extend(
                (
                    "--model",
                    self.model,
                    "--smol",
                    self.model,
                    "--slow",
                    self.model,
                    "--plan",
                    self.model,
                )
            )
        if self.config:
            argv.extend(("--config", str(self.config)))
        if attempt:
            argv.append("--continue")
        argv.append("@" + str(prompt))
        # omp's file and shell tools execute in THIS container.  The agent and every
        # process it starts run as an unprivileged per-workspace uid with an
        # allowlisted environment (no object-store keys); see builder_confinement.
        executable = shutil.which(self.executable) or self.executable
        confinement = builder_confinement.open_session(
            workspace,
            session_dir,
            readable=[
                Path(prompt),
                Path(executable),
                *([Path(self.config)] if self.config else []),
            ],
        )
        started = time.time()
        try:
            completed = _run(
                confinement.wrap(argv),
                cwd=workspace,
                timeout=self.session_time + 120,
                env=confinement.env,
                umask=confinement.umask,
            )
            outcome = {
                "returncode": completed.returncode,
                "timed_out": False,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except subprocess.TimeoutExpired as error:
            outcome = {
                "returncode": None,
                "timed_out": True,
                "stdout": error.stdout or "",
                "stderr": error.stderr or "",
            }
        finally:
            confinement.close()
        outcome["confinement"] = confinement.describe()
        outcome["elapsed_seconds"] = time.time() - started
        outcome.update(
            _counter_delta(_omp_transcript_totals(session_dir), transcript_before)
        )
        return outcome


def _agent_package_root(toolchain: Any, workspace: Path | None = None) -> Path | None:
    """The TaskCompendium tree agents may use: never the controller's overlay.

    With per-agent uids each workspace's agents get a private copy they own
    (builder_confinement.private_toolchain); otherwise the shared builder copy.
    """
    if toolchain is None:
        return None
    shared = getattr(toolchain, "builder_root", None) or toolchain.package_root
    if workspace is None:
        return shared
    return builder_confinement.private_toolchain(shared, workspace)


def _prompt(
    item: dict[str, Any],
    session: dict[str, Any],
    is_final: bool,
    source: Path | None,
    contract: Path,
    daytona_available: bool,
) -> str:
    source_text = (
        str(source)
        if source
        else "unavailable; retrieve the exact source revision from source.lock.json"
    )
    bundle = (
        """
This is the last declared build session. Before marking it complete, create task/specification.json,
task/renderings.json, task/binding.json and task/controls.json. They must import and lower with the
official TaskCompendium package. controls.json uses {"schema_version":"1","cases":[...]}; every case
has a unique id, source_author, category, class positive, negative, malformed, or partial, optional step_index,
and either a response or a workspace path relative to task/. Per step include a known-correct positive,
using the exact category string known_correct. The allowed class/category pairs are:
positive/known_correct; negative/plausible_wrong, negative/task_specific_shortcut, or
negative/reward_hack; malformed/empty_or_malformed; partial/criterion_mutation.
Include
an empty_or_malformed candidate, a plausible_wrong candidate, and a task_specific_shortcut or reward_hack
candidate. Malformed answer protocols must expect extraction_error with a null reward; a final-state
verifier may intentionally grade a missing/malformed workspace at no more than 0.2. Negative controls
must be graded at no more than 0.2. Each case may include expect.status, expect.reward_min and expect.reward_max.
These are builder-authored fixed controls, not an independent adversarial audit. Never write that
validation passed: the controller runs it independently. Do not mislabel a legitimately partly correct
criterion mutation as a whole-answer negative. Such a case uses class partial, category
criterion_mutation, a partial_credit_reason tied to the admitted rubric, a graded numeric range below
one, and assertions that bind the intended nested grading-detail fields.
After the initial runtime gate, the controller runs three fresh oracle/solver/control/attack suites
before semantic review, using GLM-5.3 and retaining every attempt. Do not fabricate those receipts.
For a Docker candidate with a declared resource budget, write task/candidate-resources.json with
exactly cpu, memory_gb and disk_gb as positive integers; the controller freezes that request for
the repeated suites. A resource request is not a measurement of enforced limits or peak usage.
For every Docker task, also write task/reset-policy.json with schema_version
"capability-reset-policy-v1", public_root equal to the binding workdir,
readiness {"command":"...","timeout_seconds":positive_integer},
process_policy {"allowed_comm":["..."],"max_count":positive_integer}, and
environment_name_policy {"allowed_names":["..."],"required_names":["..."],
"forbidden_names":["..."]}. These environment names describe the remote inspector
process; they do not prove credential absence. Optionally include mutation
{"command":"...","timeout_seconds":positive_integer} that changes real public
state. Derive the policies from actual isolated task startup and observed process
names. Do not write expected file hashes or reset receipts: the controller measures
one baseline, confirms its deletion, then tests five fresh task starts independently.
Keep readiness and mutation commands bounded. Declare additional directories honestly;
state outside the workdir needs separate evidence. An explicit candidate resource
request is required for this diagnostic. GLM quality review receives measured reset
mismatches for bounded repair; provider gaps remain pending.
"""
        if is_final
        else "Do not emit a final task bundle early unless this session's goal requires it."
    )
    daytona = (
        """
The maintained Daytona builder tools are staged at tools/daytona/. Run untrusted builds, compiles and
task executions through tools/daytona/dt.sh. DAYTONA_API_KEY remains in the process environment. Use a
pinned snapshot, fresh network-blocked sandboxes, and the existing validate_env adapter for repeated
base/oracle runs where that contract applies. Preserve dt_calls.jsonl and generated validation records.
This evidence is required for executable tasks but does not replace the independent Harbor gate.
Shared provider snapshots are outside this task's ownership. Never delete or rename existing
snapshots, including harbor__ caches, to make quota room; age and naming patterns do not establish
ownership or lack of active use. On quota exhaustion, retain the error and report needs_continuation.
Delete only sandboxes created by this task and snapshots whose exact creation receipt belongs to
this task and whose use has ended. Never prewarm a controller cache name from a different image or
recipe: a matching name does not prove image identity. Report the required infrastructure change.
"""
        if daytona_available
        else "Daytona is unavailable; record this and do not claim generated code was executed."
    )
    if daytona_available and sandbox_provider.provider() == sandbox_provider.SILO:
        # Same tools and same dt.py CLI, backed by silo.  Only the claims that
        # stop being true change: silo has no snapshot quota, so telling a
        # builder to stop on quota exhaustion would reimpose the ceiling the
        # provider exists to remove.  Snapshot ownership rules are unchanged.
        for old, new in (
            ("DAYTONA_API_KEY remains in the process environment.",
             ("The sandbox provider is silo (SILO_API_TOKEN and SILO_BROKER_RESOLVE_URL remain in the "
              "process environment); dt.sh keeps the Daytona CLI, and every sandbox has no network.")),
            ("snapshots, including harbor__ caches, to make quota room; age and naming patterns",
             "snapshots, including harbor__ caches; age and naming patterns"),
            ("On quota exhaustion, retain the error and report needs_continuation.",
             ("There is no snapshot quota. A create that reports capacity (HTTP 429) is waiting for "
              "room: retry it rather than reporting needs_continuation.")),
        ):
            if old not in daytona:
                raise RuntimeError(f"builder prompt drifted; cannot adapt it for silo: {old!r}")
            daytona = daytona.replace(old, new)
    checklist = (
        f"""The blocking construction checklist is {contract / "build_acceptance.md"}; materialize its
private build-acceptance-v1 record at task/build-acceptance.json with every checklist ID for this
capability exactly once, passed only with hash-bound measured artifacts. Artifact paths resolve
relative to task/, not the workspace: copy cited logs and sources into a private task/evidence/
directory and cite evidence/... with hashes of the packaged bytes. Do not expose private evidence
as solver resources or use parent-directory paths."""
        if item.get("construction_context") is not None
        else "No candidate-specific construction checklist applies to this accepted manifest."
    )
    return f"""Build one accepted capability task. Work only inside the current proposal workspace.
The immutable accepted blueprint is {contract / "accepted.json"}. The pinned source manifest is
{contract / "source.lock.json"}, its fail-closed composite overlay is
{contract / "composite_extension.lock.json"}, and the integration notes are
{contract / "task_contract.md"}.
The required measured evidence matrix is {contract / "quality-conditions.json"}. Build and retain the
clean-build, reset, oracle, blind-solver, repeated-grading, resource, extraction, failure-taxonomy,
counterexample, provenance, family, and code-mutation evidence it names; one successful runtime trial
does not satisfy repeated-evidence conditions.
For concrete staged-tool commands to collect clean-build and resource samples,
read {contract / "builder_measurements.md"}. Adapt its commands to the actual task,
retain raw measurements, and distinguish unavailable metrics from measured zeros.
{checklist}
Official package root: {source_text}.

CURRENT SESSION
{json.dumps(session, indent=2, sort_keys=True)}

For substantial artifacts, create .capability-progress/{_safe_name(session["session"])}.json before
extended analysis. Record the session, state=in_progress, small numbered work_units with pending or
complete status, concrete artifact paths, completed checks, and next_unit. Use write, edit, or bash tools
immediately to create a real skeleton, then implement and validate one bounded work unit at a time,
updating the checkpoint after each unit. Do not draft a large generator or many-file fixture entirely in
private reasoning before the first durable write. The checkpoint is recovery metadata, never completion
evidence and never a substitute for the required handoff or checks.

Honor depends_on and inspect prior files in handoffs/. Preserve the proposal's intended difficulty;
split substantial work across its declared sessions rather than replacing it with a toy. Research and
test claims using available tools. Limit GitHub API calls to three per session; prefer raw-content URLs and
git fetch/show for repository evidence. Use immutable sources and container image digests.
Retain the actual origin of every image digest: registry manifest metadata or Docker image inspection,
plus the exact build recipe/context and provider snapshot mapping when applicable. A 64-character suffix
in a Daytona provider reference is not evidence of an OCI digest; never relabel it as sha256: merely to
pass schema validation. An example task's locally built image ID is not evidence that this worker or
provider can pull or reconstruct it. Verify image availability and cold reconstruction using the actual
target backend. Distinguish a measured provider snapshot from a portable OCI image, and report an
unsupported image representation as an unresolved build condition rather than fabricating identity.
For an actual custom snapshot, read {contract / "builder_images.md"} and write the
hash-bound task/image-capture-request.json for the trusted GLM review/capture path.
Existing pinned public base images need no recapture. Never author an approval receipt.
Public instructions
must not mention hidden grading, evaluators, reference answers, or rewards. Use the frozen
provenance.capability_record in accepted.json exactly; do not replace or summarize its catalog identity.
Independently verify provisional reference values and reviewer-proposed corrections against the
actual public task definition before authoring the oracle. Review prose is fallible, not an answer
key. Use executable checks in the sandbox or independently grounded evidence; retain disagreements
and derivations. Correcting a demonstrably wrong provisional answer must preserve the task's inputs
and success criterion, never change those inputs merely to make a reviewer assertion true. Report
any genuinely necessary task redesign for readmission.
Read the complete accepted review, including review.issues as well as required_changes. Resolve every
concrete quantitative, cutoff, rubric, data-shape, and protocol issue in the built task. For each such
issue, retain the exact artifact path, check command, and hash-bearing evidence that demonstrates the
resolution; list those artifacts in the final handoff. An accepting review is not permission to ignore
its informational findings, and a self-report without the referenced artifact is not evidence.
A judge proposal must use TaskTrove mode judge with an explicit JudgeConfig whose GLM provider, model, and
base URL exactly match {contract / "judge-policy.json"}. Keep credentials out of TaskSpec, include task-private calibration
fixtures/resources, and exercise the native Harbor judge transport in controls. Also create private
task/judge-calibration.json with schema taskcompendium-judge-calibration-v1 and the exact specification
SHA-256. It needs at least 40 positive and 40 negative semantic variant groups that are declared and
observed on expected_judge_path=model, plus oracle, plausible_wrong, empty, and prompt_injection strata.
Every case declares its repeatable source_family, variant_group, design_label, and expected_judge_path
(model, exact, or constraints); paraphrases and related constructions share a variant_group. Every
plausible_wrong case must use model. Exact and constraint gates cannot pad the model group counts, and
fixed-task calibration must not claim generalization to unseen questions.
Pinned Harbor rejects multi-step all_required_steps and supports only mean or final aggregation. For an
accepted design that needs executable prechecks, critical gates, custom weights, penalties, or
disqualifiers around a native judge, use the maintained composite adapter rather than judge prose or
multi-step averaging. Write task/composite-verifier.json with schema
taskcompendium-composite-verifier-v1, bind the raw specification SHA-256, adapter SHA-256
{_sha256(Path(__file__).with_name("composite_verifier.py"))}, and policy SHA-256
{_sha256(Path(__file__).with_name("composite_policy.py"))}, native judge protocol SHA-256
{_sha256(Path(__file__).with_name("native_judge_protocol.py"))}, and declare every step. Machine checks are
private verifier-resource scripts with immutable image digests and roles gate, criterion, or penalty.
Retain a mutation campaign that kills every critical machine-check mutant and at least 90% of all
targeted machine-check mutants, with independently justified labels and surviving mutants disclosed.
Gate weights are omitted; criterion and penalty weights are positive. Judge criterion_weights align
exactly with the native checklist criteria; critical_indices and critical_min retain must-pass rubric
conditions. When an admitted rubric requires two independent judgments and a conditional third
adjudicator, declare judge.consensus exactly as mode=two_then_third, initial_samples=2,
disagreement_tolerance=0.0, resolution=median, and set the native JudgeModelPolicy samples to 2. The
adapter runs one additional full one-sample pass only when an initial binary criterion differs, retains
all raw judgments, native attempt counts, complete-vector counts, and actual model-completion counts, and
never converts transport failure into a tie-break. Represent
ordinal 0/1/2 or 0/1/2/3 anchors as low-to-high monotone binary threshold criteria and declare their
non-overlapping equal-weight mappings in judge.anchor_groups. A raw or resolved group such as [0,1] is
an invalid task, not partial credit. Each threshold bN means membership in the original ordinal anchors
N and above; retain the full original anchor boundary in that criterion, accept all implementations the
anchor permits, and do not strengthen a partial-credit anchor into the next anchor. Where the admitted
source omits an intermediate anchor, label any proposed interpolation explicitly and obtain a fresh
proposal review instead of claiming that the interpolation preserves an original anchor. A failed
deterministic machine gate skips the judge, returns reward zero, and retains the failed machine evidence without
fabricating raw judge or anchor scores; only an observed criterion disagreement triggers the third pass.
conditional_caps may bind a machine trigger threshold to selected judge criterion indices
and cap that section to a declared maximum fraction; use this for section-specific cutoffs such as an
overflag count that zeros one register section. A penalty script reports normalized violation magnitude
(zero means no penalty) and its weight preserves the original capped point deduction. The adapter forces
zero for any failed machine gate or critical judge criterion, otherwise computes the declared weighted
machine-plus-judge score minus penalties, and preserves infrastructure errors as null outcomes. The
adapter snapshots every declared FileSubmission/FinalState and JudgeView file, preserves transcript
evidence, and injects machine results plus original private reference context into the native judge.
Declare all filesystem evidence explicitly; never assume undeclared workspace files are copied. Retain
artifacts proving the original critical gates, weights, and penalties map exactly into this contract.
{daytona}
{bundle}

When this session's concrete work and checks are complete, write handoffs/{session["session"]}.json with:
{{"session":{json.dumps(session["session"])},"status":"complete","artifacts":["relative/path"],
"checks":[{{"name":"...","command":"...","exit_code":0}}],"notes":"..."}}.
All artifact paths must exist. If work remains, omit the completion file so the controller continues the
same OMP session. Do not fabricate check results or backend availability.
"""


def _handoff_validation(
    workspace: Path, session_name: str
) -> tuple[dict[str, Any] | None, str]:
    path = workspace / "handoffs" / f"{_safe_name(session_name)}.json"
    if not path.is_file():
        return None, "completion handoff is absent"
    try:
        document = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None, "completion handoff is not readable JSON"
    if not isinstance(document, dict):
        return None, "completion handoff must be a JSON object"
    if document.get("session") != session_name:
        return None, f"handoff session must equal {session_name!r}"
    if document.get("status") != "complete":
        return None, "handoff status must equal 'complete'"
    artifacts = document.get("artifacts")
    checks = document.get("checks")
    if not isinstance(artifacts, list) or not artifacts:
        return None, (
            "handoff artifacts must be a non-empty JSON list of existing relative "
            "path strings, not a keyed object or grouped catalog"
        )
    if not isinstance(checks, list) or not checks:
        return None, "handoff checks must be a non-empty JSON list"
    if any(
        not isinstance(check, dict)
        or not isinstance(check.get("name"), str)
        or not check["name"].strip()
        or not isinstance(check.get("command"), str)
        or not check["command"].strip()
        or check.get("exit_code") != 0
        for check in checks
    ):
        return None, (
            "each handoff check needs a non-empty name and command plus "
            "numeric exit_code 0"
        )
    for relative in artifacts:
        if not isinstance(relative, str):
            return None, "every handoff artifact must be a relative path string"
        target = (workspace / relative).resolve()
        if workspace.resolve() not in target.parents:
            return None, f"handoff artifact {relative!r} escapes the workspace"
        if not target.exists():
            return None, f"handoff artifact {relative!r} does not exist"
    return document, "valid"


def _handoff(workspace: Path, session_name: str) -> dict[str, Any] | None:
    return _handoff_validation(workspace, session_name)[0]


def _progress_checkpoint(workspace: Path, session_name: str) -> dict[str, Any] | None:
    path = workspace / ".capability-progress" / f"{_safe_name(session_name)}.json"
    if not path.is_file() or path.stat().st_size > 64 * 1024:
        return None
    try:
        document = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict) or document.get("session") != session_name:
        return None
    return document


def _progress_checkpoint_summary(
    workspace: Path, session_name: str
) -> dict[str, Any] | None:
    document = _progress_checkpoint(workspace, session_name)
    if document is None:
        return None
    path = workspace / ".capability-progress" / f"{_safe_name(session_name)}.json"
    units = document.get("work_units")
    complete = (
        sum(
            isinstance(unit, dict) and unit.get("status") == "complete"
            for unit in units
        )
        if isinstance(units, list)
        else 0
    )
    return {
        "path": str(path.relative_to(workspace)),
        "sha256": _sha256(path),
        "state": document.get("state"),
        "work_unit_count": len(units) if isinstance(units, list) else 0,
        "complete_work_unit_count": complete,
    }


def _continuation_prompt(
    workspace: Path,
    session: dict[str, Any],
    attempt: int,
    prior: dict[str, Any],
) -> str:
    name = session["session"]
    checkpoint_path = workspace / ".capability-progress" / f"{_safe_name(name)}.json"
    checkpoint = _progress_checkpoint(workspace, name)
    handoff_path = workspace / "handoffs" / f"{_safe_name(name)}.json"
    _, handoff_issue = _handoff_validation(workspace, name)
    if handoff_path.is_file():
        first_action = (
            f"An existing completion handoff at {handoff_path} is invalid: "
            f"{handoff_issue}. Your first action must edit that handoff to fix "
            "this exact defect. Preserve all real artifacts and checks; do not "
            "rerun completed work merely to restate it."
        )
    else:
        first_action = (
            "Your first action in this continuation must be a write, edit, "
            f"or bash tool call that creates or updates {checkpoint_path}. "
            "Immediately create or update a real requested artifact skeleton as well."
        )
    observed = {
        key: prior.get(key, 0)
        for key in (
            "length_stops",
            "output_tokens",
            "reasoning_tokens",
            "assistant_messages",
            "tool_calls",
            "compactions",
            "workspace_changed",
        )
    }
    return f"""Continue declared builder session {name!r}; this is bounded recovery attempt {attempt}.
The prior OMP process returned without a valid handoff. Its controller-observed metadata was:
{json.dumps(observed, indent=2, sort_keys=True)}

Do not restart the plan, re-read the complete contract, or draft the whole artifact in private reasoning.
Use the existing conversation and workspace. {first_action}
The checkpoint schema is:
{{"session":{json.dumps(name)},"state":"in_progress","work_units":[{{"id":"...","status":"pending|complete","artifacts":["..."],"checks":["..."]}}],"next_unit":"..."}}.
Current checkpoint, if valid:
{json.dumps(checkpoint, indent=2, sort_keys=True) if checkpoint is not None else "absent"}

Then implement one small work unit
at a time through tools, run its concrete check, and persist both artifact and checkpoint before planning
the next unit. For a large generator, write executable source incrementally by coherent file groups or
functions; do not mentally compose the complete generator before writing. Reuse the already-read ground
truth and record any genuinely unresolved citation in the checkpoint. A progress checkpoint is not a
completion receipt. Write handoffs/{_safe_name(name)}.json only after every declared artifact and real
acceptance check is complete.
"""


def _run_sessions(
    item: dict[str, Any],
    item_root: Path,
    agent: OMPAgent,
    source: Path | None,
    contract: Path,
    daytona_available: bool,
) -> list[dict[str, Any]]:
    proposal = item["proposal"]
    workspace = item_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "handoffs").mkdir(exist_ok=True)
    sessions: list[dict[str, Any]] = []
    completed_names: set[str] = set()
    plan = proposal["builder_plan"]
    safe_names = [_safe_name(session["session"]) for session in plan]
    if len(set(safe_names)) != len(safe_names):
        raise SynthesisError(
            "builder session names collide after filesystem normalization"
        )
    for index, session in enumerate(plan):
        name = session["session"]
        if not set(session["depends_on"]) <= completed_names:
            sessions.append({"session": name, "status": "blocked_dependency"})
            continue
        existing = _handoff(workspace, name)
        if existing:
            completed_names.add(name)
            sessions.append(
                {
                    "session": name,
                    "status": "complete",
                    "resumed_from_disk": True,
                    "handoff": existing,
                }
            )
            continue
        session_root = item_root / "sessions" / _safe_name(name)
        session_root.mkdir(parents=True, exist_ok=True)
        prompt = session_root / "prompt.md"
        prompt.write_text(
            _prompt(
                item,
                session,
                index == len(plan) - 1,
                source,
                contract,
                daytona_available,
            )
        )
        prior_status_path = session_root / "status.json"
        try:
            prior_status = _read_json(prior_status_path)
        except (OSError, json.JSONDecodeError):
            prior_status = {}
        historical_attempts = prior_status.get("attempts", [])
        if not isinstance(historical_attempts, list) or any(
            not isinstance(value, dict) for value in historical_attempts
        ):
            historical_attempts = []
        attempts = list(historical_attempts)
        attempt_numbers = []
        for path in session_root.glob("attempt-*.log"):
            match = re.fullmatch(r"attempt-(\d+)\.log", path.name)
            if match:
                attempt_numbers.append(int(match.group(1)))
        attempt_offset = max(attempt_numbers, default=-1) + 1
        transcript_root = session_root / "transcript"
        has_transcript = any(transcript_root.glob("*.jsonl"))
        historical_transcript = _omp_transcript_totals(transcript_root)
        handoff = None
        stagnant_attempts = 0
        unresolved_length_stops = 0
        continuation_reason = None
        for local_attempt in range(agent.max_continuations + 1):
            attempt = attempt_offset + local_attempt
            is_continuation = has_transcript or attempt > 0
            active_prompt = prompt
            if is_continuation:
                active_prompt = session_root / f"continuation-{attempt}.md"
                prior_observation = attempts[-1] if attempts else {}
                if has_transcript and not any(
                    key in prior_observation
                    for key in (
                        "length_stops",
                        "output_tokens",
                        "reasoning_tokens",
                    )
                ):
                    prior_observation = {
                        **prior_observation,
                        **historical_transcript,
                    }
                active_prompt.write_text(
                    _continuation_prompt(
                        workspace,
                        session,
                        attempt,
                        prior_observation,
                    )
                )
            payload_before = _workspace_payload_sha256(workspace)
            mark_activity(item_root, "builder_session", session=name, attempt=attempt)
            outcome = agent.invoke(
                workspace,
                transcript_root,
                active_prompt,
                max(attempt, 1) if is_continuation else 0,
            )
            outcome["workspace_changed"] = (
                _workspace_payload_sha256(workspace) != payload_before
            )
            log = session_root / f"attempt-{attempt}.log"
            log.write_text(
                str(outcome["stdout"]) + "\n--- STDERR ---\n" + str(outcome["stderr"])
            )
            attempts.append(
                {
                    key: value
                    for key, value in outcome.items()
                    if key not in {"stdout", "stderr"}
                }
            )
            handoff = _handoff(workspace, name)
            if handoff:
                break
            if outcome["returncode"] not in (0, None) and not outcome["timed_out"]:
                continuation_reason = "agent_process_failed"
                break
            if outcome["workspace_changed"]:
                stagnant_attempts = 0
                unresolved_length_stops = 0
            else:
                stagnant_attempts += 1
                unresolved_length_stops += outcome.get("length_stops", 0)
            if stagnant_attempts >= getattr(agent, "max_stagnant_attempts", 2):
                continuation_reason = (
                    "model_output_budget_exhausted_without_workspace_progress"
                    if unresolved_length_stops
                    else "no_workspace_progress"
                )
                break
        status = "complete" if handoff else "continuation_required"
        if not handoff and continuation_reason is None:
            continuation_reason = "continuation_limit_exhausted"
        session_status = {
            "session": name,
            "status": status,
            "attempts": attempts,
            "handoff": handoff,
        }
        if not handoff:
            session_status["continuation_reason"] = continuation_reason
            checkpoint = _progress_checkpoint_summary(workspace, name)
            if checkpoint is not None:
                session_status["progress_checkpoint"] = checkpoint
        sessions.append(session_status)
        atomic_json(session_root / "status.json", sessions[-1])
        if handoff:
            completed_names.add(name)
    return sessions


def _validate_controls(document: dict[str, Any], step_count: int) -> None:
    if document.get("schema_version") != "1" or not isinstance(
        document.get("cases"), list
    ):
        raise SynthesisError("task/controls.json must use schema_version 1")
    cases = document["cases"]
    identifiers = [case.get("id") for case in cases if isinstance(case, dict)]
    if (
        len(identifiers) != len(cases)
        or len(set(identifiers)) != len(cases)
        or not all(identifiers)
    ):
        raise SynthesisError("control cases need unique nonempty IDs")
    if any(
        not isinstance(identifier, str)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", identifier)
        for identifier in identifiers
    ):
        raise SynthesisError("control IDs must be safe portable names")
    classes = Counter(case.get("class") for case in cases)
    if not classes["positive"] or not classes["negative"] or not classes["malformed"]:
        raise SynthesisError("controls require positive, negative, and malformed cases")
    for case in cases:
        if case.get("class") not in {"positive", "negative", "malformed", "partial"}:
            raise SynthesisError(f"invalid control class: {case.get('class')}")
        if (
            "response" not in case
            and "workspace" not in case
            and "transcript" not in case
        ):
            raise SynthesisError(f"control {case['id']} has no candidate submission")
        if (
            not isinstance(case.get("source_author"), str)
            or not case["source_author"].strip()
        ):
            raise SynthesisError(f"control {case['id']} lacks source_author")
        category = case.get("category")
        allowed_categories = {
            "positive": {"known_correct"},
            "negative": {
                "plausible_wrong",
                "task_specific_shortcut",
                "reward_hack",
            },
            "malformed": {"empty_or_malformed"},
            "partial": {"criterion_mutation"},
        }
        if category not in allowed_categories[case["class"]]:
            raise SynthesisError(
                f"control {case['id']} has invalid category for {case['class']}"
            )
        expected = case.get("expect", {})
        if not isinstance(expected, dict):
            raise SynthesisError(f"control {case['id']} expect must be an object")
        for bound in ("reward_min", "reward_max"):
            if bound in expected and (
                type(expected[bound]) not in (int, float)
                or not math.isfinite(expected[bound])
                or not 0 <= expected[bound] <= 1
            ):
                raise SynthesisError(f"control {case['id']} has invalid {bound}")
        if expected.get("reward_min", 0) > expected.get("reward_max", 1):
            raise SynthesisError(f"control {case['id']} has reversed reward bounds")
        assertions = expected.get("assertions", [])
        if not isinstance(assertions, list):
            raise SynthesisError(f"control {case['id']} assertions must be a list")
        for assertion in assertions:
            path = assertion.get("path") if isinstance(assertion, dict) else None
            if (
                not isinstance(path, list)
                or len(path) < 2
                or path[0] != "detail"
                or any(
                    not ((isinstance(p, str) and p) or (type(p) is int and p >= 0))
                    for p in path
                )
                or "equals" not in assertion
                or type(assertion["equals"]) not in (str, bool, int, float, type(None))
                or (
                    type(assertion["equals"]) in (int, float)
                    and not math.isfinite(assertion["equals"])
                )
            ):
                raise SynthesisError(
                    f"control {case['id']} has invalid criterion assertion"
                )
        if case["class"] == "partial" and (
            expected.get("status") != "graded"
            or "reward_min" not in expected
            or "reward_max" not in expected
            or expected["reward_max"] >= 1
            or not assertions
            or not isinstance(case.get("partial_credit_reason"), str)
            or not case["partial_credit_reason"].strip()
        ):
            raise SynthesisError(
                f"partial control {case['id']} needs bounded reward, criterion assertions and an admitted-rubric rationale"
            )
        if case["class"] == "malformed":
            if expected.get("status") == "extraction_error":
                if (
                    expected.get("reward_min") is not None
                    or expected.get("reward_max") is not None
                ):
                    raise SynthesisError(
                        f"malformed control {case['id']} expects extraction_error but has reward bounds"
                    )
            elif expected.get("status") == "graded":
                if expected.get("reward_max", 1.0) > 0.2:
                    raise SynthesisError(
                        f"malformed graded control {case['id']} must demand reward <=0.2"
                    )
            else:
                raise SynthesisError(
                    f"malformed control {case['id']} must expect extraction_error or graded rejection"
                )
        step = case.get("step_index", 0)
        if type(step) is not int or not 0 <= step < step_count:
            raise SynthesisError(f"control {case['id']} has invalid step_index")
    for step in range(step_count):
        step_cases = [case for case in cases if case.get("step_index", 0) == step]
        step_categories = {case["category"] for case in step_cases}
        if (
            not {"known_correct", "empty_or_malformed", "plausible_wrong"}
            <= step_categories
            or not {
                "task_specific_shortcut",
                "reward_hack",
            }
            & step_categories
        ):
            raise SynthesisError(
                f"step {step} lacks a required positive/malformed/plausible/shortcut control category"
            )


def _checklist_ids(path: Path, capability_id: str, judge: bool) -> set[str]:
    lines = path.read_text().splitlines()

    def section(prefix: str) -> list[str]:
        start = next(
            (index for index, line in enumerate(lines) if line.startswith(prefix)),
            None,
        )
        if start is None:
            raise SynthesisError(f"construction checklist lacks section {prefix!r}")
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if lines[index].startswith("## ")
            ),
            len(lines),
        )
        return lines[start + 1 : end]

    selected = section(f"## {capability_id},")
    if judge:
        selected.extend(section("## Common native-judge gate"))
    identifiers = {
        match.group(1)
        for line in selected
        if (match := re.match(r"^- `([a-z0-9-]+)`:", line))
    }
    if not identifiers:
        raise SynthesisError("construction checklist section has no check IDs")
    return identifiers


def _build_acceptance_issues(
    item: dict[str, Any], bundle: Path, checklist: Path
) -> list[str]:
    record_path = bundle / "build-acceptance.json"
    if not record_path.is_file():
        return ["task/build-acceptance.json is missing"]
    try:
        record = _read_json(record_path)
    except (OSError, json.JSONDecodeError) as error:
        return [f"build acceptance is unreadable: {error}"]
    proposal = item["proposal"]
    provenance = item["provenance"]
    expected_identity = {
        "capability_id": proposal["capability_id"],
        "slot": proposal["slot"],
        "proposal_hash": item["proposal_hash"],
        "capability_record_hash": provenance["capability_record_hash"],
        "catalog_sha256": provenance["catalog_source"]["sha256"],
    }
    issues = []
    if record.get("schema_version") != "capability-build-acceptance-v1":
        issues.append("build acceptance has the wrong schema version")
    if record.get("identity") != expected_identity:
        issues.append(
            "build acceptance identity is not bound to the admitted candidate"
        )
    try:
        required = _checklist_ids(
            checklist,
            proposal["capability_id"],
            proposal["verification"] == "judge",
        )
    except (OSError, SynthesisError) as error:
        issues.append(str(error))
        return issues
    checks = record.get("checks")
    if not isinstance(checks, list) or any(
        not isinstance(check, dict) for check in checks
    ):
        issues.append("build acceptance checks must be a list of objects")
        return issues
    identifiers = [check.get("id") for check in checks]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != required:
        issues.append("build acceptance does not cover every required check exactly")
        return issues
    for check in checks:
        label = f"build check {check['id']}"
        if check.get("state") != "passed":
            issues.append(f"{label} is not passed")
        if not isinstance(check.get("claim"), str) or not check["claim"].strip():
            issues.append(f"{label} lacks a concrete claim")
        if not isinstance(check.get("summary"), str) or not check["summary"].strip():
            issues.append(f"{label} lacks a measured summary")
        artifacts = check.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            issues.append(f"{label} lacks hash-bound artifacts")
            continue
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                issues.append(f"{label} artifact {index} is malformed")
                continue
            issue = _evidence_artifact_issue(
                bundle,
                artifact.get("path"),
                artifact.get("sha256"),
                f"{label} artifact {index}",
            )
            if issue:
                issues.append(issue)
    return issues


def _controls_pass(
    controls: dict[str, Any], evidence: dict[str, Any], *, external: bool = False
) -> tuple[bool, list[str]]:
    definitions = {case["id"]: case for case in controls["cases"]}
    evidence_cases = evidence.get("cases", [])
    if not isinstance(evidence_cases, list) or len(
        {case.get("id") for case in evidence_cases if isinstance(case, dict)}
    ) != len(evidence_cases):
        return False, ["runtime evidence contains duplicate or malformed case IDs"]
    actual = {case.get("id"): case for case in evidence_cases}
    issues: list[str] = []
    if set(actual) != set(definitions):
        issues.append("runtime evidence does not cover the declared controls exactly")
        return False, issues
    for case_id, definition in definitions.items():
        record = actual[case_id]
        if external and record.get("source_author") != definition["source_author"]:
            issues.append(f"{case_id}: runtime evidence changed source_author")
        if external and record.get("category") != definition["category"]:
            issues.append(f"{case_id}: runtime evidence changed control category")
        if external and record.get("step_index", 0) != definition.get("step_index", 0):
            issues.append(f"{case_id}: runtime evidence changed step_index")
        result = record.get("result", {})
        status, reward = result.get("status"), result.get("reward")
        numeric_reward = (
            type(reward) in (int, float)
            and math.isfinite(reward)
            and 0.0 <= reward <= 1.0
        )
        if status == "graded" and not numeric_reward:
            issues.append(
                f"{case_id}: graded reward must be a finite non-boolean number in [0,1]"
            )
        if status != "graded" and reward is not None:
            issues.append(f"{case_id}: non-graded outcome must have null reward")
        expected = definition.get("expect", {})
        if expected.get("status") and status != expected["status"]:
            issues.append(
                f"{case_id}: status {status!r}, expected {expected['status']!r}"
            )
        if "reward_min" in expected and (
            not numeric_reward or reward < expected["reward_min"]
        ):
            issues.append(f"{case_id}: reward is below reward_min")
        if "reward_max" in expected and (
            not numeric_reward or reward > expected["reward_max"]
        ):
            issues.append(f"{case_id}: reward is above reward_max")
        for assertion in expected.get("assertions", []):
            try:
                value = result
                for part in assertion["path"]:
                    if isinstance(value, str):
                        value = json.loads(value)
                    value = value[part]
                target = assertion["equals"]
                compatible_types = type(value) is type(target) or (
                    type(value) in (int, float) and type(target) in (int, float)
                )
                if not compatible_types or value != target:
                    raise ValueError("criterion value differs")
            except (KeyError, IndexError, TypeError, ValueError):
                issues.append(
                    f"{case_id}: criterion assertion failed at {assertion['path']!r}"
                )
        if definition["class"] == "partial" and status != "graded":
            issues.append(f"{case_id}: partial-credit control was not graded")
        if definition["class"] == "positive" and (
            status != "graded" or not numeric_reward or reward < 0.8
        ):
            issues.append(f"{case_id}: positive control did not earn >=0.8")
        if (
            external
            and definition["class"] == "positive"
            and record.get("control_type") != "independent_solver"
        ):
            issues.append(
                f"{case_id}: positive control was not produced by the independent solver"
            )
        if definition["class"] == "negative" and (
            status != "graded" or not numeric_reward or reward > 0.2
        ):
            issues.append(f"{case_id}: negative control earned >0.2 or was not graded")
        if (
            external
            and definition["class"] in {"negative", "malformed", "partial"}
            and record.get("control_type")
            not in {"authored_adversarial_control", "independent_adversary"}
        ):
            issues.append(
                f"{case_id}: negative control did not execute through the adversarial path"
            )
        if status in {"invalid_task", "infra_error"}:
            issues.append(f"{case_id}: {status}")
    return not issues, issues


def _evidence_artifact_issue(
    base: Path, path_value: Any, digest_value: Any, label: str
) -> str | None:
    if not isinstance(path_value, str) or not isinstance(digest_value, str):
        return f"{label} lacks a raw artifact reference and digest"
    path = (base / path_value).resolve()
    if (
        base.resolve() not in path.parents
        or not path.is_file()
        or _sha256(path) != digest_value
    ):
        return f"{label} raw artifact is missing, escapes the evidence directory, or has the wrong digest"
    return None


def _bounded_incomplete_adversary(
    report: Any, evidence_root: Path, step_names: list[str]
) -> dict[str, Any] | None:
    """Recognize a retained, exhausted attack generation without calling it a grade.

    This is deliberately narrower than a generic Harbor failure.  A follow-up may
    sample a new independent attack suite, but no semantic attack adjudicator can
    review a candidate that never reached grading.
    """
    if (
        not isinstance(report, dict)
        or report.get("independent") is not True
        or report.get("state") != "needs_adjudication_or_retry"
    ):
        return None
    cases = report.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != 3
        or any(not isinstance(case, dict) for case in cases)
        or {case.get("strategy") for case in cases}
        != {"injection", "shortcut", "boundary"}
    ):
        return None
    incomplete: list[dict[str, Any]] = []
    expected_steps = set(enumerate(step_names))
    for case in cases:
        strategy = case["strategy"]
        if case.get("error") != "model_output_truncated":
            if case.get("error") is not None:
                return None
            steps = case.get("steps")
            if (
                not isinstance(steps, list)
                or len(steps) != len(step_names)
                or {
                    (step.get("step_index"), step.get("step_name"))
                    for step in steps
                    if isinstance(step, dict)
                }
                != expected_steps
            ):
                return None
            continue
        if case.get("steps") not in (None, []):
            return None
        trial_issue = _evidence_artifact_issue(
            evidence_root,
            case.get("trial_artifact"),
            case.get("trial_sha256"),
            f"adversary {strategy} incomplete trial",
        )
        if trial_issue:
            return None
        trial_path = (evidence_root / case["trial_artifact"]).resolve()
        try:
            trial = _read_json(trial_path)
            exception = trial.get("exception_info")
            if (
                trial.get("verifier_result") is not None
                or not isinstance(exception, dict)
                or exception.get("exception_type") != "RuntimeError"
                or exception.get("exception_message")
                != "Incomplete GLM solver output: length"
            ):
                return None
            request_log = trial_path.parent / "agent" / "glm-requests.jsonl"
            entries = [
                json.loads(line) for line in request_log.read_text().splitlines()
            ]
            phases = {entry.get("phase") for entry in entries}
            phase_history: dict[str, list[dict[str, Any]]] | None = None
            exhausted_phase: str | None = None
            if phases & {"boundary-planning", "boundary-finalization"}:
                # The planner and finalizer each make an independently bounded
                # transport call, so request_attempt intentionally restarts at
                # one for the finalizer.  Bind both phase histories, but judge
                # output exhaustion only within the phase that failed.
                if (
                    phases - {"boundary-planning", "boundary-finalization"}
                    or not entries
                ):
                    return None
                planning_entries = [
                    entry
                    for entry in entries
                    if entry.get("phase") == "boundary-planning"
                ]
                finalization_entries = [
                    entry
                    for entry in entries
                    if entry.get("phase") == "boundary-finalization"
                ]
                if not planning_entries or (
                    finalization_entries
                    and entries != planning_entries + finalization_entries
                ):
                    return None

                def parse_phase_attempts(
                    phase_entries: list[dict[str, Any]],
                    *,
                    require_length: bool,
                ) -> list[dict[str, Any]] | None:
                    parsed: list[dict[str, Any]] = []
                    for phase_entry in phase_entries:
                        request = phase_entry.get("request")
                        max_tokens = (
                            request.get("max_tokens")
                            if isinstance(request, dict)
                            else None
                        )
                        if (
                            type(phase_entry.get("request_attempt")) is not int
                            or not isinstance(request, dict)
                            or "max_tokens" not in request
                            or (
                                max_tokens is not None
                                and (type(max_tokens) is not int or max_tokens < 1)
                            )
                            or (
                                require_length
                                and phase_entry.get("finish_reason") != "length"
                            )
                        ):
                            return None
                        usage = (
                            phase_entry.get("usage")
                            if isinstance(phase_entry.get("usage"), dict)
                            else {}
                        )
                        details = usage.get("completion_tokens_details")
                        parsed.append(
                            {
                                "request_attempt": phase_entry["request_attempt"],
                                "max_tokens": max_tokens,
                                "finish_reason": phase_entry.get("finish_reason"),
                                "completion_tokens": usage.get("completion_tokens")
                                if isinstance(usage.get("completion_tokens"), int)
                                else None,
                                "reasoning_tokens": details.get("reasoning_tokens")
                                if isinstance(details, dict)
                                and isinstance(details.get("reasoning_tokens"), int)
                                else None,
                            }
                        )
                    if (
                        not 1 <= len(parsed) <= 4
                        or [item["request_attempt"] for item in parsed]
                        != list(range(1, len(parsed) + 1))
                        or any(
                            item["max_tokens"] is None and index != len(parsed) - 1
                            for index, item in enumerate(parsed)
                        )
                        or any(
                            parsed[index]["max_tokens"]
                            >= parsed[index + 1]["max_tokens"]
                            for index in range(len(parsed) - 1)
                            if parsed[index]["max_tokens"] is not None
                            and parsed[index + 1]["max_tokens"] is not None
                        )
                    ):
                        return None
                    return parsed

                planning_attempts = parse_phase_attempts(
                    planning_entries, require_length=not bool(finalization_entries)
                )
                if planning_attempts is None:
                    return None
                if not finalization_entries:
                    entries = planning_entries
                    attempts = planning_attempts
                    exhausted_phase = "boundary-planning"
                else:
                    # A planner may consume the configured 128K budget and then
                    # succeed on the retained remaining-context fallback. Its
                    # terminal draft is the only planner output passed onward.
                    planner_draft = (planning_entries[-1].get("message") or {}).get(
                        "content"
                    )
                    if (
                        planning_entries[-1].get("finish_reason") != "stop"
                        or not isinstance(planner_draft, str)
                        or not planner_draft.strip()
                        or any(
                            entry.get("finish_reason") != "length"
                            for entry in planning_entries[:-1]
                        )
                    ):
                        return None
                    finalization_attempts = parse_phase_attempts(
                        finalization_entries, require_length=True
                    )
                    if finalization_attempts is None:
                        return None
                    entries = finalization_entries
                    attempts = finalization_attempts
                    exhausted_phase = "boundary-finalization"
                phase_history = {
                    "boundary-planning": planning_attempts,
                    "boundary-finalization": (
                        finalization_attempts if finalization_entries else []
                    ),
                }
            else:
                attempts = []
                for entry in entries:
                    request = entry.get("request")
                    max_tokens = (
                        request.get("max_tokens") if isinstance(request, dict) else None
                    )
                    if (
                        type(entry.get("request_attempt")) is not int
                        or not isinstance(request, dict)
                        or "max_tokens" not in request
                        or (
                            max_tokens is not None
                            and (type(max_tokens) is not int or max_tokens < 1)
                        )
                        or entry.get("finish_reason") != "length"
                    ):
                        return None
                    usage = (
                        entry.get("usage")
                        if isinstance(entry.get("usage"), dict)
                        else {}
                    )
                    completion = usage.get("completion_tokens")
                    details = usage.get("completion_tokens_details")
                    attempts.append(
                        {
                            "request_attempt": entry["request_attempt"],
                            "max_tokens": max_tokens,
                            "finish_reason": entry["finish_reason"],
                            "completion_tokens": completion
                            if isinstance(completion, int)
                            else None,
                            "reasoning_tokens": details.get("reasoning_tokens")
                            if isinstance(details, dict)
                            and isinstance(details.get("reasoning_tokens"), int)
                            else None,
                        }
                    )
            if (
                not 1 <= len(attempts) <= 4
                or [entry["request_attempt"] for entry in attempts]
                != list(range(1, len(attempts) + 1))
                or any(
                    entry["max_tokens"] is None and index != len(attempts) - 1
                    for index, entry in enumerate(attempts)
                )
                or any(
                    attempts[index]["max_tokens"] >= attempts[index + 1]["max_tokens"]
                    for index in range(len(attempts) - 1)
                    if attempts[index]["max_tokens"] is not None
                    and attempts[index + 1]["max_tokens"] is not None
                )
            ):
                return None
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        incomplete.append(
            {
                "strategy": strategy,
                "trial_artifact": case["trial_artifact"],
                "trial_sha256": case["trial_sha256"],
                "request_log_artifact": str(request_log.relative_to(evidence_root)),
                "request_log_sha256": _sha256(request_log),
                "attempts": attempts,
                "exhausted_phase": exhausted_phase,
                "phase_history": phase_history,
            }
        )
    if not incomplete:
        return None
    return {
        "schema_version": "capability-incomplete-adversary-v1",
        "state": "bounded_model_output_truncated",
        "failed_strategies": [entry["strategy"] for entry in incomplete],
        "failures": incomplete,
        "required_follow_up": "fresh_independent_adversary_suite_only",
    }


def _incomplete_adversary_from_evidence(
    evidence: dict[str, Any], evidence_path: Path, harbor: Path
) -> dict[str, Any] | None:
    """Return only a fully bound exhausted-generator classification."""
    attestation = evidence.get("attestation")
    if not isinstance(attestation, dict):
        return None
    artifact = attestation.get("adversary_artifact")
    digest_value = attestation.get("adversary_artifact_sha256")
    if _evidence_artifact_issue(
        evidence_path.parent, artifact, digest_value, "independent adversary"
    ):
        return None
    try:
        report = _read_json((evidence_path.parent / artifact).resolve())
        steps = _read_json(harbor / "manifest.json")["step_names"]
        if not isinstance(steps, list) or not all(
            isinstance(step, str) for step in steps
        ):
            return None
        return _bounded_incomplete_adversary(report, evidence_path.parent, steps)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _container_step_indices(bundle: Path) -> set[int]:
    specification = _read_json(bundle / "specification.json")
    indices = set()
    for index, step in enumerate(specification.get("steps", [])):
        verifier = step.get("verifier", {}) if isinstance(step, dict) else {}
        if verifier.get("kind") == "code_answer":
            verifier = verifier.get("verifier", {})
        if verifier.get("runtime", {}).get("kind") == "container":
            indices.add(index)
    return indices


def _container_supervisor_runtimes(bundle: Path) -> dict[int, tuple[str, str]]:
    """Return every container verifier's image/interpreter pair."""
    specification = _read_json(bundle / "specification.json")
    result = {}
    for index, step in enumerate(specification.get("steps", [])):
        verifier = step.get("verifier", {}) if isinstance(step, dict) else {}
        if verifier.get("kind") == "code_answer":
            verifier = verifier.get("verifier", {})
        runtime = verifier.get("runtime", {}) if isinstance(verifier, dict) else {}
        if not isinstance(runtime, dict):
            continue
        image = runtime.get("image") if isinstance(runtime, dict) else None
        supervisor = (
            runtime.get("supervisor_python", "python3")
            if isinstance(runtime, dict)
            else "python3"
        )
        if (
            runtime.get("kind") == "container"
            and isinstance(image, str)
            and isinstance(supervisor, str)
        ):
            result[index] = (image, supervisor)
    return result


def _nondefault_container_supervisors(bundle: Path) -> dict[int, tuple[str, str]]:
    """Return container runtimes which must carry supervisor-aware receipts."""
    return {
        index: runtime
        for index, runtime in _container_supervisor_runtimes(bundle).items()
        if runtime[1] != "python3"
    }


def _compatible_private_verifier_record(
    result: Any,
    adapter_sha256: str,
    runtime: tuple[str, str] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the exact legacy or supervisor-aware receipt form in *result*.

    Default-python evidence may predate recipe fields.  New runtime evidence
    emits them for every container verifier, including default Python.  Once
    any strong field is present, validate the complete recipe-bound form;
    nondefault supervisors always require that form.
    """
    if runtime is None:
        return _private_verifier_record(result, adapter_sha256)
    image, supervisor = runtime
    detail = result.get("detail") if isinstance(result, dict) else None
    strong_keys = {
        "verifier_supervisor_python",
        "verifier_bootstrap_command_sha256",
        "verifier_snapshot_recipe_sha256",
        "verifier_requested_resource_profile",
    }
    strong_present = isinstance(detail, dict) and any(key in detail for key in strong_keys)
    if supervisor != "python3" or strong_present:
        return _private_verifier_record(
            result,
            adapter_sha256,
            runtime_image=image,
            supervisor_python=supervisor,
        )
    return _private_verifier_record(result, adapter_sha256)


def _private_verifier_record(
    result: Any,
    adapter_sha256: str,
    *,
    runtime_image: str | None = None,
    supervisor_python: str = "python3",
) -> tuple[dict[str, Any] | None, str | None]:
    detail = result.get("detail") if isinstance(result, dict) else None
    expected_bootstrap = verifier_bootstrap_sha256()
    if (
        not isinstance(detail, dict)
        or detail.get("verifier_isolation") not in sandbox_provider.NETWORK_BLOCKED_ISOLATION
        or detail.get("verifier_adapter_sha256") != adapter_sha256
        or detail.get("verifier_bootstrap_sha256") != expected_bootstrap
    ):
        return None, "lacks bound Daytona private-verifier evidence"
    sandbox_id = detail.get("verifier_sandbox_id")
    snapshot = detail.get("verifier_snapshot")
    if not isinstance(sandbox_id, str) or not sandbox_id:
        return None, "lacks a private-verifier sandbox ID"
    if not isinstance(snapshot, str) or not snapshot:
        return None, "lacks a private-verifier snapshot"
    cleanup = detail.get("verifier_cleanup")
    if "verifier_cleanup" in detail and (
        not isinstance(cleanup, dict)
        or cleanup.get("state") != "deleted"
        or not isinstance(cleanup.get("attempts"), list)
        or not cleanup["attempts"]
        or not isinstance(cleanup.get("observations"), list)
        or not cleanup["observations"]
        or not all(
            isinstance(item, dict)
            for item in cleanup["attempts"] + cleanup["observations"]
        )
        or cleanup["observations"][-1].get("state") != "not_found"
    ):
        return None, "has unconfirmed private-verifier cleanup"
    record = {
        "sandbox_id": sandbox_id,
        "snapshot": snapshot,
        "adapter_sha256": adapter_sha256,
        "bootstrap_sha256": expected_bootstrap,
    }
    if cleanup is not None:
        record["cleanup"] = cleanup
    profile = None
    if "verifier_requested_resource_profile" in detail:
        try:
            profile = profile_from_receipt(detail["verifier_requested_resource_profile"])
        except ValueError:
            return None, "has a malformed verifier resource profile"
    if runtime_image is not None:
        recipe = verifier_snapshot_recipe(runtime_image, supervisor_python)
        recipe_sha256 = hashlib.sha256(recipe.encode()).hexdigest()
        if profile is not None:
            expected_snapshot = snapshot_name("cap-verifier", recipe, profile)
        else:
            # The recipe-only identity is accepted solely for historical
            # receipts that have no requested resource profile at all.
            expected_snapshot = f"cap-verifier-{recipe_sha256[:20]}"
        if (
            detail.get("verifier_supervisor_python") != supervisor_python
            or detail.get("verifier_bootstrap_command_sha256")
            != verifier_bootstrap_sha256(supervisor_python)
            or detail.get("verifier_snapshot_recipe_sha256") != recipe_sha256
            or snapshot != expected_snapshot
        ):
            return None, "lacks its supervisor-aware snapshot recipe"
        record.update(
            supervisor_python=supervisor_python,
            bootstrap_command_sha256=verifier_bootstrap_sha256(supervisor_python),
            snapshot_recipe_sha256=recipe_sha256,
        )
    return record, None


def _composite_private_record(
    result: Any,
    checks: list[dict[str, Any]],
    adapter_sha256: str,
    policy_sha256: str,
    config_sha256: str,
) -> tuple[dict[str, Any] | None, str | None]:
    detail = result.get("detail") if isinstance(result, dict) else None
    machine_results = (
        detail.get("machine_results") if isinstance(detail, dict) else None
    )
    if (
        not isinstance(detail, dict)
        or detail.get("composite_adapter_sha256") != adapter_sha256
        or detail.get("composite_policy_sha256") != policy_sha256
        or detail.get("composite_config_sha256") != config_sha256
        or not isinstance(machine_results, list)
        or any(not isinstance(item, dict) for item in machine_results)
        or [item.get("id") for item in machine_results]
        != [check["id"] for check in checks]
    ):
        return None, "lacks bound composite machine-check evidence"
    records = []
    for machine_result, check in zip(machine_results, checks, strict=True):
        supervisor_python = check.get("supervisor_python", "python3")
        record, issue = _private_verifier_record(
            machine_result,
            _sha256(Path(__file__).with_name("daytona_verifier.py")),
            **(
                {
                    "runtime_image": check["image"],
                    "supervisor_python": supervisor_python,
                }
                if supervisor_python != "python3"
                else {}
            ),
        )
        if issue:
            return None, f"machine check {machine_result.get('id')} {issue}"
        records.append({"id": machine_result["id"], **record})
    return {
        "adapter_sha256": adapter_sha256,
        "policy_sha256": policy_sha256,
        "config_sha256": config_sha256,
        "machine_checks": records,
    }, None


def _private_ids(record: dict[str, Any]) -> list[str]:
    if "machine_checks" in record:
        return [item["sandbox_id"] for item in record["machine_checks"]]
    return [record["sandbox_id"]]


def _attestation_issues(
    evidence: dict[str, Any],
    evidence_path: Path,
    bundle: Path,
    harbor: Path,
    controls: Path,
    specification_sha256: str,
    *,
    require_solver_pass: bool = True,
    require_independent_adversary: bool = True,
) -> list[str]:
    lock = _read_json(SOURCE_LOCK)
    attestation = evidence.get("attestation", {})
    expected = {
        "kind": "harbor_control_run",
        "harbor_revision": lock["transitive_revisions"]["harbor"],
        "specification_sha256": specification_sha256,
        "package_sha256": _tree_sha256(harbor),
        "package_manifest_sha256": _sha256(harbor / "manifest.json"),
        "controls_sha256": _sha256(controls),
    }
    judge_policy = bundle / "judge-policy.json"
    if judge_policy.is_file():
        expected["judge_policy_sha256"] = _sha256(judge_policy)
    binding_document = _read_json(bundle / "binding.json")
    container_steps = _container_step_indices(bundle)
    container_supervisors = _container_supervisor_runtimes(bundle)
    composite_path = bundle / "composite-verifier.json"
    composite_steps = {}
    composite_adapter_sha256 = _sha256(
        Path(__file__).with_name("composite_verifier.py")
    )
    composite_policy_sha256 = _sha256(Path(__file__).with_name("composite_policy.py"))
    composite_config_sha256 = None
    if composite_path.is_file():
        from .composite_policy import validate_composite_config

        composite_config_sha256 = _sha256(composite_path)
        composite_steps = validate_composite_config(
            _read_json(composite_path),
            specification_sha256=_sha256(bundle / "specification.json"),
            adapter_sha256=composite_adapter_sha256,
            policy_sha256=composite_policy_sha256,
            step_count=len(_read_json(harbor / "manifest.json")["step_names"]),
        )

    def private_record(result: Any, step_index: int):
        if step_index in composite_steps:
            return _composite_private_record(
                result,
                composite_steps[step_index]["machine_checks"],
                composite_adapter_sha256,
                composite_policy_sha256,
                composite_config_sha256,
            )
        return _compatible_private_verifier_record(
            result, verifier_adapter_sha256, container_supervisors.get(step_index)
        )

    verifier_adapter_sha256 = _sha256(Path(__file__).with_name("daytona_verifier.py"))
    if binding_document.get("environment", {}).get("kind") == "docker":
        expected.update(
            {
                "daytona_environment_adapter_sha256": _sha256(
                    Path(__file__).with_name("daytona_environment.py")
                ),
            }
        )
    if container_steps or composite_steps:
        expected.update(
            {
                "daytona_verifier_adapter_sha256": verifier_adapter_sha256,
                "daytona_verifier_bootstrap_sha256": verifier_bootstrap_sha256(),
            }
        )
    if composite_steps:
        from .composite_extension import runtime_attestation_bindings

        expected.update(runtime_attestation_bindings(composite_config_sha256))
    issues = [
        f"attestation {key} is not bound to the exported artifact"
        for key, value in expected.items()
        if attestation.get(key) != value
    ]
    isolation = attestation.get("isolation", {})
    if (
        isolation.get("network") != "blocked"
        or isolation.get("fresh_environment_per_case") is not True
    ):
        issues.append("attestation lacks network-blocked fresh isolation per case")
    isolation_cases = isolation.get("cases")
    candidate_sandbox_ids = []
    definitions = _read_json(controls)["cases"]
    expected_case_ids = {case["id"] for case in definitions}
    if (
        not isinstance(isolation_cases, list)
        or len(isolation_cases) != len(expected_case_ids)
        or any(not isinstance(case, dict) for case in isolation_cases)
        or {case.get("case_id") for case in isolation_cases if isinstance(case, dict)}
        != expected_case_ids
    ):
        issues.append("isolation evidence does not cover the declared controls exactly")
    else:
        sandbox_ids = []
        for case in isolation_cases:
            mechanism = case.get("mechanism")
            if mechanism not in {
                "no-tool-host-boundary",
                "shellsim-no-host-access",
                *sandbox_provider.NETWORK_BLOCKED_ISOLATION,
            }:
                issues.append(
                    f"isolation case {case.get('case_id')} has an unknown mechanism"
                )
            if mechanism in sandbox_provider.NETWORK_BLOCKED_ISOLATION:
                provider_issue = _evidence_artifact_issue(
                    evidence_path.parent,
                    case.get("provider_artifact"),
                    case.get("provider_artifact_sha256"),
                    f"Daytona isolation case {case.get('case_id')}",
                )
                if provider_issue:
                    issues.append(provider_issue)
                if (
                    not isinstance(case.get("sandbox_id"), str)
                    or not case["sandbox_id"]
                ):
                    issues.append(
                        f"Daytona isolation case {case.get('case_id')} lacks a sandbox ID"
                    )
                else:
                    sandbox_ids.append(case["sandbox_id"])
                    candidate_sandbox_ids.append(case["sandbox_id"])
        if len(sandbox_ids) != len(set(sandbox_ids)):
            issues.append("Daytona isolation reused a sandbox across control cases")
    private_verifier_ids = []
    solver = attestation.get("solver", {})
    if solver.get("independent") is not True or not solver.get("run_id"):
        issues.append("attestation lacks an independently executed solver run")
    if require_solver_pass and solver.get("state") != "passed":
        issues.append("independent solver needs adjudication after bounded retries")
    oracle = attestation.get("oracle", {})
    if oracle.get("authored") is not True or not oracle.get("run_id"):
        issues.append("attestation lacks an executed authored reference control")
    oracle_issue = _evidence_artifact_issue(
        evidence_path.parent,
        attestation.get("oracle_artifact"),
        attestation.get("oracle_artifact_sha256"),
        "authored reference control",
    )
    if oracle_issue:
        issues.append(oracle_issue)
    else:
        oracle_report = _read_json(
            (evidence_path.parent / attestation["oracle_artifact"]).resolve()
        )
        positive_definitions = {
            case["id"]: case for case in definitions if case["class"] == "positive"
        }
        oracle_cases = oracle_report.get("cases")
        if (
            oracle_report.get("state") != "passed"
            or oracle_report.get("source") != "authored_reference_controls"
            or not isinstance(oracle_cases, list)
            or any(not isinstance(case, dict) for case in oracle_cases)
            or {case.get("case_id") for case in oracle_cases}
            != set(positive_definitions)
        ):
            issues.append("authored reference controls lack exact positive coverage")
        else:
            for oracle_case in oracle_cases:
                label = f"authored reference {oracle_case.get('case_id')}"
                for artifact_kind in ("trial", "grading"):
                    artifact_issue = _evidence_artifact_issue(
                        evidence_path.parent,
                        oracle_case.get(f"{artifact_kind}_artifact"),
                        oracle_case.get(f"{artifact_kind}_sha256"),
                        f"{label} {artifact_kind}",
                    )
                    if artifact_issue:
                        issues.append(artifact_issue)
                grading_path = oracle_case.get("grading_artifact")
                if isinstance(grading_path, str) and not _evidence_artifact_issue(
                    evidence_path.parent,
                    grading_path,
                    oracle_case.get("grading_sha256"),
                    label,
                ):
                    raw_oracle = _read_json(
                        (evidence_path.parent / grading_path).resolve()
                    )
                    reward = raw_oracle.get("reward")
                    if (
                        oracle_case.get("result") != raw_oracle
                        or raw_oracle.get("status") != "graded"
                        or type(reward) not in (int, float)
                        or not math.isfinite(reward)
                        or reward < 0.8
                    ):
                        issues.append(f"{label} did not prove the grader")
                    step_index = oracle_case.get("step_index", 0)
                    if step_index in container_steps or step_index in composite_steps:
                        record, private_issue = private_record(raw_oracle, step_index)
                        if private_issue:
                            issues.append(f"{label} {private_issue}")
                        else:
                            if oracle_case.get("private_verifier") != record:
                                issues.append(
                                    f"{label} private-verifier summary differs from its raw result"
                                )
                            private_verifier_ids.extend(_private_ids(record))
                oracle_isolation = oracle_case.get("isolation")
                # Evidence may come from an earlier Daytona run or a frozen
                # bundle, so a docker binding accepts either proven mechanism.
                expected_mechanisms = {
                    "none": {"no-tool-host-boundary"},
                    "shellsim": {"shellsim-no-host-access"},
                    "docker": sandbox_provider.NETWORK_BLOCKED_ISOLATION,
                }[binding_document.get("environment", {}).get("kind")]
                if (
                    not isinstance(oracle_isolation, dict)
                    or oracle_isolation.get("mechanism") not in expected_mechanisms
                ):
                    issues.append(f"{label} has the wrong isolation mechanism")
                if binding_document.get("environment", {}).get("kind") == "docker":
                    if not isinstance(oracle_isolation, dict):
                        issues.append(f"{label} lacks Daytona isolation")
                    else:
                        provider_issue = _evidence_artifact_issue(
                            evidence_path.parent,
                            oracle_isolation.get("provider_artifact"),
                            oracle_isolation.get("provider_artifact_sha256"),
                            f"{label} Daytona isolation",
                        )
                        if provider_issue:
                            issues.append(provider_issue)
                        sandbox_id = oracle_isolation.get("sandbox_id")
                        if not isinstance(sandbox_id, str) or not sandbox_id:
                            issues.append(f"{label} lacks a Daytona sandbox ID")
                        else:
                            candidate_sandbox_ids.append(sandbox_id)
    if require_independent_adversary:
        adversarial = attestation.get("adversarial", {})
        if adversarial.get("authored_controls_executed") is not True:
            issues.append("attestation lacks executed authored adversarial controls")
        adversary_issue = _evidence_artifact_issue(
            evidence_path.parent,
            attestation.get("adversary_artifact"),
            attestation.get("adversary_artifact_sha256"),
            "independent adversary",
        )
        if adversary_issue:
            issues.append(adversary_issue)
        else:
            adversary_path = (
                evidence_path.parent / attestation["adversary_artifact"]
            ).resolve()
            adversary_report = _read_json(adversary_path)
            manifest_step_names = _read_json(harbor / "manifest.json")["step_names"]
            incomplete_adversary = _bounded_incomplete_adversary(
                adversary_report, evidence_path.parent, manifest_step_names
            )
            if (
                adversarial.get("independent_attack_executed") is not True
                and incomplete_adversary is None
            ):
                issues.append("attestation lacks an independent adversarial attack")
            if adversary_report.get("state") != "passed" and incomplete_adversary is None:
                issues.append("independent adversary report needs adjudication or retry")
            if adversary_report.get("independent") is not True:
                issues.append("independent adversary report is not independent")
            attack_cases = adversary_report.get("cases")
            if (
                not isinstance(attack_cases, list)
                or len(attack_cases) != 3
                or {case.get("strategy") for case in attack_cases}
                != {"injection", "shortcut", "boundary"}
            ):
                issues.append("independent adversary report has incomplete attack coverage")
            else:
                adversary_sandbox_ids = []
                for attack_case in attack_cases:
                    attack_issue = _evidence_artifact_issue(
                        evidence_path.parent,
                        attack_case.get("trial_artifact"),
                        attack_case.get("trial_sha256"),
                        f"adversary {attack_case.get('strategy')} trial",
                    )
                    if attack_issue:
                        issues.append(attack_issue)
                    attack_steps = attack_case.get("steps")
                    strategy_incomplete = (
                        incomplete_adversary is not None
                        and attack_case.get("strategy")
                        in incomplete_adversary["failed_strategies"]
                    )
                    if strategy_incomplete:
                        if attack_steps not in (None, []):
                            issues.append(
                                f"adversary {attack_case.get('strategy')} incomplete "
                                "generation unexpectedly contains step outcomes"
                            )
                    elif (
                        not isinstance(attack_steps, list)
                        or len(attack_steps) != len(manifest_step_names)
                        or {
                            (step.get("step_index"), step.get("step_name"))
                            for step in attack_steps
                            if isinstance(step, dict)
                        }
                        != set(enumerate(manifest_step_names))
                    ):
                        issues.append(
                            f"adversary {attack_case.get('strategy')} lacks exact step coverage"
                        )
                    else:
                        for attack_step in attack_steps:
                            for artifact_kind in ("transcript", "grading"):
                                attack_issue = _evidence_artifact_issue(
                                    evidence_path.parent,
                                    attack_step.get(f"{artifact_kind}_artifact"),
                                    attack_step.get(f"{artifact_kind}_sha256"),
                                    f"adversary {attack_case.get('strategy')} step "
                                    f"{attack_step.get('step_index')} {artifact_kind}",
                                )
                                if attack_issue:
                                    issues.append(attack_issue)
                                elif artifact_kind == "grading":
                                    grading_path = (
                                        evidence_path.parent
                                        / attack_step["grading_artifact"]
                                    ).resolve()
                                    raw_attack_result = _read_json(grading_path)
                                    if attack_step.get("result") != raw_attack_result:
                                        issues.append(
                                            f"adversary {attack_case.get('strategy')} step "
                                            f"{attack_step.get('step_index')} summary differs from its raw result"
                                        )
                                    attack_step_index = attack_step.get("step_index")
                                    if (
                                        attack_step_index in container_steps
                                        or attack_step_index in composite_steps
                                    ):
                                        record, private_issue = private_record(
                                            raw_attack_result, attack_step_index
                                        )
                                        if private_issue:
                                            issues.append(
                                                f"adversary {attack_case.get('strategy')} step "
                                                f"{attack_step.get('step_index')} {private_issue}"
                                            )
                                        else:
                                            if (
                                                attack_step.get("private_verifier")
                                                != record
                                            ):
                                                issues.append(
                                                    f"adversary {attack_case.get('strategy')} step "
                                                    f"{attack_step.get('step_index')} private-verifier summary "
                                                    "differs from its raw result"
                                                )
                                            private_verifier_ids.extend(
                                                _private_ids(record)
                                            )
                    attack_isolation = attack_case.get("isolation")
                    if (
                        binding_document.get("environment", {}).get("kind") == "docker"
                        and not strategy_incomplete
                    ):
                        if not isinstance(attack_isolation, dict):
                            issues.append(
                                f"adversary {attack_case.get('strategy')} lacks Daytona isolation"
                            )
                        else:
                            attack_provider_issue = _evidence_artifact_issue(
                                evidence_path.parent,
                                attack_isolation.get("provider_artifact"),
                                attack_isolation.get("provider_artifact_sha256"),
                                f"adversary {attack_case.get('strategy')} Daytona isolation",
                            )
                            if attack_provider_issue:
                                issues.append(attack_provider_issue)
                            if attack_isolation.get("sandbox_id"):
                                adversary_sandbox_ids.append(attack_isolation["sandbox_id"])
                                candidate_sandbox_ids.append(attack_isolation["sandbox_id"])
                control_sandbox_ids = [
                    case["sandbox_id"]
                    for case in isolation_cases or []
                    if isinstance(case, dict) and case.get("sandbox_id")
                ]
                all_sandbox_ids = control_sandbox_ids + adversary_sandbox_ids
                if len(all_sandbox_ids) != len(set(all_sandbox_ids)):
                    issues.append(
                        "Daytona isolation reused a sandbox across controls or attacks"
                    )
            if incomplete_adversary is not None:
                issues.append(INCOMPLETE_ADVERSARY_ISSUE)
    else:
        diagnostic = attestation.get("adversarial", {})
        if diagnostic.get("diagnostic_new_attacks") != "not_run_primary_bound":
            issues.append("diagnostic run lacks the primary-bound adversary marker")
        if attestation.get("adversary_artifact") is not None or attestation.get("adversary_artifact_sha256") is not None:
            issues.append("diagnostic run must not inherit an adversary artifact")
    raw_issue = _evidence_artifact_issue(
        evidence_path.parent,
        attestation.get("run_artifact"),
        attestation.get("run_artifact_sha256"),
        "Harbor run",
    )
    if raw_issue:
        issues.append(raw_issue)
    solver_issue = _evidence_artifact_issue(
        evidence_path.parent,
        attestation.get("solver_artifact"),
        attestation.get("solver_artifact_sha256"),
        "independent solver",
    )
    if solver_issue:
        issues.append(solver_issue)
    else:
        solver_attempts = _read_json(
            (evidence_path.parent / attestation["solver_artifact"]).resolve()
        )
        positive_ids = {
            case["id"] for case in definitions if case["class"] == "positive"
        }
        retry_limit = solver.get("retry_limit")
        if (
            not isinstance(solver_attempts, list)
            or not solver_attempts
            or any(not isinstance(attempt, dict) for attempt in solver_attempts)
            or {attempt.get("case_id") for attempt in solver_attempts} != positive_ids
            or type(retry_limit) is not int
            or retry_limit < 1
        ):
            issues.append("independent solver artifact has invalid case coverage")
        else:
            grouped = {
                case_id: [
                    attempt
                    for attempt in solver_attempts
                    if attempt.get("case_id") == case_id
                ]
                for case_id in positive_ids
            }
            for case_id, attempts in grouped.items():
                if len(attempts) > retry_limit or [
                    attempt.get("attempt") for attempt in attempts
                ] != list(range(1, len(attempts) + 1)):
                    issues.append(
                        f"independent solver {case_id} exceeded or misnumbered bounded retries"
                    )
                passed_attempt = False
                for attempt in attempts:
                    label = (
                        f"independent solver {case_id} attempt {attempt.get('attempt')}"
                    )
                    selected_control_artifacts = {
                        case.get("artifact") for case in evidence.get("cases", [])
                    }
                    selected_attempt = (
                        attempt.get("grading_artifact") in selected_control_artifacts
                    )
                    for artifact_kind in ("trial", "transcript", "grading"):
                        artifact_issue = _evidence_artifact_issue(
                            evidence_path.parent,
                            attempt.get(f"{artifact_kind}_artifact"),
                            attempt.get(f"{artifact_kind}_sha256"),
                            f"{label} {artifact_kind}",
                        )
                        if artifact_issue:
                            issues.append(artifact_issue)
                    grading_path = attempt.get("grading_artifact")
                    if isinstance(grading_path, str) and not _evidence_artifact_issue(
                        evidence_path.parent,
                        grading_path,
                        attempt.get("grading_sha256"),
                        label,
                    ):
                        raw_solver = _read_json(
                            (evidence_path.parent / grading_path).resolve()
                        )
                        reward = raw_solver.get("reward")
                        if attempt.get("result") != raw_solver:
                            issues.append(
                                f"{label} summary differs from its raw result"
                            )
                        if (
                            raw_solver.get("status") == "graded"
                            and type(reward) in (int, float)
                            and math.isfinite(reward)
                            and reward >= 0.8
                        ):
                            passed_attempt = True
                        step_index = attempt.get("step_index", 0)
                        if (
                            step_index in container_steps
                            or step_index in composite_steps
                        ):
                            record, private_issue = private_record(
                                raw_solver, step_index
                            )
                            if private_issue:
                                issues.append(f"{label} {private_issue}")
                            else:
                                if attempt.get("private_verifier") != record:
                                    issues.append(
                                        f"{label} private-verifier summary differs from its raw result"
                                    )
                                if not selected_attempt:
                                    private_verifier_ids.extend(_private_ids(record))
                    solver_isolation = attempt.get("isolation")
                    if binding_document.get("environment", {}).get("kind") == "docker":
                        if not isinstance(solver_isolation, dict):
                            issues.append(f"{label} lacks Daytona isolation")
                        else:
                            provider_issue = _evidence_artifact_issue(
                                evidence_path.parent,
                                solver_isolation.get("provider_artifact"),
                                solver_isolation.get("provider_artifact_sha256"),
                                f"{label} Daytona isolation",
                            )
                            if provider_issue:
                                issues.append(provider_issue)
                            sandbox_id = solver_isolation.get("sandbox_id")
                            if not isinstance(sandbox_id, str) or not sandbox_id:
                                issues.append(f"{label} lacks a Daytona sandbox ID")
                            elif not selected_attempt:
                                candidate_sandbox_ids.append(sandbox_id)
                if solver.get("state") == "passed" and not passed_attempt:
                    issues.append(
                        f"independent solver {case_id} has no successful bounded attempt"
                    )
    for case in evidence.get("cases", []):
        case_issue = _evidence_artifact_issue(
            evidence_path.parent,
            case.get("artifact"),
            case.get("artifact_sha256"),
            f"case {case.get('id')}",
        )
        if case_issue:
            issues.append(case_issue)
            continue
        case_artifact = (evidence_path.parent / case["artifact"]).resolve()
        raw_result = _read_json(case_artifact)
        if case.get("result") != raw_result:
            issues.append(f"case {case.get('id')} summary differs from its raw result")
        step_index = case.get("step_index", 0)
        if step_index in container_steps or step_index in composite_steps:
            record, private_issue = private_record(raw_result, step_index)
            if private_issue:
                issues.append(f"case {case.get('id')} {private_issue}")
            else:
                if case.get("private_verifier") != record:
                    issues.append(
                        f"case {case.get('id')} private-verifier summary differs from its raw result"
                    )
                private_verifier_ids.extend(_private_ids(record))
    if len(private_verifier_ids) != len(set(private_verifier_ids)):
        issues.append("Daytona reused a private-verifier sandbox across controls")
    if len(candidate_sandbox_ids) != len(set(candidate_sandbox_ids)):
        issues.append("Daytona reused a candidate sandbox across runtime trials")
    if set(private_verifier_ids) & set(candidate_sandbox_ids):
        issues.append("Daytona reused a candidate sandbox for private verification")
    if not (bundle / "specification.json").is_file():
        issues.append("specification disappeared during runtime validation")
    return issues


def _external_controls(
    runner: list[str],
    bundle: Path,
    harbor: Path,
    controls: Path,
    evidence: Path,
    timeout: int,
) -> dict:
    completed = _run(
        [
            *runner,
            "--package",
            str(harbor),
            "--bundle",
            str(bundle),
            "--controls",
            str(controls),
            "--output",
            str(evidence),
        ],
        timeout=timeout,
    )
    if completed.returncode and not (completed.returncode == 2 and evidence.is_file()):
        raise SynthesisError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or "runtime runner failed"
        )
    if not evidence.is_file():
        raise SynthesisError("runtime runner did not write evidence")
    return _read_json(evidence)


def _repeated_quality_diagnostics(
    item_root: Path,
    toolchain: OfficialToolchain,
    validation_timeout: int,
    daytona_tools: Path | None,
) -> dict:
    """Compose runtime, fixed-grading and task-reset evidence for GLM review."""
    measured = _repeat_and_grading_diagnostics(
        item_root, toolchain, validation_timeout, daytona_tools
    )
    if measured.get("state") != "ready" and measured.get("reviewable") is not True:
        return measured
    accepted = _read_json(item_root / "contract/accepted.json")
    # Proposals name this surface "container"; only the lowered binding uses
    # "docker". Dispatch from the validated proposal vocabulary here.
    if accepted["proposal"].get("environment") == "container":
        from .reset_runner import run_frozen_reset

        reset = run_frozen_reset(item_root, toolchain, validation_timeout, daytona_tools)
    else:
        from .non_docker_reset import run_frozen_non_docker_reset

        reset = run_frozen_non_docker_reset(
            item_root, toolchain, validation_timeout,
            shellsim_bridge=os.environ.get("TASKCOMPENDIUM_SHELLSIM_BRIDGE"),
        )
    result = {
        **measured,
        "reset_diagnostics": {
            key: value for key, value in reset.items() if key != "extra_files"
        },
        "extra_files": {**measured.get("extra_files", {}), **reset.get("extra_files", {})},
    }
    if reset.get("state") != "ready":
        result["state"] = "pending"
        result["reviewable"] = (
            (measured.get("state") == "ready" or measured.get("reviewable") is True)
            and reset.get("state") == "semantic_failed"
            and reset.get("reviewable") is True
        )
        result["issues"] = [
            *measured.get("issues", []),
            "task reset diagnostics are " + str(reset.get("state")),
        ]
    # Public reset consistency does not certify absence of private files,
    # credentials, or external-directory state. Keep those review conditions.
    return result


def _repeat_and_grading_diagnostics(
    item_root: Path,
    toolchain: OfficialToolchain,
    validation_timeout: int,
    daytona_tools: Path | None,
) -> dict:
    """Run measured repeatability before semantic review, with no repair spend."""
    from .diagnostics import run_repeated_diagnostics

    helper = item_root / "workspace/tools/daytona/dt.py"
    if not helper.is_file() and daytona_tools is not None:
        helper = daytona_tools / "dt.py"
    bridge = os.environ.get("TASKCOMPENDIUM_SHELLSIM_BRIDGE")
    resources = item_root / "workspace/task/candidate-resources.json"
    binding_kind = (
        _read_json(item_root / "workspace/task/binding.json")["environment"]["kind"]
        if resources.is_file()
        else None
    )
    repeated = run_repeated_diagnostics(
        item_root,
        toolchain.source_package_root or toolchain.package_root,
        validation_timeout,
        daytona_helper=helper if helper.is_file() else None,
        shellsim_bridge=Path(bridge) if bridge else None,
        candidate_resources=resources if binding_kind == "docker" else None,
        primary_adversary_bound=True,
    )
    if repeated.get("state") != "ready" and repeated.get("reviewable") is not True:
        return repeated
    # Judge tasks already pass repeated blind calibration before this helper.
    # Requiring identical stochastic rewards would change their admitted rubric.
    accepted = _read_json(item_root / "contract/accepted.json")
    judge = accepted["proposal"]["verification"] == "judge"
    composed = (item_root / "workspace/task/composite-verifier.json").is_file()
    if judge and not composed:
        return {
            **repeated,
            "fixed_grading": {
                "state": "not_applicable",
                "reason": "judge tasks use the preceding repeated blind calibration gate",
                "machine_component_repeatability": "unassessed",
            },
        }
    if judge:
        from .composite_grading_diagnostics import run_composite_grading_diagnostics

        grading = run_composite_grading_diagnostics(
            item_root, repeated, toolchain, validation_timeout, daytona_tools,
        )
        grading["scope"] = "composed_executable_checks_only"
        grading["judge_evidence"] = "preceding repeated blind calibration"
    else:
        from .grading_diagnostics import run_grading_diagnostics

        grading = run_grading_diagnostics(
            Path(repeated["input_manifest"]).parent,
            toolchain.source_package_root or toolchain.package_root,
            Path(repeated["attempt"]) / "fixed-grading",
            timeout=validation_timeout,
        )
    result = {
        **repeated,
        "fixed_grading": {
            key: value for key, value in grading.items() if key != "extra_files"
        },
        "extra_files": {
            **repeated.get("extra_files", {}),
            **grading.get("extra_files", {}),
        },
    }
    if not judge and grading.get("state") in {"ready", "semantic_failed"} and grading.get("reviewable") is True:
        result["unassessed_recipe_rows"] = [
            row for row in repeated.get("unassessed_recipe_rows", [])
            if row != "reward_determinism_10_regrades"
        ]
    if grading.get("state") == "unsupported" and grading.get("unassessed") is True:
        # An input shape the regrade subsystem cannot replay is UNASSESSED, not
        # failed.  A final-state task submits a workspace, so its controls carry
        # no fixed response string for a deterministic text replay; judge tasks
        # already take this path above as "not_applicable".  Downgrading here
        # blocked semantic review for a task whose repeated evaluation had
        # passed 3/3 on every gate.  Keep the repeated verdict, leave
        # reward_determinism_10_regrades in unassessed_recipe_rows, and let
        # fixed_grading carry the reason.
        return result
    if grading.get("state") != "ready":
        result["state"] = "pending"
        result["reviewable"] = (
            repeated.get("reviewable") is True
            and grading.get("reviewable") is True
            and grading.get("state") == "semantic_failed"
        )
        result["issues"] = [
            *repeated.get("issues", []),
            "fixed-input grading diagnostics are " + str(grading.get("state")),
        ]
    return result


def _attack_adjudication(
    item_root: Path, root: Path, agent: OMPAgent
) -> tuple[dict[str, Any], Path]:
    from .attack_adjudication import run_adjudication, validate_resolution
    from .inference import digest
    from .quality import sha256, source_files

    parent = root / "attack-adjudication" / item_root.name
    prior = sorted(parent.glob("attempt-*")) if parent.is_dir() else []
    for review_root in reversed(prior):
        try:
            return validate_resolution(item_root, review_root), review_root
        except (OSError, ValueError, TypeError, KeyError):
            # A rejected or uncertain adjudication is a durable result for its
            # immutable packet. Do not keep sampling reviewers until one clears
            # the reward alarm. Construction must change before another review.
            try:
                manifest = _read_json(review_root / "input-manifest.json")
                identity = {
                    key: value
                    for key, value in manifest.items()
                    if key != "snapshot_hash"
                }
                result = _read_json(review_root / "result.json")
                current = {
                    name: sha256(path) for name, path in source_files(item_root).items()
                }
                if (
                    manifest.get("schema_version")
                    == "capability-attack-adjudication-v1-input"
                    and manifest.get("snapshot_hash") == digest(identity)
                    and manifest.get("files") == current
                    and result.get("schema_version")
                    == "capability-attack-adjudication-v1-result"
                    and result.get("snapshot_hash") == manifest["snapshot_hash"]
                    and result.get("state") != "resolved"
                ):
                    return result, review_root
            except (OSError, ValueError, TypeError, KeyError):
                pass
    attempt = len(prior) + 1
    review_root = parent / f"attempt-{attempt}"
    result = run_adjudication(item_root, review_root, agent)
    if result.get("state") == "resolved":
        validate_resolution(item_root, review_root)
    return result, review_root


def _synthesize_attempt(
    item: dict[str, Any],
    root: Path,
    agent: OMPAgent,
    toolchain: OfficialToolchain | None,
    runtime_runner: Path | None,
    validation_timeout: int,
    daytona_tools: Path | None = None,
) -> dict[str, Any]:
    proposal = item["proposal"]
    key = f"{proposal['capability_id']}:{proposal['slot']}"
    item_root = root / "items" / f"{_safe_name(key)}-{item['proposal_hash'][:12]}"
    contract = item_root / "contract"
    contract.mkdir(parents=True, exist_ok=True)
    atomic_json(contract / "accepted.json", item)
    shutil.copy2(SOURCE_LOCK, contract / "source.lock.json")
    shutil.copy2(
        PROJECT_ROOT / "vendor" / "task_spec" / "composite_extension.lock.json",
        contract / "composite_extension.lock.json",
    )
    shutil.copy2(
        PROJECT_ROOT
        / "vendor"
        / "task_spec"
        / "patches"
        / "composite_required_extension.patch",
        contract / "composite_required_extension.patch",
    )
    shutil.copy2(
        PROJECT_ROOT / "docs" / "task_contract.md", contract / "task_contract.md"
    )
    shutil.copy2(
        PROJECT_ROOT / "docs" / "builder_measurements.md",
        contract / "builder_measurements.md",
    )
    shutil.copy2(
        PROJECT_ROOT / "docs" / "builder_images.md",
        contract / "builder_images.md",
    )
    from .quality import COMMON_CONDITIONS

    quality_conditions = dict(COMMON_CONDITIONS)
    if proposal["verification"] == "code":
        quality_conditions["quality-code-mutations"] = (
            "For code or composite grading, kill 100% of critical and at least 90% of all targeted "
            "grader mutants; retain surviving mutants for adjudication."
        )
    atomic_json(
        contract / "quality-conditions.json",
        {
            "schema_version": "capability-quality-conditions-v1",
            "conditions": quality_conditions,
        },
    )
    checklist = _build_checklist_path(item["proposal_hash"])
    if item.get("construction_context") is not None and checklist.is_file():
        shutil.copy2(checklist, contract / "build_acceptance.md")
        atomic_json(
            contract / "build_acceptance-source.json",
            {
                "proposal_hash": item["proposal_hash"],
                "source": str(checklist.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(checklist),
            },
        )
    # The judge policy is staged whenever the GLM base URL is configured, not
    # only for judge proposals: builders sometimes add a judge verifier to a
    # simple/code task, and the runtime then requires the frozen policy.
    configured_base = os.environ.get("GLM_BASE_URL")
    judge_policy = contract / "judge-policy.json"
    if not configured_base:
        # Never leave a policy from an earlier configuration in place.
        judge_policy.unlink(missing_ok=True)
        if proposal["verification"] == "judge":
            result = {
                "key": key,
                "proposal_hash": item["proposal_hash"],
                "sessions": [],
                "state": "pending_judge_policy",
                "issues": ["GLM judge base URL is unavailable"],
                "item_root": str(item_root),
            }
            write_status(item_root, result)
            return result
    else:
        atomic_json(
            judge_policy,
            {
                "provider": os.environ.get("CAPABILITY_JUDGE_PROVIDER", "glm"),
                "model": os.environ.get("CAPABILITY_JUDGE_MODEL", "glm-5.3"),
                "base_url": _normalized_glm_base(configured_base),
            },
        )
    if daytona_tools:
        # The workspace is agent-owned: stage without following a planted link.
        (item_root / "workspace").mkdir(parents=True, exist_ok=True)
        for name in ("dt.py", "dt.sh", "validate_env.py", "verify.py", "adapter.py"):
            builder_confinement.stage_file(
                item_root / "workspace", f"tools/daytona/{name}", daytona_tools / name
            )
    sessions = _run_sessions(
        item,
        item_root,
        agent,
        _agent_package_root(toolchain, item_root / "workspace"),
        contract,
        daytona_tools is not None,
    )
    result: dict[str, Any] = {
        "key": key,
        "proposal_hash": item["proposal_hash"],
        "sessions": sessions,
        "state": "pending_build",
        "issues": [],
        "item_root": str(item_root),
    }
    if any(session["status"] != "complete" for session in sessions):
        result["issues"].append("one or more declared build sessions need continuation")
        for session in sessions:
            reason = session.get("continuation_reason")
            if reason:
                result["issues"].append(
                    f"builder session {session['session']} stopped: {reason}"
                )
        write_status(item_root, result)
        return result
    bundle = item_root / "workspace" / "task"
    required = tuple(
        bundle / name
        for name in (
            "specification.json",
            "renderings.json",
            "binding.json",
            "controls.json",
        )
    )
    missing = [
        str(path.relative_to(item_root)) for path in required if not path.is_file()
    ]
    if missing:
        result["issues"].append("missing final bundle files: " + ", ".join(missing))
        write_status(item_root, result)
        return result
    from .runtime import judge_step_indices

    try:
        spec_uses_judge = bool(
            judge_step_indices(_read_json(bundle / "specification.json"))
        )
    except (OSError, ValueError):
        # Bundle validation below reports the unreadable specification.
        spec_uses_judge = False
    bundle_policy = bundle / "judge-policy.json"
    if proposal["verification"] == "judge" or spec_uses_judge:
        if not judge_policy.is_file():
            result.update(
                state="pending_judge_policy",
                issues=[
                    "specification uses a judge verifier but no judge policy "
                    "could be staged: GLM judge base URL is unavailable"
                ],
            )
            write_status(item_root, result)
            return result
        shutil.copy2(judge_policy, bundle_policy)
    elif (
        bundle_policy.is_file()
        and judge_policy.is_file()
        and bundle_policy.read_bytes() == judge_policy.read_bytes()
    ):
        # A controller copy from an earlier attempt whose judge verifier a
        # repair removed; left in place it would skew the runtime attestation.
        bundle_policy.unlink()
    try:
        controls = _read_json(bundle / "controls.json")
        _validate_environment_fidelity(item, _read_json(bundle / "binding.json"))
        raw_specification = _read_json(bundle / "specification.json")
        steps = (
            raw_specification.get("steps")
            if isinstance(raw_specification, dict)
            else None
        )
        if not isinstance(steps, list) or not steps:
            raise SynthesisError("specification has no steps")
        _validate_controls(controls, len(steps))
        if proposal["verification"] == "judge":
            from .judge import validate_task_calibration_fixture

            # Malformed builder-authored fixtures are repairable task inputs.
            # Validate them before model calls so transport/measurement failures
            # can retain their separate pending-calibration disposition.
            validate_task_calibration_fixture(
                _read_json(bundle / "judge-calibration.json"),
                _sha256(bundle / "specification.json"),
                3,
                0.15,
                composite=(bundle / "composite-verifier.json").is_file(),
            )
        from .image_pipeline import process_image_construction

        image_arguments: dict[str, Any] = {
            "item_root": item_root,
            "capture_tools": (daytona_tools or PROJECT_ROOT / "daytona-tools") / "capture-tools",
            "scripts_root": PROJECT_ROOT / "scripts",
            # Marks "image_review" in the conveyor when the GLM review starts.
            "agent": ActivityAgent(agent, item_root, "image_review"),
            "builder_session_ids": {str(session["session"]) for session in sessions}
            | {path.stem for path in (item_root / "sessions").glob("*/transcript/*.jsonl")},
            "review_base": root / "image-reviews" / item_root.name,
        }
        # Marks capture / cold-pull commands, only if the controller still
        # accepts a command_runner with a callable default.
        command_runner = activity_command_runner(process_image_construction, item_root)
        if command_runner is not None:
            image_arguments["command_runner"] = command_runner
        mark_activity(item_root, "image_construction")
        try:
            images = process_image_construction(**image_arguments)
        except OSError as error:
            # Local disk/filesystem trouble inside the image controller (ledger,
            # receipts, review directories) says nothing about the builder's
            # task: a transient infrastructure wait, never "invalid task bundle".
            from .image_pipeline import controller_io_hold

            images = controller_io_hold(error)
        result["custom_images"] = images
        # A publisher rejection the builder can fix (rootfs review) is repair
        # input exactly like a repairable image step: it needs a new capture
        # request from the builder, hence a new attempt and packet.
        builder_repairable = (
            images["state"] == "failed_terminal" and images.get("builder_repairable") is True
        )
        if images["state"] != "ready":
            if images["state"] == "repairable" or builder_repairable:
                result.update(
                    state="failed",
                    issues=[
                        "invalid task bundle: " + str(issue)
                        for issue in images.get("issues") or [images["reason"]]
                    ],
                )
            elif images["state"] == "failed_terminal":
                # A step that can never succeed for these bytes: terminal,
                # never repair input (the issue prefix is not a repair prefix).
                stage = image_failure_stage(images.get("failure_stage"))
                reason = str(
                    images.get("reason") or images.get("failure_stage") or "failed_terminal"
                )
                extra = images.get("issues") if isinstance(images.get("issues"), list) else []
                result.update(
                    state="failed",
                    failure_stage=stage,
                    issues=[
                        f"{stage} failed terminally: {reason}",
                        *(str(issue) for issue in extra),
                    ],
                )
            else:
                result.update(
                    state="pending_image_" + images["state"].removeprefix("pending_"),
                    issues=[images.get("reason", images["state"])],
                )
            write_status(item_root, result)
            return result
        composite_path = bundle / "composite-verifier.json"
        if composite_path.is_file():
            if proposal["verification"] != "judge":
                raise SynthesisError(
                    "composite verifier requires proposal verification=judge"
                )
            from .composite_policy import validate_composite_config

            validate_composite_config(
                _read_json(composite_path),
                specification_sha256=_sha256(bundle / "specification.json"),
                adapter_sha256=_sha256(
                    Path(__file__).with_name("composite_verifier.py")
                ),
                policy_sha256=_sha256(Path(__file__).with_name("composite_policy.py")),
                step_count=len(steps),
            )
    except (OSError, json.JSONDecodeError, SynthesisError, ValueError) as error:
        result.update(state="failed", issues=[f"invalid task bundle: {error}"])
        write_status(item_root, result)
        return result
    if item.get("construction_context") is not None:
        acceptance_issues = _build_acceptance_issues(
            item, bundle, contract / "build_acceptance.md"
        )
        if acceptance_issues:
            result.update(
                state="pending_build_acceptance",
                issues=acceptance_issues,
            )
            write_status(item_root, result)
            return result
    if toolchain is None:
        result.update(
            state="pending_schema_validation",
            issues=["pinned TaskCompendium toolchain is unavailable"],
        )
        write_status(item_root, result)
        return result
    harbor = item_root / "harbor"
    heal = getattr(toolchain, "heal", None)
    if callable(heal):
        # Restore a drifted controller overlay before lowering, calibration and
        # runtime use it; a failure here surfaces through those gates.
        try:
            healed = heal()
        except (SynthesisError, OSError, json.JSONDecodeError) as error:
            result["toolchain_heal_error"] = str(error)[:REASON_EVIDENCE]
        else:
            if healed is not None:
                result["toolchain_heal"] = healed
    mark_activity(item_root, "taskcompendium_lowering")
    try:
        result["taskcompendium"] = toolchain.validate_and_lower(
            bundle, harbor, validation_timeout
        )
        from .composite_extension import (
            PATCHED_LOWERING_SHA256,
            PATCHED_RUNNER_SHA256,
            PATCHED_VERIFIER_SHA256,
        )

        result["taskcompendium"]["composite_extension_overlay_sha256"] = (
            PATCHED_VERIFIER_SHA256
        )
        result["taskcompendium"]["composite_runner_overlay_sha256"] = (
            PATCHED_RUNNER_SHA256
        )
        result["taskcompendium"]["composite_lowering_overlay_sha256"] = (
            PATCHED_LOWERING_SHA256
        )
        if (bundle / "composite-verifier.json").is_file():
            from .composite_extension import (
                install_extension_marker,
                validate_extension_marker,
            )

            pins = {
                "adapter_sha256": _sha256(Path(__file__).with_name("composite_verifier.py")),
                "policy_sha256": _sha256(Path(__file__).with_name("composite_policy.py")),
                "config_sha256": _sha256(bundle / "composite-verifier.json"),
            }
            if (harbor / "composite-specification.json").is_file():
                validate_extension_marker(harbor, **pins, supported=True)
            else:
                shutil.copy2(
                    bundle / "composite-verifier.json",
                    harbor / "composite-verifier.json",
                )
                install_extension_marker(harbor, **pins)
        result["state"] = "lowered"
    except (
        SynthesisError,
        OSError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        result.update(
            state=_lowering_failure_state(bundle, error),
            issues=[f"TaskCompendium validation/lowering failed: {error}"],
        )
        write_status(item_root, result)
        return result
    if proposal["verification"] == "judge":
        calibration_root = item_root / "judge-calibration"
        calibration_path = calibration_root / "judge-calibration.json"
        try:
            from types import SimpleNamespace

            from .judge import calibrate_task

            mark_activity(item_root, "judge_calibration")
            calibration_exit = calibrate_task(
                SimpleNamespace(
                    out=str(calibration_root),
                    bundle=str(bundle),
                    taskcompendium_source=str(
                        getattr(toolchain, "source_package_root", None)
                        or toolchain.package_root
                    ),
                    toolchain=(
                        toolchain if isinstance(toolchain, OfficialToolchain) else None
                    ),
                    api_key_env="GLM_API_TOKEN",
                    concurrency=int(
                        os.environ.get("CAPABILITY_JUDGE_CALIBRATION_CONCURRENCY", "64")
                    ),
                    repeats=3,
                    max_spread=0.15,
                    timeout=validation_timeout,
                )
            )
            calibration = _read_json(calibration_path)
            if (
                calibration_exit != 0
                or calibration.get("state") != "passed"
                or calibration.get("specification_sha256")
                != _sha256(bundle / "specification.json")
                or not isinstance(calibration.get("fixture_hash"), str)
            ):
                raise SynthesisError("native judge calibration did not pass")
            result["judge_calibration"] = {
                "artifact": str(calibration_path),
                "artifact_sha256": _sha256(calibration_path),
                "fixture_hash": calibration["fixture_hash"],
            }
        except (
            SynthesisError,
            OSError,
            json.JSONDecodeError,
            RuntimeError,
            ValueError,
        ) as error:
            compatibility = _runtime_image_compatibility_failures(item_root, bundle)
            if compatibility:
                _record_image_compatibility_failure(result, compatibility)
                write_status(item_root, result)
                return result
            raw_path = calibration_root / "taskcompendium-judge-results.json"
            fixture_path = bundle / "judge-calibration.json"
            if calibration_path.is_file() and raw_path.is_file() and fixture_path.is_file():
                result["judge_calibration_failure"] = {
                    "artifact": str(calibration_path),
                    "artifact_sha256": _sha256(calibration_path),
                    "raw_artifact": str(raw_path),
                    "raw_artifact_sha256": _sha256(raw_path),
                    "fixture_artifact": str(fixture_path),
                    "fixture_artifact_sha256": _sha256(fixture_path),
                }
            result.update(
                state="pending_judge_calibration",
                issues=[f"native judge calibration is incomplete: {error}"],
            )
            write_status(item_root, result)
            return result
    evidence_path = item_root / "runtime-evidence.json"
    mark_activity(item_root, "runtime_controls")
    try:
        runner_command = (
            [str(runtime_runner)]
            if runtime_runner
            else toolchain.runtime_command()
            if hasattr(toolchain, "runtime_command")
            else []
        )
        if runner_command:
            evidence = _external_controls(
                runner_command,
                bundle,
                harbor,
                bundle / "controls.json",
                evidence_path,
                validation_timeout,
            )
            passed, issues = _controls_pass(controls, evidence, external=True)
            attestation_issues = _attestation_issues(
                evidence,
                evidence_path,
                bundle,
                harbor,
                bundle / "controls.json",
                result["taskcompendium"]["specification_sha256"],
            )
            issues.extend(attestation_issues)
            passed = passed and not issues
        elif (
            proposal["environment"] == "reasoning"
            and proposal["verification"] != "judge"
        ):
            evidence = toolchain.direct_controls(
                bundle, bundle / "controls.json", validation_timeout
            )
            atomic_json(evidence_path, evidence)
            passed, issues = _controls_pass(controls, evidence)
        else:
            result.update(
                state="pending_runtime",
                issues=[
                    f"{proposal['environment']}/{proposal['verification']} requires an external Harbor-conformant runtime runner"
                ],
            )
            write_status(item_root, result)
            return result
        result["runtime_evidence"] = str(evidence_path)
        if not passed:
            if evidence.get("attestation", {}).get("solver", {}).get("state") == (
                "needs_adjudication"
            ):
                result.update(
                    state="pending_solver_adjudication",
                    issues=issues,
                )
                write_status(item_root, result)
                return result
            incomplete_adversary = _incomplete_adversary_from_evidence(
                evidence, evidence_path, harbor
            )
            if incomplete_adversary is not None:
                result.update(
                    state="pending_adversary_retry",
                    issues=issues,
                    incomplete_adversary=incomplete_adversary,
                )
                write_status(item_root, result)
                return result
            adjudicable_attack_issues = {
                "attestation lacks an independent adversarial attack",
                "independent adversary report needs adjudication or retry",
            }
            if issues and set(issues) <= adjudicable_attack_issues:
                try:
                    mark_activity(item_root, "attack_adjudication")
                    adjudication, adjudication_root = _attack_adjudication(
                        item_root, root, agent
                    )
                    adjudication_result = adjudication_root / "result.json"
                    result["attack_adjudication"] = {
                        "state": adjudication.get("state"),
                        "artifact": str(adjudication_result),
                        "artifact_sha256": _sha256(adjudication_result),
                        "snapshot_hash": adjudication.get("snapshot_hash"),
                    }
                    if adjudication.get("state") == "resolved":
                        issues = [
                            issue
                            for issue in issues
                            if issue not in adjudicable_attack_issues
                        ]
                        passed = not issues
                    else:
                        result.update(
                            state="pending_attack_adjudication",
                            issues=adjudication.get("issues")
                            or ["rewarded independent attacks need adjudication"],
                        )
                        write_status(item_root, result)
                        return result
                except (
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    KeyError,
                ) as error:
                    result.update(
                        state="pending_attack_adjudication",
                        issues=[
                            f"independent attack adjudication is incomplete: {error}"
                        ],
                    )
                    write_status(item_root, result)
                    return result
            if not passed:
                pending_adversary = {
                    "attestation lacks an independent adversarial attack",
                    "independent adversary lacks a raw artifact reference and digest",
                    "independent adversary report needs adjudication or retry",
                }
                if issues and set(issues) <= pending_adversary:
                    result.update(
                        state="runtime_controls_passed_pending_adversary",
                        issues=issues,
                    )
                    write_status(item_root, result)
                    return result
                result.update(state="failed", issues=issues)
                write_status(item_root, result)
                return result
        if not runner_command:
            result.update(
                state="controls_passed_pending_rollout",
                issues=[
                    "direct controls passed, but the Harbor runtime runner is unavailable"
                ],
            )
            write_status(item_root, result)
            return result
    except (
        SynthesisError,
        OSError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        result.update(state="failed", issues=[f"runtime controls failed: {error}"])
        compatibility = _runtime_image_compatibility_failures(item_root, bundle)
        if compatibility:
            _record_image_compatibility_failure(result, compatibility)
        write_status(item_root, result)
        return result
    result["runtime_validated"] = True
    result["state"] = "validated"
    try:
        mark_activity(item_root, "repeated_diagnostics")
        diagnostics = _repeated_quality_diagnostics(
            item_root, toolchain, validation_timeout, daytona_tools
        )
        result["repeated_diagnostics"] = {
            key: value for key, value in diagnostics.items() if key != "extra_files"
        }
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        result.update(
            state="pending_repeated_diagnostics",
            issues=[f"repeated runtime diagnostics are incomplete: {type(error).__name__}"],
        )
        write_status(item_root, result)
        return result
    if diagnostics.get("state") != "ready" and diagnostics.get("reviewable") is not True:
        result.update(
            state="pending_repeated_diagnostics",
            issues=["repeated runtime diagnostics lack complete validated evidence"],
        )
        write_status(item_root, result)
        return result
    quality_parent = root / "quality" / item_root.name
    attempt = 1
    while (quality_parent / f"attempt-{attempt}").exists():
        attempt += 1
    quality_root = quality_parent / f"attempt-{attempt}"
    try:
        from .quality import run_review

        # Item-owned diagnostics are already part of the protected quality/repair
        # snapshot. Only add external evidence here, avoiding duplicate payloads.
        quality_extra = {
            name: path for name, path in diagnostics.get("extra_files", {}).items()
            if not Path(path).resolve().is_relative_to(item_root / "diagnostics")
        }
        adjudication_record = result.get("attack_adjudication")
        if isinstance(adjudication_record, dict):
            adjudication_root = Path(adjudication_record["artifact"]).parent
            quality_extra.update({
                f"controller/attack-adjudication/{name}": adjudication_root / name
                for name in ("input-manifest.json", "receipt.json", "result.json")
            })
        from .acceptance import GATE_RECEIPT

        # The post-review gate state beside the review, whatever the verdict:
        # an accepting review is an acceptance claim only when this says "ready"
        # (acceptance.post_review_gate_passed).  It is fixed before the review
        # starts, so it is written before the review too: a new-code review
        # whose post-review write failed is never judged by the legacy
        # reconstruction (which only applies to reviews without a receipt).
        gate_state = {
            "repeated_diagnostics_state": diagnostics.get("state"),
            "reviewable": diagnostics.get("reviewable"),
        }
        atomic_json(
            quality_root / GATE_RECEIPT,
            {**gate_state, "phase": "pre_review", "at": time.time()},
        )
        mark_activity(item_root, "quality_review", attempt=attempt)
        quality = (
            run_review(item_root, quality_root, agent, extra_files=quality_extra)
            if quality_extra
            else run_review(item_root, quality_root, agent)
        )
        atomic_json(
            quality_root / GATE_RECEIPT,
            {**gate_state, "phase": "post_review", "at": time.time()},
        )
        quality_result = quality_root / "result.json"
        result["quality_review"] = {
            "state": quality.get("state"),
            "artifact": str(quality_result),
            "artifact_sha256": _sha256(quality_result),
            "snapshot_hash": quality.get("snapshot_hash"),
        }
        if quality.get("state") != "accept":
            issues = quality.get("issues")
            if not isinstance(issues, list) or not issues:
                issues = [
                    "independent semantic review did not accept the runtime-validated task: "
                    + str(quality.get("state"))
                ]
            result.update(state="pending_quality_review", issues=issues)
            write_status(item_root, result)
            return result
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        result.update(
            state="pending_quality_review",
            issues=[f"independent semantic review is incomplete: {error}"],
        )
        write_status(item_root, result)
        return result
    if diagnostics.get("state") != "ready":
        result.update(
            state="pending_repeated_diagnostics",
            issues=["semantic review accepted despite failed repeated runtime gates"],
        )
        write_status(item_root, result)
        return result
    result["state"] = "quality_accepted"
    export = root / "validated" / item_root.name
    if export.exists():
        shutil.rmtree(export)
    shutil.copytree(harbor, export)
    result["export"] = str(export)
    from .acceptance import record_acceptance

    # Bind the acceptance to the reviewed bytes so a resume keeps it.
    record_acceptance(item, key, root, item_root, result)
    write_status(item_root, result)
    return result


def _repair_feedback(
    item: dict[str, Any], result: dict[str, Any], round_: int, item_root: Path
) -> dict:
    capability_prefix = item["proposal"]["capability_id"].split(".", 1)[0]
    audits = []
    audit_root = PROJECT_ROOT / "docs" / "audits"
    for path in sorted(audit_root.glob("*.md")):
        if path.is_file():
            text = path.read_text()
            if not (
                path.name.startswith(capability_prefix) or item["proposal_hash"] in text
            ):
                continue
            audits.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": _sha256(path),
                    "text": text,
                }
            )
    from .quality import COMMON_CONDITIONS, has_executable_verifier

    quality_conditions = dict(COMMON_CONDITIONS)
    bundle = item_root / "workspace" / "task"
    specification = {}
    try:
        specification = _read_json(bundle / "specification.json")
    except (OSError, json.JSONDecodeError):
        pass
    if (
        item["proposal"].get("verification") == "code"
        or has_executable_verifier(specification)
        or (bundle / "composite-verifier.json").is_file()
    ):
        quality_conditions["quality-code-mutations"] = (
            "Kill 100% of critical and at least 90% of all targeted code-grader mutants; "
            "justify mutant labels independently and retain surviving mutants."
        )
    checklist = _build_checklist_path(item["proposal_hash"])
    task_contract = PROJECT_ROOT / "docs" / "task_contract.md"
    return {
        "schema_version": "capability-construction-repair-feedback-v1",
        "round": round_,
        "failed_state": result.get("state"),
        "issues": result.get("issues", []),
        "custom_images": result.get("custom_images"),
        "verifier_image_compatibility_failures": result.get("verifier_image_compatibility_failures"),
        "runtime_evidence": result.get("runtime_evidence"),
        "repeated_diagnostics": result.get("repeated_diagnostics"),
        "attack_adjudication": result.get("attack_adjudication"),
        "judge_calibration_failure": (
            {
                **result["judge_calibration_failure"],
                "metrics": _read_json(
                    Path(result["judge_calibration_failure"]["artifact"])
                )["metrics"],
                "issues": _read_json(
                    Path(result["judge_calibration_failure"]["artifact"])
                )["issues"],
            }
            if result.get("state") == "pending_judge_calibration"
            and _measured_judge_calibration_failure(
                result, expected_item_root=item_root,
                expected_key=f"{item['proposal']['capability_id']}:{item['proposal']['slot']}",
                expected_proposal_hash=item["proposal_hash"],
            )
            else None
        ),
        "attack_adjudication_receipt": (
            {
                "artifact": str(Path(result["attack_adjudication"]["artifact"]).parent / "receipt.json"),
                "artifact_sha256": _sha256(
                    Path(result["attack_adjudication"]["artifact"]).parent / "receipt.json"
                ),
            }
            if result.get("state") == "pending_attack_adjudication"
            and _proven_exploit_for_repair(result)
            else None
        ),
        "quality_review": result.get("quality_review"),
        "measured_audits": audits,
        "required_quality_conditions": quality_conditions,
        "current_construction_checklist": {
            "path": str(checklist.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(checklist),
            "text": checklist.read_text(),
        }
        if checklist.is_file()
        else None,
        "current_task_contract": {
            "path": str(task_contract.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(task_contract),
            "text": task_contract.read_text(),
        },
    }


def _archive_attempt(item_root: Path, root: Path, round_: int) -> Path:
    destination = root / "repair-history" / item_root.name / f"attempt-{round_}"
    destination.mkdir(parents=True, exist_ok=False)
    status = item_root / "status.json"
    if status.is_file():
        shutil.copy2(status, destination / "status.json")
    for name in (
        "harbor",
        "runtime-trials",
        "judge-calibration",
        "diagnostics",
        "runtime-evidence.json",
        "solver-transcripts.json",
        "independent-adversary.json",
        "authored-oracle.json",
    ):
        source = item_root / name
        if source.exists():
            shutil.move(str(source), destination / name)
    return destination


def _repair_activity(
    item_root: Path,
    repair_root: Path,
    attempt: int,
    prior_result: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    """Publish repair liveness without changing the last gate result."""
    status = item_root / "status.json"
    started = datetime.now(UTC).isoformat()
    activity = {
        "schema_version": "capability-controller-operation-v1",
        "operation": "construction_repair",
        "state": "active",
        "attempt": attempt,
        "source_prior_state": prior_result.get("state"),
        "source_status": str(status),
        "source_status_sha256": _sha256(status) if status.is_file() else None,
        "repair_directory": str(repair_root),
        "transcript_directory": str(repair_root / "transcript"),
        "started_at": started,
        "completed_at": None,
        "outcome": None,
    }
    current = item_root / "controller" / "active-operation.json"
    history = item_root / "controller" / "operations" / f"repair-attempt-{attempt}.json"
    atomic_json(current, activity)
    atomic_json(history, activity)
    return activity, current, history


def _close_repair_activity(
    activity: dict[str, Any],
    current: Path,
    history: Path,
    state: str,
    outcome: dict[str, Any],
) -> None:
    closed = {
        **activity,
        "state": state,
        "completed_at": datetime.now(UTC).isoformat(),
        "outcome": outcome,
    }
    atomic_json(current, closed)
    atomic_json(history, closed)


def _record_image_compatibility_failure(result: dict, failures: list[dict]) -> None:
    result.update(
        state="failed",
        issues=[
            (
            "invalid task bundle: authored verifier image uses Python 3.11, but the "
            "pinned TaskCompendium supervisor requires Python >=3.12. Rebuild the "
            "verifier image with a compatible supervisor and update its exact image "
            "reference/supervisor_python; preserve the task's intended behavior."
            )
        ],
        verifier_image_compatibility_failures=failures,
    )


def _runtime_image_compatibility_failures(item_root: Path, bundle: Path) -> list[dict]:
    """Accept only trusted, recipe-bound image incompatibility receipts as repair input."""
    from .daytona_policy import verifier_snapshot_recipe

    try:
        runtimes = set(_container_supervisor_runtimes(bundle).values())
        adapter_sha = _sha256(PROJECT_ROOT / "capability_pipeline/daytona_verifier.py")
        composite_path = bundle / "composite-verifier.json"
        if composite_path.is_file():
            from .composite_policy import validate_composite_config

            policies = validate_composite_config(
                _read_json(composite_path),
                specification_sha256=_sha256(bundle / "specification.json"),
                adapter_sha256=_sha256(PROJECT_ROOT / "capability_pipeline/composite_verifier.py"),
                policy_sha256=_sha256(PROJECT_ROOT / "capability_pipeline/composite_policy.py"),
                step_count=len(_read_json(bundle / "specification.json")["steps"]),
            )
            runtimes.update(
                (check["image"], check.get("supervisor_python", "python3"))
                for policy in policies.values() for check in policy["machine_checks"]
            )
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return []

    def records(value):
        if not isinstance(value, dict):
            return
        detail = value.get("detail")
        if value.get("status") != "infra_error" or value.get("reward") is not None or not isinstance(detail, dict):
            return
        receipt = detail.get("verifier_image_compatibility")
        if isinstance(receipt, dict):
            yield receipt
            return
        for machine in detail.get("machine_results", []) if isinstance(detail.get("machine_results", []), list) else []:
            yield from records(machine)

    failures = []
    for parent in (item_root / "runtime-trials", item_root / "judge-calibration"):
        for path in sorted(parent.rglob("verifier/taskcompendium-result.json")):
            if any(part.is_symlink() for part in (path, *path.parents)):
                continue
            try:
                for receipt in records(_read_json(path)):
                    pair = (receipt.get("image"), receipt.get("supervisor_python"))
                    if not all(isinstance(value, str) for value in pair) or pair not in runtimes:
                        continue
                    if (receipt.get("schema_version") != "capability-verifier-image-compatibility-v1"
                            or receipt.get("reason") != "supervisor_python_too_old"
                            or receipt.get("adapter_sha256") != adapter_sha
                            or receipt.get("pinned_requirement") != "numpy==2.5.3"
                            or receipt.get("required_python") != ">=3.12"
                            or receipt.get("observed_python") != "3.11"
                            or receipt.get("snapshot_recipe_sha256") != hashlib.sha256(verifier_snapshot_recipe(*pair).encode()).hexdigest()
                            or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("build_log_sha256", "")))
                            or not all(isinstance(receipt.get(key), str) and receipt[key] for key in ("snapshot_name", "snapshot_id", "provider_log_excerpt"))):
                        continue
                    failures.append({"artifact": str(path), "artifact_sha256": _sha256(path), "receipt": receipt})
            except (OSError, ValueError, TypeError, KeyError):
                continue
    return failures


# Ungraded trial outcomes that say nothing about the task.  Each maps a
# provider_cause to the family that decides how the retry is gated:
#   sandbox  -- sandbox provider/broker; gated on a fresh create/delete probe
#   glm      -- GLM judge/solver transport or malformed judge output; backoff only
#   staging  -- the job's own staged controller code vanished; gated on its presence
# Measured on catalog-full-construct-003 (2026-09-29): 248 held runtime gates
# were VerifierTimeoutError, 100% on sandboxed verifiers, while the successful
# sandboxed verifier phases on one shard had p50 163 s / p90 505 s / max 597 s
# against the 600 s budget; native verifiers peaked at 77 s.
INFRASTRUCTURE_CAUSE_FAMILY = {
    "ProviderRateLimitExhausted": "sandbox",
    "DaytonaRateLimitError": "sandbox",
    "DaytonaNotFoundError": "sandbox",
    "VerifierTimeoutError": "sandbox",
    "SiloNotFoundError": "sandbox",
    "SiloSandboxLost": "sandbox",
    "SiloTransportError": "sandbox",
    "EnvironmentStartTimeoutError": "sandbox",
    "PrivateVerifierPreparation": "sandbox",
    "GLMJudgeMalformedVerdict": "glm",
    "GLMJudgeTransport": "glm",
    "GLMTransportUnavailable": "glm",
    "StagingCodeMissing": "staging",
    "ToolchainIntegrity": "toolchain",
    "StaleJudgePolicyPath": "controller",
    # evaluation.inspect_attempt compared the attested solver state with the
    # authored expectations instead of the runtime's own rule (fixed there).
    "SolverStateCheck": "controller",
    # runtime.py aborted the whole gate when a graded independent-solver
    # attempt had hit its turn cap (fixed there: the grade is the outcome).
    "SolverTurnCapGateAbort": "controller",
}
_SANDBOXED_VERIFIERS = (
    "daytona_verifier:DaytonaSemanticVerifier",
    "daytona_verifier:CapturingDaytonaSemanticVerifier",
    "composite_verifier:CompositeSemanticVerifier",
)
_GRADING_INFRASTRUCTURE_ERRORS = (
    (re.compile(r"^(?:SiloNotFoundError|DaytonaNotFoundError): (?:sandbox|session) '[^']*' not found"), "SiloNotFoundError"),
    (re.compile(r"^SiloError: sandbox '[^']*' is destroying"), "SiloSandboxLost"),
    (re.compile(r"^TransportError: cannot reach https?://[^ ]+"), "SiloTransportError"),
    (re.compile(r"^(?:ProviderRateLimitExhausted|DaytonaRateLimitError)\b"), "ProviderRateLimitExhausted"),
    (
        re.compile(
            r"^RuntimeError: private verifier (?:preparation|upload) failed: .*"
            r"(?:SiloError|SiloNotFoundError|TransportError|timed out|timeout)"
        ),
        "PrivateVerifierPreparation",
    ),
    (re.compile(r"^Malformed judge verdict$"), "GLMJudgeMalformedVerdict"),
    (re.compile(r"^Judge request failed: "), "GLMJudgeTransport"),
    (re.compile(r"No such file or directory: '[^']*/capability-pipeline-staging/submissions/"), "StagingCodeMissing"),
)
_STAGING_CODE_MISSING = re.compile(
    r"No such file or directory: '[^']*/capability-pipeline-staging/submissions/[^']+'"
)
# 2026-09-28 controller builds did not stage the frozen judge policy for
# spec-level judge verifiers on non-judge proposals; the current controller does.
_STALE_JUDGE_POLICY = re.compile(
    r"FileNotFoundError: \[Errno 2\] No such file or directory: "
    r"'[^']*/items/[^/']+/workspace/task/judge-policy\.json'"
)
_TOOLCHAIN_INTEGRITY = re.compile(
    r"^native judge calibration is incomplete: TaskCompendium (?:source hash mismatch: |"
    r"archive file set mismatch|composite extension (?:file lock|patch hash|overlay hash) mismatch|"
    r"composite runtime overlay hash mismatch: |composite overlay base hash differs)"
)
_NAMED_TRIAL = re.compile(r"inspect (runtime-trials/[^\s;'\"]+?)/result\.json")


def _trial_infrastructure_cause(trial: dict[str, Any], name: str | None = None) -> str | None:
    """provider_cause for one retained, ungraded Harbor trial, or None.

    ``name`` is the trial directory name; ``control-*-attempt-N`` trials are
    independent-solver attempts.
    """
    exception = trial.get("exception_info")
    if not isinstance(exception, dict):
        return None
    exception_type = exception.get("exception_type")
    message = exception.get("exception_message")
    if not isinstance(exception_type, str) or not isinstance(message, str):
        return None
    if exception_type in {"SiloNotFoundError", "EnvironmentStartTimeoutError", "TransportError"}:
        # The candidate or verifier sandbox itself was lost or unreachable.  The
        # runtime aborts the gate on the exception even when a grade was written
        # first, so a retained verifier result does not make this a task outcome.
        return "SiloTransportError" if exception_type == "TransportError" else exception_type
    if trial.get("verifier_result") is not None:
        if (
            exception_type == "TurnCapExhaustedError"
            and isinstance(name, str)
            and re.fullmatch(r"control-.+-attempt-\d+", name)
        ):
            return "SolverTurnCapGateAbort"
        return None
    for cause in ("ProviderRateLimitExhausted", "DaytonaRateLimitError", "DaytonaNotFoundError"):
        if cause in message or cause == exception_type:
            return cause
    config = trial.get("config") if isinstance(trial.get("config"), dict) else {}
    verifier = str((config.get("verifier") or {}).get("import_path") or "")
    if exception_type == "VerifierTimeoutError" and verifier.endswith(_SANDBOXED_VERIFIERS):
        # The grader's own runtime timeout bounds the grader inside the sandbox;
        # a Harbor phase timeout here is sandbox provisioning/transport time.
        return "VerifierTimeoutError"
    if exception_type == "GradingInfrastructureError":
        try:
            detail = json.loads(message)
        except ValueError:
            detail = None
        error = detail.get("error") if isinstance(detail, dict) else None
        if isinstance(error, str):
            for pattern, cause in _GRADING_INFRASTRUCTURE_ERRORS:
                if pattern.search(error):
                    return cause
        return None
    if exception_type == "RuntimeError" and message.startswith(
        "GLM transport unavailable for the whole infrastructure hold"
    ):
        return "GLMTransportUnavailable"
    if _STAGING_CODE_MISSING.search(message):
        return "StagingCodeMissing"
    return None


def _runtime_infrastructure_failures(
    item_root: Path, result: dict[str, Any]
) -> list[dict[str, str]]:
    """Identify retained, ungraded provider failures eligible for revalidation.

    When the runtime issue names the trial that stopped the gate, only that
    trial counts: older trial directories from earlier attempts can remain.
    """
    if result.get("state") != "failed" or result.get("runtime_validated") is True:
        return []
    issues = result.get("issues")
    runtime_issues = [
        issue
        for issue in issues if isinstance(issue, str) and issue.startswith("runtime controls failed:")
    ] if isinstance(issues, list) else []
    if not runtime_issues:
        return []
    named = {match.group(1) for issue in runtime_issues for match in _NAMED_TRIAL.finditer(issue)}
    if named:
        paths = sorted(
            item_root / relative / "result.json"
            for relative in named
            if ".." not in Path(relative).parts
        )
    else:
        paths = sorted((item_root / "runtime-trials").glob("*/result.json"))
    failures = []
    for path in paths:
        if not path.is_file() or any(part.is_symlink() for part in (path, *path.parents) if item_root in part.parents):
            continue
        try:
            trial = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(trial, dict):
            continue
        provider_cause = _trial_infrastructure_cause(trial, path.parent.name)
        if provider_cause is not None:
            failures.append(
                {
                    "trial": path.parent.name,
                    "artifact": str(path),
                    "artifact_sha256": _sha256(path),
                    "exception_type": trial["exception_info"]["exception_type"],
                    "provider_cause": provider_cause,
                }
            )
    return failures


def _diagnostics_infrastructure_failures(
    item_root: Path, result: dict[str, Any]
) -> list[dict[str, str]]:
    """Repeated-diagnostics cells that failed on infrastructure, not on the task.

    Positive evidence only: every non-valid cell must be a runtime_error whose
    retained trials include at least one infrastructure-classified exception,
    or the known controller checker defect (solver-state comparison); any other
    evidence disagreement keeps the item where it is.
    """
    diagnostics = result.get("repeated_diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("state") == "ready":
        return []
    attempt_value = diagnostics.get("attempt")
    if not isinstance(attempt_value, str) or "/diagnostics/" not in attempt_value:
        return []
    # Status paths are absolute in the writing job's work root; a relaunch
    # under another run name has a different one.  Resolve inside this item.
    attempt = item_root / "diagnostics" / attempt_value.rsplit("/diagnostics/", 1)[1]
    if ".." in attempt.relative_to(item_root).parts or not attempt.is_dir():
        return []
    try:
        matrix = _read_json(attempt / "evaluation" / "matrix.json")
    except (OSError, json.JSONDecodeError):
        return []
    cells = matrix.get("cells") if isinstance(matrix, dict) else None
    if not isinstance(cells, list) or not cells:
        return []
    failures: list[dict[str, str]] = []
    for cell in cells:
        if not isinstance(cell, dict) or type(cell.get("attempt")) is not int:
            return []
        if cell.get("state") == "valid":
            continue
        if cell.get("state") == "invalid_evidence" and cell.get("issues") == [
            "solver state disagrees with recorded grades"
        ]:
            failures.append(
                {
                    "trial": f"diagnostics-{cell['attempt']:03d}",
                    "evidence": "solver state disagrees with recorded grades",
                    "provider_cause": "SolverStateCheck",
                }
            )
            continue
        if cell.get("state") != "runtime_error":
            return []
        trials = attempt / "evaluation" / "attempts" / f"{cell['attempt']:03d}" / "runtime-trials"
        found = None
        for path in sorted(trials.glob("*/result.json")):
            try:
                trial = _read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            cause = _trial_infrastructure_cause(trial, path.parent.name) if isinstance(trial, dict) else None
            if cause is not None:
                found = {
                    "trial": f"diagnostics-{cell['attempt']:03d}/{path.parent.name}",
                    "artifact": str(path),
                    "artifact_sha256": _sha256(path),
                    "exception_type": trial["exception_info"]["exception_type"],
                    "provider_cause": cause,
                }
                break
        if found is None:
            return []
        failures.append(found)
    return failures


def _infrastructure_hold(item_root: Path, result: dict[str, Any]) -> dict[str, Any] | None:
    """Classify an ungraded runtime-gate failure as infrastructure, with evidence.

    Returns ``{"gate", "families", "causes", "failures"}`` or None.  A hold
    never becomes GLM repair input; synthesize_one re-runs the gate instead.
    """
    state = result.get("state")
    recorded = result.get("runtime_infrastructure")
    exhausted_wait = result.get("wait_exhausted")
    if isinstance(recorded, dict) and isinstance(recorded.get("hold"), dict) and (
        state == "pending_runtime_infrastructure"
        or (
            state == "failed"
            and isinstance(exhausted_wait, dict)
            and exhausted_wait.get("kind") == "runtime_infrastructure"
        )
    ):
        return recorded["hold"]
    failures: list[dict[str, Any]] = []
    gate = None
    if state == "failed" and result.get("runtime_validated") is not True:
        failures = _runtime_infrastructure_failures(item_root, result)
        gate = "runtime_controls"
        if not failures:
            head = next(
                (
                    issue for issue in result.get("issues") or []
                    if isinstance(issue, str) and issue.startswith("runtime controls failed:")
                ),
                "",
            )
            lines = [line for line in head.strip().splitlines() if line.strip()]
            last = lines[-1] if lines else ""
            if _STAGING_CODE_MISSING.search(last):
                failures = [{"trial": None, "evidence": last[:REASON_EVIDENCE], "provider_cause": "StagingCodeMissing"}]
            elif _STALE_JUDGE_POLICY.search(last):
                failures = [{"trial": None, "evidence": last[:REASON_EVIDENCE], "provider_cause": "StaleJudgePolicyPath"}]
    elif state == "pending_judge_calibration":
        issues = result.get("issues") or []
        head = issues[0] if issues and isinstance(issues[0], str) else ""
        if _TOOLCHAIN_INTEGRITY.match(head):
            gate = "judge_calibration"
            failures = [{"trial": None, "evidence": head[:REASON_EVIDENCE], "provider_cause": "ToolchainIntegrity"}]
    elif state == "pending_repeated_diagnostics":
        failures = _diagnostics_infrastructure_failures(item_root, result)
        gate = "repeated_diagnostics"
    if not failures:
        return None
    causes = sorted({str(failure["provider_cause"]) for failure in failures})
    return {
        "gate": gate,
        "families": sorted({INFRASTRUCTURE_CAUSE_FAMILY[cause] for cause in causes}),
        "causes": causes,
        "failures": failures,
    }


def _validate_infrastructure_health_receipt(path: Path) -> dict[str, Any]:
    try:
        receipt = _read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        raise SynthesisError(f"infrastructure health receipt is unreadable: {error}")
    attempts = receipt.get("provisioning_attempts")
    deletion_lookups = receipt.get("deletion_lookups")
    completed_at = receipt.get("completed_at")
    if (
        receipt.get("schema_version") != "capability-daytona-health-v1"
        or receipt.get("state") != "passed"
        or receipt.get("network_block_all") is not True
        or receipt.get("network_block_all_requested") is not True
        or receipt.get("network_block_all_observed") is not True
        or receipt.get("deleted") is not True
        or receipt.get("lookup_after_delete") != "not_found"
        or not isinstance(receipt.get("snapshot"), str)
        or not receipt["snapshot"]
        or not isinstance(receipt.get("snapshot_id"), str)
        or not receipt["snapshot_id"]
        or not isinstance(receipt.get("sandbox_id"), str)
        or not receipt["sandbox_id"]
        or not isinstance(attempts, list)
        or not attempts
        or attempts[-1].get("state") != "created"
        or not isinstance(deletion_lookups, list)
        or not deletion_lookups
        or any(not isinstance(entry, dict) for entry in deletion_lookups)
        or deletion_lookups[-1].get("state") != "not_found"
        or not isinstance(completed_at, str)
    ):
        raise SynthesisError("infrastructure health receipt did not pass create/delete")
    elapsed_values = [entry.get("elapsed_seconds") for entry in deletion_lookups]
    if (
        any(type(value) not in (int, float) for value in elapsed_values)
        or any(
            not math.isfinite(value) or value < 0
            for value in elapsed_values
            if type(value) in (int, float)
        )
        or elapsed_values[0] != 0
        or elapsed_values != sorted(elapsed_values)
        or elapsed_values[-1] > 60
        or any(
            entry.get("state") not in {"present", "lookup_error", "not_found"}
            for entry in deletion_lookups
        )
    ):
        raise SynthesisError("infrastructure health deletion observations are invalid")
    try:
        completed = datetime.fromisoformat(completed_at)
    except ValueError as error:
        raise SynthesisError(
            "infrastructure health receipt has invalid completed_at"
        ) from error
    now = datetime.now(UTC)
    if completed.tzinfo is None:
        raise SynthesisError(
            "infrastructure health receipt completed_at lacks timezone"
        )
    completed = completed.astimezone(UTC)
    if completed > now + timedelta(minutes=5) or now - completed > timedelta(hours=2):
        raise SynthesisError("infrastructure health receipt is not fresh")
    return receipt


def _archive_infrastructure_revalidation(
    item_root: Path, root: Path, attempt: int
) -> Path:
    destination = (
        root / "infrastructure-history" / item_root.name / f"revalidation-{attempt}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    status = item_root / "status.json"
    if status.is_file():
        shutil.copy2(status, destination / "status.json")
    for name in (
        "harbor",
        "runtime-trials",
        "judge-calibration",
        "runtime-evidence.json",
        "solver-transcripts.json",
        "independent-adversary.json",
        "authored-oracle.json",
        "diagnostics",
    ):
        source = item_root / name
        if source.exists():
            shutil.move(str(source), destination / name)
    return destination


def _runtime_infra_max_retries() -> int:
    raw = os.environ.get("CAPABILITY_RUNTIME_INFRA_MAX_RETRIES", "")
    try:
        value = int(raw) if raw.strip() else DEFAULT_RUNTIME_INFRA_MAX_RETRIES
    except ValueError as error:
        raise SynthesisError(
            f"CAPABILITY_RUNTIME_INFRA_MAX_RETRIES must be a whole number: {raw!r}"
        ) from error
    if not 0 <= value <= 20:
        raise SynthesisError("CAPABILITY_RUNTIME_INFRA_MAX_RETRIES must be between 0 and 20")
    return value


def _infrastructure_revalidations(root: Path, item_root: Path) -> int:
    return len(tuple((root / "infrastructure-history" / item_root.name).glob("revalidation-*")))


def _hold_for_infrastructure(
    root: Path,
    item_root: Path,
    result: dict[str, Any],
    *,
    probe: dict[str, Any] | None = None,
    hold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn an ungraded infrastructure failure into a waiting hold, or a
    terminal failure once the durable revalidation cap is spent.

    Anything that is not classified infrastructure is returned unchanged.
    """
    hold = hold or _infrastructure_hold(item_root, result)
    if hold is None:
        return result
    used = _infrastructure_revalidations(root, item_root)
    maximum = _runtime_infra_max_retries()
    prior = result.get("runtime_infrastructure") if isinstance(result.get("runtime_infrastructure"), dict) else {}
    held = dict(result)
    original_issues = [
        issue for issue in (result.get("issues") or [])
        if isinstance(issue, str)
        and not issue.startswith(("runtime infrastructure hold", "runtime infrastructure retries exhausted", "wait_budget_exhausted:"))
    ]
    summary = f"({hold['gate']}): {', '.join(hold['causes'])}"
    record = {
        "schema_version": "capability-runtime-infrastructure-v1",
        "hold": hold,
        "gate_state": prior.get("gate_state") or (
            result.get("state") if result.get("state") != "pending_runtime_infrastructure" else None
        ),
        "first_seen": prior.get("first_seen") or time.time(),
        "revalidations_used": used,
        "max_revalidations": maximum,
        "last_probe": probe if probe is not None else prior.get("last_probe"),
    }
    held.pop("wait_exhausted", None)
    if used >= maximum:
        held.update(
            state="failed",
            failure_stage="runtime_infrastructure",
            issues=[
                f"runtime infrastructure retries exhausted after {used} gate re-run(s) {summary}",
                *original_issues,
            ],
            runtime_infrastructure={**record, "exhausted": True},
        )
        return held
    held.update(
        state="pending_runtime_infrastructure",
        issues=[
            (
                f"runtime infrastructure hold {summary}; gate re-run {used + 1}/{maximum} "
                "follows a passing health check"
            ),
            *original_issues,
        ],
        runtime_infrastructure=record,
    )
    held.pop("failure_stage", None)
    return held


_PROBE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_PROBE_CACHE_LOCK = threading.Lock()


def _probe_ttl_seconds() -> float:
    raw = os.environ.get("CAPABILITY_INFRA_PROBE_TTL_SECONDS", "")
    try:
        value = float(raw) if raw.strip() else 600.0
    except ValueError:
        return 600.0
    return value if math.isfinite(value) and value >= 0 else 600.0


def _sandbox_probe(root: Path, item_root: Path, daytona_tools: Path | None) -> dict[str, Any]:
    """One provider probe per job per TTL: held items share its verdict.

    Hundreds of held items probing at once would add load to a saturated
    provider; the question "can the provider create a sandbox now" does not
    depend on the item.  Unavailable verdicts are not shared.
    """
    from .infrastructure_health import probe_item_provider

    key = str(daytona_tools or os.environ.get("CAPABILITY_DAYTONA_TOOLS") or "")
    with _PROBE_CACHE_LOCK:
        cached = _PROBE_CACHE.get(key)
        if cached is not None and time.time() - cached[0] < _probe_ttl_seconds():
            receipt_path = cached[1].get("receipt")
            if cached[1]["state"] != "passed" or (receipt_path and Path(receipt_path).is_file()):
                return {**cached[1], "shared": True}
        directory = root / "infrastructure-health" / item_root.name
        number = len(tuple(directory.glob("probe-*.json"))) + 1
        output = directory / f"probe-{number}.json"
        try:
            receipt = probe_item_provider(item_root, output, daytona_tools=daytona_tools)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            receipt = {"state": "failed", "error_type": type(error).__name__}
        check = {
            "state": receipt.get("state"),
            "receipt": str(output) if output.is_file() else None,
            "snapshot": receipt.get("snapshot"),
            "sandbox_id": receipt.get("sandbox_id"),
            "error_type": receipt.get("error_type"),
            "elapsed_seconds": receipt.get("elapsed_seconds"),
        }
        if receipt.get("state") == "passed":
            try:
                _validate_infrastructure_health_receipt(output)
            except SynthesisError as error:
                check.update(state="failed", error_type="ReceiptValidation", error=str(error))
            else:
                check["receipt_sha256"] = _sha256(output)
        if check["state"] in {"passed", "failed"}:
            _PROBE_CACHE[key] = (time.time(), check)
        return check


def _probe_infrastructure(
    root: Path,
    item_root: Path,
    hold: dict[str, Any],
    toolchain: Any,
    daytona_tools: Path | None,
) -> dict[str, Any]:
    """Cheap, automatic evidence that the failed dependency is usable again.

    ``state``: passed (retry now), failed (keep waiting), or unavailable (no
    way to check; retry anyway, still within the durable cap).
    """
    families = set(hold.get("families") or [])
    checks: dict[str, Any] = {}
    if "sandbox" in families:
        checks["sandbox"] = _sandbox_probe(root, item_root, daytona_tools)
    if "toolchain" in families:
        heal = getattr(toolchain, "heal", None)
        if toolchain is None:
            checks["toolchain"] = {"state": "failed", "error": "pinned TaskCompendium toolchain is unavailable"}
        elif not callable(heal):
            checks["toolchain"] = {"state": "unavailable"}
        else:
            try:
                healed = heal()
            except (SynthesisError, OSError, json.JSONDecodeError) as error:
                checks["toolchain"] = {"state": "failed", "error": str(error)[:REASON_EVIDENCE]}
            else:
                checks["toolchain"] = {"state": "passed", "healed": healed}
    if "staging" in families:
        required = [
            PROJECT_ROOT / "capability_pipeline" / "runtime.py",
            PROJECT_ROOT / "capability_pipeline" / "daytona_verifier.py",
            SOURCE_LOCK,
            DRIVER,
            *((daytona_tools / "dt.py",) if daytona_tools else ()),
        ]
        missing = [str(path) for path in required if not path.is_file()]
        checks["staging"] = (
            {"state": "failed", "missing": missing[:8]} if missing else {"state": "passed"}
        )
    for family in families - {"sandbox", "toolchain", "staging"}:
        # glm: its own runtime hold already waited out transport; the conveyor
        # backoff is the gate.  controller: fixed in the running controller.
        checks[family] = {"state": "unavailable"}
    states = {check["state"] for check in checks.values()}
    overall = "failed" if "failed" in states else "passed" if states == {"passed"} else "unavailable"
    return {"state": overall, "at": time.time(), "checks": checks}


def _retry_runtime_infrastructure(
    item: dict[str, Any],
    root: Path,
    agent: OMPAgent,
    toolchain: OfficialToolchain | None,
    runtime_runner: Path | None,
    validation_timeout: int,
    daytona_tools: Path | None,
    item_root: Path,
    prior: dict[str, Any],
    hold: dict[str, Any],
    *,
    operator_health: dict[str, Any] | None = None,
    operator_health_path: Path | None = None,
) -> dict[str, Any]:
    """Re-run a runtime gate that failed on infrastructure, when it is healthy."""
    used = _infrastructure_revalidations(root, item_root)
    if used >= _runtime_infra_max_retries():
        result = _hold_for_infrastructure(root, item_root, prior, hold=hold)
        write_status(item_root, result)
        return result
    probe = None
    health, health_path = operator_health, operator_health_path
    if health is None:
        probe = _probe_infrastructure(root, item_root, hold, toolchain, daytona_tools)
        if probe["state"] == "failed":
            result = _hold_for_infrastructure(root, item_root, prior, probe=probe, hold=hold)
            write_status(item_root, result)
            return result
        sandbox = probe["checks"].get("sandbox") or {}
        if sandbox.get("state") == "passed" and sandbox.get("receipt"):
            health_path = Path(sandbox["receipt"])
            health = {"snapshot": sandbox.get("snapshot"), "sandbox_id": sandbox.get("sandbox_id")}
    attempt = used + 1
    history = _archive_infrastructure_revalidation(item_root, root, attempt)
    failures = [dict(failure) for failure in hold.get("failures") or []]
    for failure in failures:
        artifact = failure.get("artifact")
        if not isinstance(artifact, str):
            continue
        original = Path(artifact)
        try:
            archived = history / original.relative_to(item_root)
        except ValueError:
            continue
        if not archived.is_file() or _sha256(archived) != failure.get("artifact_sha256"):
            raise SynthesisError("archived infrastructure failure evidence changed")
        failure["original_artifact"] = str(original)
        failure["artifact"] = str(archived)
    # The archive keeps a copy of status.json; restore the retained status so
    # the fresh attempt's writes carry its transition history.
    write_status(item_root, dict(prior))
    result = _synthesize_attempt(
        item,
        root,
        agent,
        toolchain,
        runtime_runner,
        validation_timeout,
        daytona_tools,
    )
    revalidation: dict[str, Any] = {
        "attempt": attempt,
        "automatic": operator_health is None,
        "gate": hold.get("gate"),
        "causes": hold.get("causes"),
        "prior_failures": failures,
        "prior_attempt": str(history),
        "probe": probe,
    }
    if health_path is not None:
        revalidation.update(
            health_receipt=str(health_path),
            health_receipt_sha256=_sha256(health_path),
            health_snapshot=(health or {}).get("snapshot"),
            health_sandbox_id=(health or {}).get("sandbox_id"),
        )
    result["infrastructure_revalidation"] = revalidation
    result.pop("runtime_infrastructure", None)
    result = _hold_for_infrastructure(root, item_root, result)
    write_status(item_root, result)
    return result


def _validated_pending_adversary_retry(item_root: Path) -> dict[str, Any]:
    """Bind a retry request to retained, ungraded exhaustion evidence only."""
    status_path = item_root / "status.json"
    evidence_path = item_root / "runtime-evidence.json"
    harbor = item_root / "harbor"
    try:
        status = _read_json(status_path)
        evidence = _read_json(evidence_path)
    except (OSError, json.JSONDecodeError) as error:
        raise SynthesisError(
            "adversary retry lacks retained status or runtime evidence"
        ) from error
    retained = status.get("incomplete_adversary")
    current = _incomplete_adversary_from_evidence(evidence, evidence_path, harbor)
    state = status.get("state")
    if state == "pending_adversary_retry":
        if not isinstance(retained, dict) or retained != current:
            raise SynthesisError(
                "pending adversary retry evidence is absent or altered"
            )
    elif state == "failed":
        # Older controller bytes labeled this exact no-grade shape as failed.
        # Permit a forward-only reclassification only if every non-adversary
        # control gate is retained and passed; do not rewrite historical state.
        legacy_issues = status.get("issues")
        if (
            not isinstance(legacy_issues, list)
            or len(legacy_issues) != len(LEGACY_INCOMPLETE_ADVERSARY_ISSUES)
            or set(legacy_issues) != LEGACY_INCOMPLETE_ADVERSARY_ISSUES
        ):
            raise SynthesisError(
                "--retry-adversary failed-state compatibility requires the exact "
                "legacy incomplete-adversary issues"
            )
        controls_path = item_root / "workspace" / "task" / "controls.json"
        try:
            controls = _read_json(controls_path)
            controls_passed, control_issues = _controls_pass(
                controls, evidence, external=True
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise SynthesisError(
                "legacy adversary retry lacks verifiable passed control evidence"
            ) from error
        attestation = evidence.get("attestation")
        if (
            not controls_passed
            or not isinstance(attestation, dict)
            or attestation.get("oracle", {}).get("authored") is not True
            or attestation.get("solver", {}).get("state") != "passed"
            or attestation.get("adversarial", {}).get("authored_controls_executed")
            is not True
        ):
            raise SynthesisError(
                "legacy adversary retry has non-adversary gate failures: "
                + ", ".join(control_issues)
            )
    else:
        raise SynthesisError(
            "--retry-adversary requires pending_adversary_retry or the exact "
            "legacy incomplete-adversary failed state"
        )
    if current is None:
        raise SynthesisError(
            "adversary retry evidence is absent, altered, or not bounded output exhaustion"
        )
    return current


def _archive_adversary_revalidation(item_root: Path, root: Path, attempt: int) -> Path:
    destination = (
        root / "adversary-history" / item_root.name / f"revalidation-{attempt}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    for name in (
        "status.json",
        "harbor",
        "runtime-trials",
        "runtime-evidence.json",
        "solver-transcripts.json",
        "independent-adversary.json",
        "authored-oracle.json",
    ):
        source = item_root / name
        if source.exists():
            shutil.move(str(source), destination / name)
    return destination


def _fresh_construction_repair_allowed(
    result: dict[str, Any], *, expected_item_root: Path | None = None,
    expected_key: str | None = None, expected_proposal_hash: str | None = None,
) -> bool:
    """Keep ungraded transport and adjudication outcomes out of GLM task repair."""
    state = result.get("state")
    if state == "pending_attack_adjudication":
        return _proven_exploit_for_repair(result)
    if state == "pending_judge_calibration":
        return _measured_judge_calibration_failure(
            result, expected_item_root=expected_item_root,
            expected_key=expected_key, expected_proposal_hash=expected_proposal_hash,
        )
    if state == "pending_build_acceptance":
        return True
    if state == "pending_quality_review":
        review = result.get("quality_review")
        return isinstance(review, dict) and review.get("state") in {
            "repair", "reject", "insufficient_evidence"
        }
    if state != "failed":
        return False
    issues = result.get("issues")
    if not isinstance(issues, list) or not issues:
        return False
    if all(
        isinstance(issue, str)
        and issue.startswith(("invalid task bundle:", "TaskCompendium validation/lowering failed:"))
        for issue in issues
    ):
        return True
    # Controller-observed semantic mismatches can improve through a task edit.
    # A graded empty/malformed response contradicting the authored extraction
    # expectation is also actionable: grading completed, so this is a task
    # contract mismatch rather than an ungraded extraction/transport failure.
    # Unknown attestation, isolation, extraction, or transport failures remain
    # operational holds until classified; a model cannot repair an ungraded run.
    semantic_control = re.compile(
        r"[^:]+: (?:reward is below reward_min|reward is above reward_max|"
        r"criterion assertion failed at .+|positive control did not earn >=0\.8|"
        r"negative control earned >0\.2|"
        r"status 'graded', expected 'extraction_error')"
    )
    return all(isinstance(issue, str) and semantic_control.fullmatch(issue) for issue in issues)


def _measured_judge_calibration_failure(
    result: dict[str, Any], *, expected_item_root: Path | None = None,
    expected_key: str | None = None, expected_proposal_hash: str | None = None,
) -> bool:
    """Only complete, correctly routed graded calibration can drive task edits."""
    from .inference import digest
    from .judge import (
        _assess_calibration,
        _judge_path,
        _machine_gate_path,
        validate_task_calibration_fixture,
    )

    record = result.get("judge_calibration_failure")
    if not isinstance(record, dict):
        return False
    try:
        item_root = Path(result["item_root"]).resolve()
        if (
            (expected_item_root is not None and item_root != expected_item_root.resolve())
            or (expected_key is not None and result.get("key") != expected_key)
            or (expected_proposal_hash is not None and result.get("proposal_hash") != expected_proposal_hash)
        ):
            return False
        bundle = item_root / "workspace" / "task"
        report_path = Path(record["artifact"]).resolve()
        raw_path = Path(record["raw_artifact"]).resolve()
        fixture_path = Path(record["fixture_artifact"]).resolve()
        if (
            report_path != item_root / "judge-calibration" / "judge-calibration.json"
            or raw_path != item_root / "judge-calibration" / "taskcompendium-judge-results.json"
            or fixture_path != bundle / "judge-calibration.json"
            or any(
                _sha256(path) != record[key]
                for path, key in (
                    (report_path, "artifact_sha256"),
                    (raw_path, "raw_artifact_sha256"),
                    (fixture_path, "fixture_artifact_sha256"),
                )
            )
        ):
            return False
        fixture = _read_json(fixture_path)
        raw = _read_json(raw_path)
        report = _read_json(report_path)
        composite_path = bundle / "composite-verifier.json"
        composite = composite_path.is_file()
        mode = "taskcompendium-composite-harbor" if composite else "taskcompendium-native-judge"
        repeats, max_spread = report.get("repeats"), report.get("max_spread")
        if (
            report.get("state") != "failed"
            or report.get("mode") != mode
            or (composite and raw.get("mode") != mode)
            or report.get("specification_sha256") != _sha256(bundle / "specification.json")
            or report.get("fixture_hash") != digest(fixture)
            or report.get("composite_config_sha256")
            != (_sha256(composite_path) if composite else None)
            or report.get("results") != raw.get("results")
            or report.get("failures") != raw.get("failures")
            or report.get("failures") != {}
            or type(repeats) is not int
            or type(max_spread) not in (int, float)
            or repeats != 3
            or max_spread != 0.15
        ):
            return False
        cases = validate_task_calibration_fixture(
            fixture, report["specification_sha256"], repeats, max_spread,
            composite=composite,
        )
        results = report["results"]
        expected = {f"{case['id']}:{repeat}" for case in cases for repeat in range(repeats)}
        if not isinstance(results, dict) or set(results) != expected:
            return False
        for case in cases:
            for repeat in range(repeats):
                graded = results[f"{case['id']}:{repeat}"]
                reward = graded.get("reward")
                if (
                    graded.get("status") != "graded"
                    or type(reward) not in (int, float)
                    or not math.isfinite(reward)
                    or not 0 <= reward <= 1
                    or _judge_path(graded) != case["expected_judge_path"]
                    or (
                        composite
                        and _machine_gate_path(graded)
                        != ("passed" if case["expected_machine_gate"] == "pass" else "failed")
                    )
                ):
                    return False
        issues, metrics = _assess_calibration(
            cases, results, {}, repeats, max_spread,
            require_model_judgment=True, require_composite_gate_pass=composite,
        )
        if not issues or issues != report.get("issues") or metrics != report.get("metrics"):
            return False
        if composite:
            expected_hashes = {
                "composite_adapter_sha256": _sha256(Path(__file__).with_name("composite_verifier.py")),
                "composite_policy_sha256": _sha256(Path(__file__).with_name("composite_policy.py")),
                "composite_config_sha256": _sha256(composite_path),
            }
            if any(
                not isinstance(graded.get("detail"), dict)
                or any(graded["detail"].get(name) != value for name, value in expected_hashes.items())
                for graded in results.values()
            ):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _proven_exploit_for_repair(result: dict[str, Any]) -> bool:
    """Require a complete, immutable adjudication of graded rewarded attacks."""
    from .attack_adjudication import SCHEMA, rewarded_steps, validate_receipt
    from .inference import digest
    from .quality import sha256, source_files

    record = result.get("attack_adjudication")
    if not isinstance(record, dict) or record.get("state") != "needs_repair_or_retry":
        return False
    try:
        item_root = Path(result["item_root"]).resolve()
        result_path = Path(record["artifact"]).resolve()
        review_root = result_path.parent
        if result_path.name != "result.json" or sha256(result_path) != record["artifact_sha256"]:
            return False
        manifest = _read_json(review_root / "input-manifest.json")
        identity = {key: value for key, value in manifest.items() if key != "snapshot_hash"}
        if (
            manifest.get("schema_version") != SCHEMA + "-input"
            or manifest.get("snapshot_hash") != digest(identity)
            or manifest["snapshot_hash"] != record.get("snapshot_hash")
            or manifest.get("files")
            != {name: sha256(path) for name, path in source_files(item_root).items()}
        ):
            return False
        adjudication = _read_json(result_path)
        receipt_path = review_root / "receipt.json"
        if (
            adjudication.get("schema_version") != SCHEMA + "-result"
            or adjudication.get("snapshot_hash") != manifest["snapshot_hash"]
            or adjudication.get("state") != "needs_repair_or_retry"
            or adjudication.get("receipt_sha256") != sha256(receipt_path)
            or adjudication.get("execution", {}).get("returncode") != 0
            or adjudication.get("execution", {}).get("timed_out") is not False
            or adjudication.get("reviewer_policy", {}).get("independent_session") is not True
        ):
            return False
        receipt = _read_json(receipt_path)
        remaining = validate_receipt(receipt, manifest, review_root / "input")
        report = _read_json(item_root / "independent-adversary.json")
        targets = rewarded_steps(report)
        if (
            not remaining
            or remaining != adjudication.get("issues")
            or any(
                not issue.endswith(": rewarded attack requires independent adjudication")
                for issue in remaining
            )
        ):
            return False
        if report.get("independent") is not True:
            return False
        for case in report.get("cases", []):
            if case.get("error") or not case.get("steps"):
                return False
            for step in case["steps"]:
                if step.get("result", {}).get("status") != "graded":
                    return False
                for kind in ("grading", "transcript"):
                    path = step.get(kind + "_artifact")
                    if not isinstance(path, str) or (
                        step.get(kind + "_sha256") != manifest["files"].get(path)
                    ):
                        return False
                if _read_json(item_root / step["grading_artifact"]) != step["result"]:
                    return False
        decisions = {
            (case["strategy"], case["step_index"]): case["disposition"]
            for case in receipt["cases"]
        }
        return (
            bool(targets)
            and any(decisions.get(key) == "exploit" for key in targets)
            and all(
                decisions.get(key) in {"exploit", "legitimate_correct", "legitimate_partial"}
                for key in targets
            )
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _used_repair_rounds(root: Path, item_name: str) -> set[int]:
    numbers: set[int] = set()
    for parent in (root / "repairs" / item_name, root / "repair-budget" / item_name):
        for path in parent.glob("attempt-*"):
            match = re.fullmatch(r"attempt-(\d+)", path.name)
            if match:
                numbers.add(int(match.group(1)))
    return numbers


def _reserve_repair_round(root: Path, item_root: Path, attempt: int) -> dict[str, str]:
    """Count the round before invoking GLM and retain its triggering failure."""
    status = item_root / "status.json"
    if not status.is_file():
        raise SynthesisError("repair requires a retained source failure status")
    reservation = root / "repair-budget" / item_root.name / f"attempt-{attempt}"
    reservation.mkdir(parents=True, exist_ok=False)
    snapshot = reservation / "source-status.json"
    shutil.copy2(status, snapshot)
    record = {
        "artifact": str(snapshot),
        "artifact_sha256": _sha256(snapshot),
        "source_status_sha256": _sha256(status),
    }
    atomic_json(reservation / "reservation.json", record)
    return record


def _max_repair_rounds() -> int:
    """CAPABILITY_MAX_REPAIR_ROUNDS, validated (SynthesisError: a deterministic precondition)."""
    raw = os.environ.get("CAPABILITY_MAX_REPAIR_ROUNDS", "2")
    try:
        value = int(raw)
    except ValueError as error:
        raise SynthesisError(
            f"CAPABILITY_MAX_REPAIR_ROUNDS must be a whole number: {raw!r}"
        ) from error
    if value < 0 or value > 5:
        raise SynthesisError(
            "CAPABILITY_MAX_REPAIR_ROUNDS must be between zero and five"
        )
    return value


def synthesize_one(
    item: dict[str, Any],
    root: Path,
    agent: OMPAgent,
    toolchain: OfficialToolchain | None,
    runtime_runner: Path | None,
    validation_timeout: int,
    daytona_tools: Path | None = None,
    infrastructure_health: dict[str, Any] | None = None,
    infrastructure_health_path: Path | None = None,
    retry_adversary: bool = False,
    acceptance_seeds: list[dict[str, Any]] = (),
    retry_exhausted_waits: bool = False,
) -> dict[str, Any]:
    # The repair-loop states; conveyor.STATE_TABLE is the single definition.
    repairable = set(REPAIRABLE_STATES)
    proposal = item["proposal"]
    key = f"{proposal['capability_id']}:{proposal['slot']}"
    item_root = root / "items" / f"{_safe_name(key)}-{item['proposal_hash'][:12]}"
    prior_status = item_root / "status.json"
    prior_result = None
    # A quality acceptance is terminal.  On resume (including a relaunch under a
    # new run name) keep it without re-running any gate or review, provided the
    # reviewed task artefacts are byte-identical; otherwise fall through.
    from .acceptance import resume_accepted

    try:
        restored_status = _read_json(prior_status) if prior_status.is_file() else None
    except (OSError, json.JSONDecodeError):
        restored_status = None
    resumed = resume_accepted(
        item, key, root, item_root, restored_status, list(acceptance_seeds)
    )
    if resumed is not None:
        return resumed
    if prior_status.is_file():
        try:
            candidate = _read_json(prior_status)
            reopened = (
                retry_exhausted_waits
                and candidate.get("state") == "failed"
                and isinstance(candidate.get("wait_exhausted"), dict)
            )
            if reopened:
                # Explicit operator request (--retry-exhausted-waits): re-enter
                # the step whose conveyor wait budget ran out, with a fresh budget.
                pass
            elif candidate.get("state") == "pending_readmission":
                # Terminal for synthesis: readmission yields a new proposal hash,
                # hence a new item.  Re-entering would re-pay every runtime gate.
                return candidate
            # A bounded attack generator exhaustion has no candidate or grade
            # for the semantic repair loop to improve.  Preserve it until an
            # explicit fresh adversary-suite revalidation is requested.
            elif candidate.get("state") == "pending_adversary_retry" or (
                retry_adversary and candidate.get("state") == "failed"
            ):
                if not retry_adversary:
                    return candidate
                exhausted = _validated_pending_adversary_retry(item_root)
                history_parent = root / "adversary-history" / item_root.name
                attempt = len(tuple(history_parent.glob("revalidation-*"))) + 1
                history = _archive_adversary_revalidation(item_root, root, attempt)
                # The archive moved status.json; restore the retained status so
                # the fresh attempt's writes carry its transition history.
                write_status(item_root, dict(candidate))
                result = _synthesize_attempt(
                    item,
                    root,
                    agent,
                    toolchain,
                    runtime_runner,
                    validation_timeout,
                    daytona_tools,
                )
                result["adversary_revalidation"] = {
                    "attempt": attempt,
                    "prior_attempt": str(history),
                    "prior_incomplete_adversary": exhausted,
                    "policy": "fresh_full_independent_adversary_suite",
                }
                write_status(item_root, result)
                return result
            elif candidate.get("state") in repairable or candidate.get("state") == (
                "pending_runtime_infrastructure"
            ):
                prior_result = candidate
                # A prior job's hold ends with this re-entry.
                prior_result.pop("wait_hold", None)
        except (OSError, json.JSONDecodeError):
            pass
    max_rounds = _max_repair_rounds()
    repair_parent = root / "repairs" / item_root.name
    used_numbers = _used_repair_rounds(root, item_root.name)
    used_rounds = max(used_numbers, default=0)
    if prior_result is not None:
        # An ungraded provider/transport/toolchain failure is re-run, never
        # repaired: automatically after a health probe (or with the operator's
        # --retry-infrastructure receipt), within a durable per-item cap.
        hold = _infrastructure_hold(item_root, prior_result)
        if hold is not None:
            return _retry_runtime_infrastructure(
                item,
                root,
                agent,
                toolchain,
                runtime_runner,
                validation_timeout,
                daytona_tools,
                item_root,
                prior_result,
                hold,
                operator_health=infrastructure_health,
                operator_health_path=infrastructure_health_path,
            )
    repairs = list(prior_result.get("repairs", [])) if prior_result else []
    result = prior_result or _synthesize_attempt(
        item,
        root,
        agent,
        toolchain,
        runtime_runner,
        validation_timeout,
        daytona_tools,
    )
    # Fresh machine feedback enters this bounded revision loop. The same
    # eligibility check applies after each validation and across resumes, so
    # an ungraded provider failure cannot become a task edit on the next turn.
    repair_identity = {
        "expected_item_root": item_root,
        "expected_key": key,
        "expected_proposal_hash": item["proposal_hash"],
    }
    repair_enabled = _fresh_construction_repair_allowed(result, **repair_identity)
    available_rounds = max(0, max_rounds - used_rounds) if repair_enabled else 0
    for round_ in range(available_rounds):
        if result.get("state") not in repairable:
            break
        if not _fresh_construction_repair_allowed(result, **repair_identity):
            break
        attempt_number = used_rounds + round_ + 1
        repair_root = repair_parent / f"attempt-{attempt_number}"
        failure_snapshot = _reserve_repair_round(root, item_root, attempt_number)
        activity, current_activity, activity_history = _repair_activity(
            item_root, repair_root, attempt_number, result
        )
        mark_activity(item_root, "construction_repair", attempt=attempt_number)
        try:
            from .repair import run_repair

            repaired = run_repair(
                item_root,
                repair_root,
                agent,
                _repair_feedback(item, result, attempt_number, item_root),
                source=_agent_package_root(toolchain, item_root / "workspace"),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            _close_repair_activity(
                activity,
                current_activity,
                activity_history,
                "error",
                {
                    "exception_type": type(error).__name__,
                    "message": str(error),
                },
            )
            result["repair_issue"] = (
                f"bounded construction repair could not run: {error}"
            )
            result["repairs"] = repairs
            write_status(item_root, result)
            return result
        repair_result = repair_root / "result.json"
        _close_repair_activity(
            activity,
            current_activity,
            activity_history,
            "completed",
            {
                "repair_state": repaired.get("state"),
                "result_artifact": str(repair_result),
                "result_artifact_sha256": (
                    _sha256(repair_result) if repair_result.is_file() else None
                ),
            },
        )
        repair_record = {
            "round": attempt_number,
            "state": repaired.get("state"),
            "artifact": str(repair_result),
            "artifact_sha256": _sha256(repair_result),
            "changed_files": repaired.get("changed_files", []),
            "source_failure": failure_snapshot,
        }
        repairs.append(repair_record)
        if repaired.get("state") == "needs_readmission":
            result.update(
                state="pending_readmission",
                issues=[
                    "construction repair found that the admitted design needs readmission"
                ],
                repairs=repairs,
            )
            write_status(item_root, result)
            return result
        if repaired.get("state") != "ready_for_validation":
            receipt = repair_root / "receipt.json"
            remaining = []
            if receipt.is_file():
                try:
                    remaining = _read_json(receipt).get("remaining_issues", [])
                except (OSError, json.JSONDecodeError):
                    pass
            result["issues"] = (
                repaired.get("issues") or remaining or result.get("issues", [])
            )
            continue
        history = _archive_attempt(item_root, root, attempt_number)
        repair_record["prior_attempt"] = str(history)
        result = _synthesize_attempt(
            item,
            root,
            agent,
            toolchain,
            runtime_runner,
            validation_timeout,
            daytona_tools,
        )
    if repairs:
        result["repairs"] = repairs
        if result.get("state") in repairable and repairs[-1]["state"] != (
            "ready_for_validation"
        ):
            result["repair_issue"] = "bounded construction repair needs continuation"
    final_used = max(_used_repair_rounds(root, item_root.name), default=0)
    actionable = _fresh_construction_repair_allowed(result, **repair_identity)
    result["repair_budget"] = {
        "used": final_used,
        "max": max_rounds,
        "exhausted": final_used >= max_rounds,
    }
    if actionable and final_used >= max_rounds:
        result["terminal_disposition"] = "rejected"
    else:
        result.pop("terminal_disposition", None)
    result = _hold_for_infrastructure(root, item_root, result)
    write_status(item_root, result)
    return result


def _conveyor_classification(result: dict[str, Any], entry: ConveyorEntry):
    """conveyor.classify_result, with this item's repair eligibility."""
    return classify_result(
        result,
        repair_actionable=lambda value: _fresh_construction_repair_allowed(
            dict(value),
            expected_item_root=entry.item_root,
            expected_key=entry.key,
            expected_proposal_hash=entry.proposal_hash,
        ),
    )


def synthesize(args) -> int:
    if args.limit is not None and args.limit < 1:
        raise SynthesisError("limit must be positive")
    if args.concurrency < 1:
        raise SynthesisError("concurrency must be positive")
    try:
        conveyor_config = ConveyorConfig.from_env()
    except ConveyorConfigError as error:
        raise SynthesisError(str(error)) from error
    # Validated before any item runs: inside an item call a bad value would be
    # a controller exception on every item instead of one clear job error.
    _max_repair_rounds()
    accepted = load_accepted(Path(args.accepted), args.limit)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    retry_infrastructure = bool(getattr(args, "retry_infrastructure", False))
    retry_adversary = bool(getattr(args, "retry_adversary", False))
    retry_exhausted_waits = bool(getattr(args, "retry_exhausted_waits", False))
    receipt_value = getattr(args, "infrastructure_health_receipt", None)
    source_health_path = Path(receipt_value).resolve() if receipt_value else None
    if retry_infrastructure and source_health_path is None:
        raise SynthesisError(
            "--retry-infrastructure requires --infrastructure-health-receipt"
        )
    if source_health_path is not None and not retry_infrastructure:
        raise SynthesisError(
            "--infrastructure-health-receipt requires --retry-infrastructure"
        )
    if retry_adversary and retry_infrastructure:
        raise SynthesisError(
            "--retry-adversary and --retry-infrastructure are separate revalidation paths"
        )
    infrastructure_health = (
        _validate_infrastructure_health_receipt(source_health_path)
        if source_health_path
        else None
    )
    health_path = source_health_path
    with (root / ".controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if source_health_path is not None:
            health_path = root / "infrastructure-health-receipt.json"
            if health_path.is_file():
                if _sha256(health_path) != _sha256(source_health_path):
                    raise SynthesisError("frozen infrastructure health receipt changed")
            else:
                temporary = health_path.with_suffix(".tmp")
                shutil.copy2(source_health_path, temporary)
                temporary.replace(health_path)
        started = time.time()
        validated_root = root / "validated"
        if validated_root.exists():
            shutil.rmtree(validated_root)
        toolchain_error = None
        try:
            toolchain = OfficialToolchain.resolve(root, args.taskcompendium_source)
            builder_copy = getattr(toolchain, "builder_copy", None)
            if callable(builder_copy):
                try:
                    toolchain = builder_copy()
                except OSError as error:
                    print(json.dumps({"event": "builder_toolchain_copy_failed", "error": repr(error)}), flush=True)
        except (
            SynthesisError,
            OSError,
            json.JSONDecodeError,
            subprocess.SubprocessError,
            ValueError,
        ) as error:
            toolchain, toolchain_error = None, str(error)
        executable = shutil.which(args.omp) if os.sep not in args.omp else args.omp
        if not executable or not Path(executable).exists():
            raise SynthesisError(f"OMP executable is unavailable: {args.omp}")
        runtime_runner = (
            Path(args.runtime_runner).resolve() if args.runtime_runner else None
        )
        if runtime_runner and (
            not runtime_runner.is_file() or not os.access(runtime_runner, os.X_OK)
        ):
            raise SynthesisError("runtime runner must be an executable file")
        daytona_value = args.daytona_tools or os.environ.get("CAPABILITY_DAYTONA_TOOLS")
        daytona_tools = Path(daytona_value).resolve() if daytona_value else None
        if daytona_tools:
            missing_tools = [
                name
                for name in (
                    "dt.py",
                    "dt.sh",
                    "validate_env.py",
                    "verify.py",
                    "adapter.py",
                )
                if not (daytona_tools / name).is_file()
            ]
            if missing_tools:
                raise SynthesisError(
                    "Daytona tool directory is incomplete: " + ", ".join(missing_tools)
                )
        overlay_value = (
            args.research_overlay
            or os.environ.get("CAPABILITY_OMP_CONFIG")
            or os.environ.get("RESEARCH_OVERLAY")
        )
        research_overlay = Path(overlay_value).resolve() if overlay_value else None
        if research_overlay and not research_overlay.is_file():
            raise SynthesisError(f"research overlay is unavailable: {research_overlay}")
        agent = OMPAgent(
            str(executable),
            args.model,
            args.session_time,
            args.max_continuations,
            research_overlay,
        )
        atomic_json(
            root / "run.json",
            {
                "stage": "synthesize",
                "accepted": str(Path(args.accepted).resolve()),
                "accepted_count": len(accepted),
                "concurrency": args.concurrency,
                "tier": args.tier,
                "session_time": args.session_time,
                "max_continuations": args.max_continuations,
                "max_stagnant_attempts": agent.max_stagnant_attempts,
                "taskcompendium_revision": _read_json(SOURCE_LOCK)["revision"],
                "toolchain_error": toolchain_error,
                "runtime_runner_sha256": _sha256(runtime_runner)
                if runtime_runner
                else None,
                "daytona_tools": str(daytona_tools) if daytona_tools else None,
                "research_overlay": str(research_overlay) if research_overlay else None,
                "retry_infrastructure": retry_infrastructure,
                "retry_adversary": retry_adversary,
                "retry_exhausted_waits": retry_exhausted_waits,
                "conveyor": conveyor_config.describe(),
                "infrastructure_health_receipt": str(health_path)
                if health_path
                else None,
                "infrastructure_health_receipt_sha256": _sha256(health_path)
                if health_path
                else None,
                "started": started,
            },
        )
        from .acceptance import load_seeds

        seed_value = getattr(args, "acceptance_seed", None)
        seed_path = Path(seed_value) if seed_value else None
        if seed_path is not None and not seed_path.is_absolute() and not seed_path.exists():
            # Staged next to accepted.json (inputs/source/) by submit.sh.
            seed_path = Path(args.accepted).parent / seed_path
        acceptance_seeds = load_seeds(seed_path)
        entries = []
        for item in accepted:
            proposal = item["proposal"]
            key = f"{proposal['capability_id']}:{proposal['slot']}"
            entries.append(
                ConveyorEntry(
                    key=key,
                    item_root=root / "items" / f"{_safe_name(key)}-{item['proposal_hash'][:12]}",
                    proposal_hash=item["proposal_hash"],
                    # Bind this item's seeds now: a free ``key`` in the lambda would be
                    # read at call time and hand every item the last item's seeds.
                    call=lambda value=item, seeds=acceptance_seeds.get(key, []): synthesize_one(
                        value,
                        root,
                        agent,
                        toolchain,
                        runtime_runner,
                        args.validation_timeout,
                        daytona_tools,
                        infrastructure_health,
                        health_path,
                        retry_adversary,
                        seeds,
                        retry_exhausted_waits,
                    ),
                )
            )
        # Every item runs to a terminal state here: waiting items are re-queued
        # with a not-before time (holding no slot) until their budget runs out.
        outcome = ConveyorScheduler(
            root,
            entries,
            args.concurrency,
            classify=_conveyor_classification,
            # SynthesisError is a deterministic controller precondition; retrying
            # it cannot help.  Anything else gets a bounded number of retries.
            retry_exception=lambda error: not isinstance(error, SynthesisError),
            config=conveyor_config,
        ).run()
        ordered, failures = outcome.results, outcome.failures
        by_task_id: dict[str, list[dict[str, Any]]] = {}
        for result in ordered:
            task_id = (result.get("taskcompendium") or {}).get("id")
            if task_id:
                by_task_id.setdefault(task_id, []).append(result)
        for task_id, collisions in by_task_id.items():
            if len(collisions) < 2:
                continue
            for result in collisions:
                export = result.pop("export", None)
                if export and Path(export).exists():
                    shutil.rmtree(export)
                result.update(
                    state="failed",
                    issues=[f"duplicate generated TaskSpec id: {task_id}"],
                    failure_stage="duplicate_task_id",
                    conveyor={
                        **(result.get("conveyor") or {}),
                        "class": "terminal",
                        "reason": "duplicate_task_id",
                    },
                )
                write_status(Path(result["item_root"]), result, intermediate=False)
        atomic_json(root / "tasks.json", ordered)
        counts = Counter(result["state"] for result in ordered)
        terminal = summarize(ordered)
        all_terminal = len(ordered) == len(accepted) and all(
            (result.get("conveyor") or {}).get("class") == "terminal" for result in ordered
        )
        report = {
            "stage": "synthesize",
            "accepted_inputs": len(accepted),
            "completed_items": len(ordered),
            "states": dict(counts),
            "runtime_validated_tasks": sum(
                result.get("runtime_validated") is True for result in ordered
            ),
            "quality_accepted_tasks": counts["quality_accepted"],
            "failures": failures,
            "toolchain_error": toolchain_error,
            "elapsed_seconds": time.time() - started,
            # Kept for coverage/generate compatibility: "complete" only when every
            # item is quality accepted.  ``all_terminal`` is the job outcome.
            "state": "complete"
            if counts["quality_accepted"] == len(accepted) and not failures
            else "needs_continuation",
            "all_terminal": all_terminal,
            **terminal,
            "conveyor": outcome.summary,
        }
        # A failed item is an outcome, not a request for continuation: exit 0
        # once every item is terminal.  A missing toolchain is a job fault an
        # Iris retry can clear (items re-enter cheaply), so it stays exit 2.
        # So is a transient controller exception: those items are held (state
        # kept), and only a retry or relaunch re-enters them.
        exception_holds = int(outcome.summary.get("controller_exception_holds", 0))
        report["controller_exception_holds"] = exception_holds
        exit_code = (
            0 if all_terminal and toolchain_error is None and not exception_holds else 2
        )
        report["exit_code"] = exit_code
        atomic_json(root / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return exit_code


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "synthesize", help="Build accepted proposals in durable OMP session DAGs"
    )
    parser.add_argument(
        "--accepted", required=True, help="accepted.json emitted by propose"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--tier", choices=("interactive", "bulk"), default="interactive"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--omp", default="omp")
    parser.add_argument(
        "--model",
        default="glm-orion/glm-5.3",
        help="OMP model selector for every model role",
    )
    parser.add_argument(
        "--session-time", type=int, default=28_800, help="seconds per OMP continuation"
    )
    parser.add_argument("--max-continuations", type=int, default=8)
    parser.add_argument(
        "--validation-timeout",
        type=int,
        default=14_400,
        help="seconds for the complete retained runtime gate, including long adversary calls",
    )
    parser.add_argument(
        "--taskcompendium-source", help="exact pinned lib/taskcompendium checkout"
    )
    parser.add_argument(
        "--runtime-runner", help="executable Harbor control runner boundary"
    )
    parser.add_argument(
        "--daytona-tools",
        help="directory containing maintained dt/validate_env helpers",
    )
    parser.add_argument(
        "--research-overlay", help="OMP config enabling the staged research provider"
    )
    parser.add_argument(
        "--retry-infrastructure",
        action="store_true",
        help="rerun only a retained ungraded Daytona provider failure",
    )
    parser.add_argument(
        "--infrastructure-health-receipt",
        help="passed capability-daytona-health-v1 create/delete receipt",
    )
    parser.add_argument(
        "--retry-adversary",
        action="store_true",
        help="rerun every fresh independent adversary after retained bounded output exhaustion",
    )
    parser.add_argument(
        "--acceptance-seed",
        help="prior quality_accepted status.json file, JSON list or directory; each "
        "claim is re-verified against the restored review and task bytes",
    )
    parser.add_argument(
        "--retry-exhausted-waits",
        action="store_true",
        help="re-enter items whose conveyor wait budget ran out (failed with "
        "wait_exhausted), each with a fresh budget",
    )
    parser.set_defaults(func=synthesize)

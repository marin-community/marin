"""Reconstruct the pre-receipt controller's post-review gate from retained evidence.

Every deployed pre-ledger controller (synthesis.py 29251c2081d0, 759adfb8bcef,
40c5259574db and 4bd6b32b6e12; the gate code and every diagnostics module are
byte-identical across all 482 run-003 submissions) decided acceptance as::

    diagnostics = _repeated_quality_diagnostics(...)   # repeat -> grading -> reset
    if diagnostics.state != "ready" and diagnostics.reviewable is not True:
        -> pending_repeated_diagnostics          (no semantic review)
    review = run_review(item_root, quality/<item>/attempt-N)
    if review.state != "accept": -> pending_quality_review
    if diagnostics.state != "ready":
        -> pending_repeated_diagnostics "semantic review accepted despite failed
           repeated runtime gates"
    -> quality_accepted

An accepting review therefore proves acceptance only if ``diagnostics.state``
was ``"ready"`` when the review ran.  New controllers write that value beside the
review (``controller-gate.json``); legacy reviews have no receipt.  This module
recomputes it as a pure function of the review's own frozen input snapshot
(``quality/<item>/attempt-N/input`` plus ``input-manifest.json``).  The snapshot
is taken immediately after the diagnostics call and ``run_review`` verifies the
item did not change during review, so the snapshot's ``diagnostics/**`` is
exactly the evidence the gate saw.  Nothing is executed: no runner, sandbox or
model.

What the old code computed, stage by stage (all reproduced here):

* repeated (``diagnostics.run_repeated_diagnostics``): walks
  ``sorted(diagnostics/attempt-*)`` and returns the first attempt that has a
  ``result.json`` and whose ``inputs/manifest.json`` is unreadable (-> pending)
  or carries exactly the current item identity and controller identity (reuse),
  else runs a new attempt.  State is ``ready`` iff the evaluation matrix says
  ``repeated_runtime_passed`` with a complete inventory of three ``valid``
  cells; ``reviewable`` iff the inventory is complete and valid.
* fixed grading: not applicable for judge tasks without a composite verifier;
  composite replay (``diagnostics/composite-grading``) for composed judge tasks;
  otherwise ``grading_diagnostics`` under ``<attempt>/fixed-grading`` unless
  the bundle is outside the regrade surface (``unsupported``, which keeps the
  repeated state).  A non-ready grading state makes the result pending.
* reset: ``reset_runner`` (container) or ``non_docker_reset``; a non-ready
  reset makes the result pending.

Some identity inputs are not in the snapshot: the controller's own source hashes,
the TaskCompendium lock, the validation timeout, the Daytona helper
(``workspace/tools`` is excluded from review snapshots) and the ShellSim
bridge.  Every retained diagnostics attempt records them, so each distinct
recorded environment is a candidate for the review-time environment, and the
true one is among them (the attempt the gate used carries it).  A candidate
under which the old code provably could not have reached review (for example a
reset attempt frozen under a different controller) is eliminated.  The verdict
is ``ready`` only if every remaining candidate reconstructs to ``ready``;
``not_ready`` only if every one reconstructs to a reviewable non-ready state;
anything else, including any unreadable or absent decision input, is
``unknown`` (the caller treats that as not proven).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from .inference import digest

SCHEMA = "capability-legacy-gate-reconstruction-v1"
READY, NOT_READY, UNKNOWN = "ready", "not_ready", "unknown"

_DIAGNOSTICS_SCHEMA = "capability-runtime-evaluation-bundle-v1"
_DERIVED_IDENTITY = (
    "item_root_name",
    "proposal_hash",
    "accepted_contract_sha256",
    "harbor_tree_sha256",
    "bundle_tree_sha256",
    "specification_sha256",
    "binding_sha256",
    "controls_sha256",
    "runtime_evidence_sha256",
)
_RESET_CONTROLLER = (
    "capability_pipeline/reset_runner.py",
    "capability_pipeline/reset_diagnostics.py",
    "capability_pipeline/daytona_environment.py",
    "capability_pipeline/daytona_resources.py",
    "capability_pipeline/daytona_snapshot.py",
    "capability_pipeline/image_runtime_metadata.py",
    "capability_pipeline/provider_retry.py",
    "capability_pipeline/runtime.py",
    "capability_pipeline/synthesis.py",
)
_COMPOSITE_CONTROLLER = (
    "capability_pipeline/composite_grading_diagnostics.py",
    "capability_pipeline/fixed_grading_capture.py",
    "capability_pipeline/daytona_verifier.py",
    "capability_pipeline/daytona_resources.py",
    "capability_pipeline/composite_verifier.py",
    "capability_pipeline/composite_policy.py",
    "capability_pipeline/runtime.py",
    "capability_pipeline/diagnostics.py",
    "capability_pipeline/evaluation.py",
)


class _Missing(Exception):
    """A decision input the snapshot lists but does not retain intact."""


class _Unknown(Exception):
    """The old code's behaviour cannot be decided from the retained evidence."""


class _Impossible(Exception):
    """Under this candidate environment the old code could not have reviewed."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parts(relative: str) -> tuple[str, ...]:
    # The old code sorted pathlib paths, which compare component-wise.
    return PurePosixPath(relative).parts


class _Snapshot:
    """Read-only, digest-checked view of one quality review input snapshot."""

    def __init__(self, review_root: Path) -> None:
        self.input = review_root / "input"
        try:
            manifest = json.loads((review_root / "input-manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise _Missing(f"input-manifest.json is unreadable: {error}") from error
        files, item_files = manifest.get("files"), manifest.get("item_files")
        if not isinstance(files, dict) or not isinstance(item_files, list):
            raise _Missing("input-manifest.json lacks files/item_files")
        identity = {key: value for key, value in manifest.items() if key != "snapshot_hash"}
        if manifest.get("snapshot_hash") != digest(identity):
            raise _Missing("input-manifest.json does not match its own snapshot_hash")
        if any(not isinstance(name, str) or name not in files for name in item_files):
            raise _Missing("input-manifest.json item_files are not all hashed")
        self.files: dict[str, str] = files
        self.item: set[str] = set(item_files)
        self._cache: dict[str, bytes] = {}

    def listed(self, relative: str) -> bool:
        """Whether the item held this file when the review snapshot was taken."""
        return relative in self.item

    def under(self, prefix: str) -> list[str]:
        return sorted((name for name in self.item if name.startswith(prefix)), key=_parts)

    def data(self, relative: str) -> bytes:
        if relative in self._cache:
            return self._cache[relative]
        if relative not in self.files:
            raise _Missing(f"{relative} is not in the review snapshot")
        try:
            raw = (self.input / relative).read_bytes()
        except OSError as error:
            raise _Missing(f"{relative} is listed but unreadable") from error
        if _sha256(raw) != self.files[relative]:
            raise _Missing(f"{relative} bytes differ from the review manifest")
        self._cache[relative] = raw
        return raw

    def json(self, relative: str) -> Any:
        """Parsed JSON; ``ValueError`` if the retained bytes are not JSON."""
        return json.loads(self.data(relative))


# ---------------------------------------------------------------- identities


def _tree_sha256(snapshot: _Snapshot, prefix: str) -> str:
    """runtime.tree_sha256 (evaluation.input_hash of a directory), from digests."""
    value = hashlib.sha256()
    for name in snapshot.under(prefix):
        value.update(name[len(prefix):].encode())
        value.update(b"\0" + snapshot.files[name].encode() + b"\n")
    return value.hexdigest()


def _file_tree_sha256(snapshot: _Snapshot, prefix: str) -> str:
    """reset_runner._file_tree_sha256: [path, size, sha256] for every file."""
    entries = [
        [name[len(prefix):], len(snapshot.data(name)), snapshot.files[name]]
        for name in snapshot.under(prefix)
    ]
    return _sha256(json.dumps(entries, separators=(",", ":")).encode())


def _derived_identity(snapshot: _Snapshot, item_name: str) -> tuple[dict, dict, dict]:
    """The parts of diagnostics._item_identity recomputable from the snapshot."""
    for name in (
        "contract/accepted.json", "harbor/manifest.json", "workspace/task/specification.json",
        "workspace/task/binding.json", "workspace/task/controls.json",
    ):
        if not snapshot.listed(name):
            raise _Unknown(f"{name} was absent, so diagnostics could not have run")
    accepted = snapshot.json("contract/accepted.json")
    derived = {
        "item_root_name": item_name,
        "proposal_hash": accepted.get("proposal_hash"),
        "accepted_contract_sha256": snapshot.files["contract/accepted.json"],
        "harbor_tree_sha256": _tree_sha256(snapshot, "harbor/"),
        "bundle_tree_sha256": _tree_sha256(snapshot, "workspace/task/"),
        "specification_sha256": snapshot.files["workspace/task/specification.json"],
        "binding_sha256": snapshot.files["workspace/task/binding.json"],
        "controls_sha256": snapshot.files["workspace/task/controls.json"],
        "runtime_evidence_sha256": snapshot.files.get("runtime-evidence.json")
        if snapshot.listed("runtime-evidence.json") else None,
    }
    resources = {"provided": False}
    if snapshot.listed("workspace/task/candidate-resources.json"):
        binding = snapshot.json("workspace/task/binding.json")
        if binding["environment"]["kind"] == "docker":
            resources = {
                "provided": True,
                "sha256": snapshot.files["workspace/task/candidate-resources.json"],
            }
    return accepted, derived, resources


def _derived_match(identity: Any, derived: dict, resources: dict) -> bool:
    if not isinstance(identity, dict):
        return False
    invocation = identity.get("invocation")
    if not isinstance(invocation, dict):
        return False
    gate = invocation.get("primary_adversarial_gate")
    return (
        all(identity.get(key) == derived[key] for key in _DERIVED_IDENTITY)
        and invocation.get("candidate_resources") == resources
        and isinstance(gate, dict)
        and gate.get("runtime_evidence_sha256") == derived["runtime_evidence_sha256"]
    )


def _attempt_name_key(name: str) -> tuple[str, ...]:
    return _parts(name)


# ------------------------------------------------------------------- stages


def _repeated(snapshot: _Snapshot, attempt: str) -> tuple[str, bool, dict]:
    """(state, reviewable) run_repeated_diagnostics returned for this attempt."""
    base = f"diagnostics/{attempt}"
    try:
        recorded = snapshot.json(f"{base}/result.json")
    except ValueError as error:
        raise _Unknown(f"{base}/result.json is not JSON") from error
    if not isinstance(recorded, dict):
        raise _Unknown(f"{base}/result.json is not an object")
    recorded_pair = (recorded.get("state"), recorded.get("reviewable") is True)
    if not snapshot.listed(f"{base}/evaluation/matrix.json"):
        if recorded_pair[0] == "ready" or recorded_pair[1]:
            raise _Unknown(f"{base} records a reviewable result without an evaluation matrix")
        raise _Impossible(f"{base} has no evaluation matrix (pending, not reviewable)")
    matrix = snapshot.json(f"{base}/evaluation/matrix.json")
    cells = matrix.get("cells") if isinstance(matrix, dict) else None
    reviewable = (
        isinstance(matrix, dict)
        and matrix.get("complete_attempt_inventory") is True
        and isinstance(cells, list)
        and len(cells) == 3
        and all(isinstance(cell, dict) and cell.get("state") == "valid" for cell in cells)
    )
    ready = reviewable and matrix.get("state") == "repeated_runtime_passed"
    derived_pair = ("ready" if ready else "pending", reviewable)
    if recorded_pair != derived_pair:
        # A reused attempt returns the matrix-derived value; a fresh one returns
        # result.json.  Which happened is not recorded.
        raise _Unknown(
            f"{base}/result.json {recorded_pair} disagrees with its matrix {derived_pair}"
        )
    detail = {
        "attempt": attempt,
        "state": derived_pair[0],
        "reviewable": reviewable,
        "evaluation_state": matrix.get("state") if isinstance(matrix, dict) else None,
    }
    return derived_pair[0], reviewable, detail


def _unsupported_reason(snapshot: _Snapshot) -> str | None:
    """grading_diagnostics._unsupported on the frozen bundle (== workspace/task)."""
    from . import regrade

    for name in ("binding.json", "specification.json", "controls.json", "composite-verifier.json"):
        if snapshot.listed(f"workspace/task/{name}"):
            snapshot.data(f"workspace/task/{name}")  # digest-check what is read
    try:
        _, _, kind = regrade._runtime_fields(snapshot.input / "workspace/task")
        regrade.validate_controls(snapshot.json("workspace/task/controls.json"), binding_kind=kind)
    except _Missing:
        raise
    except Exception as error:  # noqa: BLE001 - mirrors the old catch-all.
        return str(error)
    return None


def _grading(
    snapshot: _Snapshot, accepted: dict, attempt: str, env: dict
) -> tuple[str, bool, bool, dict]:
    """(state, reviewable, unassessed) of the fixed-grading stage."""
    judge = accepted["proposal"]["verification"] == "judge"
    composed = snapshot.listed("workspace/task/composite-verifier.json")
    if judge and not composed:
        return "not_applicable", False, False, {"kind": "judge_not_applicable"}
    if judge:
        return _composite_grading(snapshot, attempt, env)
    reason = _unsupported_reason(snapshot)
    base = f"diagnostics/{attempt}/fixed-grading"
    if reason is not None:
        if snapshot.under(f"{base}/"):
            raise _Unknown(f"{base} exists although the bundle is unsupported: {reason}")
        return "unsupported", False, True, {"kind": "unsupported", "reason": reason}
    if not snapshot.listed(f"{base}/grading-diagnostics.json"):
        raise _Impossible(f"{base}/grading-diagnostics.json is absent (pending)")
    binding = snapshot.json(f"{base}/binding.json") if snapshot.listed(f"{base}/binding.json") else None
    if not isinstance(binding, dict):
        raise _Impossible(f"{base}/binding.json is absent (pending)")
    if (
        binding.get("controller") != env["controller_files"]
        or binding.get("timeout_seconds") != env["timeout"]
    ):
        raise _Impossible(f"{base}/binding.json was frozen by another controller (drift -> pending)")
    recorded = snapshot.json(f"{base}/grading-diagnostics.json")
    state = recorded.get("state") if isinstance(recorded, dict) else None
    if state not in {"ready", "semantic_failed"}:
        raise _Impossible(f"{base} recorded {state!r} (not reviewable)")
    return state, True, False, {"kind": "fixed_grading", "state": state}


def _composite_grading(
    snapshot: _Snapshot, attempt: str, env: dict
) -> tuple[str, bool, bool, dict]:
    matrix_sha = snapshot.files.get(f"diagnostics/{attempt}/evaluation/matrix.json")
    manifest = snapshot.json(f"diagnostics/{attempt}/inputs/manifest.json")
    candidates = []
    for name in snapshot.under("diagnostics/composite-grading/"):
        parts = _parts(name)
        if len(parts) != 4 or parts[3] != "binding.json":
            continue
        binding = snapshot.json(name)
        source = binding.get("source") if isinstance(binding, dict) else None
        if not isinstance(source, dict):
            continue
        key = hashlib.sha256(
            json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        if (
            parts[2] == f"attempt-{key}"
            and source.get("repeated_matrix_sha256") == matrix_sha
            and source.get("repeated_plan_sha256") == manifest.get("plan_sha256")
            and source.get("current_controls_sha256") == snapshot.files.get("workspace/task/controls.json")
            and source.get("current_harbor_manifest_sha256") == snapshot.files.get("harbor/manifest.json")
            and source.get("daytona_helper_sha256") == env["helper_sha256"]
        ):
            candidates.append((parts[2], binding))
    if len(candidates) != 1:
        raise _Unknown(f"{len(candidates)} composite-grading attempts match repeated {attempt}")
    directory, binding = candidates[0]
    expected_controller = {name: env["controller_files"].get(name) for name in _COMPOSITE_CONTROLLER}
    if (
        binding.get("controller") != expected_controller
        or binding.get("timeout_seconds") != env["timeout"]
        or binding.get("taskcompendium_source_lock_sha256") != env["source_lock_sha256"]
    ):
        raise _Impossible(f"composite-grading/{directory} was frozen by another controller")
    summary_name = f"diagnostics/composite-grading/{directory}/summary.json"
    if not snapshot.listed(summary_name):
        raise _Impossible(f"{summary_name} is absent (pending)")
    state = snapshot.json(summary_name).get("state")
    if state not in {"ready", "semantic_failed"}:
        raise _Impossible(f"{summary_name} recorded {state!r} (not reviewable)")
    return state, True, False, {"kind": "composite_grading", "attempt": directory, "state": state}


def _docker_reset(snapshot: _Snapshot, env: dict) -> tuple[str, bool, dict]:
    from .daytona_resources import DaytonaResourceProfile

    resources = "workspace/task/candidate-resources.json"
    policy = "workspace/task/reset-policy.json"

    def authoring(reason: str) -> tuple[str, bool, dict]:
        return "semantic_failed", True, {"kind": "docker_reset", "state": "semantic_failed", "reason": reason}

    if not snapshot.listed(resources):
        return authoring("candidate-resources.json is required")
    try:
        request = snapshot.json(resources)
    except ValueError:
        return authoring("candidate-resources.json is invalid")
    if not isinstance(request, dict):
        return authoring("candidate-resources.json is invalid")
    if set(request) != {"cpu", "memory_gb", "disk_gb"}:
        return authoring("candidate-resources.json must contain cpu, memory_gb, disk_gb")
    try:
        DaytonaResourceProfile(
            cpu=request["cpu"], memory_gb=request["memory_gb"], disk_gb=request["disk_gb"],
            source="adapter_kwargs",
        )
    except ValueError:
        return authoring("candidate-resources.json has invalid capacity")
    if not snapshot.listed(policy):
        return authoring("reset-policy.json is required")
    try:
        if not isinstance(snapshot.json(policy), dict):
            return authoring("reset-policy.json is invalid")
    except ValueError:
        return authoring("reset-policy.json is invalid")
    if env["helper_sha256"] is None:
        raise _Impossible("no pinned Daytona helper (docker reset input -> pending)")
    source = {
        "harbor_file_sha256": _file_tree_sha256(snapshot, "harbor/"),
        "policy_sha256": snapshot.files[policy],
        "candidate_resources_sha256": snapshot.files[resources],
        "daytona_helper_sha256": env["helper_sha256"],
    }
    key = _sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode())[:20]
    base = f"diagnostics/reset/attempt-{key}"
    if not snapshot.listed(f"{base}/binding.json"):
        raise _Impossible(f"{base}/binding.json is absent (freeze failed -> pending)")
    binding = snapshot.json(f"{base}/binding.json")
    frozen = binding.get("source") if isinstance(binding, dict) else None
    if not isinstance(frozen, dict) or {k: v for k, v in frozen.items() if k != "harbor_tree_sha256"} != source:
        raise _Impossible(f"{base}/binding.json source differs (-> pending)")
    expected_controller = {name: env["controller_files"].get(name) for name in _RESET_CONTROLLER}
    frozen_controller = binding.get("controller")
    if (
        not isinstance(frozen_controller, dict)
        or {name: frozen_controller.get(name) for name in _RESET_CONTROLLER} != expected_controller
        or binding.get("timeout_seconds") != env["timeout"]
        or binding.get("taskcompendium_source_lock_sha256") != env["source_lock_sha256"]
    ):
        raise _Impossible(f"{base} was frozen by another controller (drift -> pending)")
    summary = f"{base}/summary.json"
    if not snapshot.listed(summary):
        # Either an authoring rejection after freeze (semantic_failed) or a
        # pending runner; neither is ready, and which one is not retained.
        raise _Unknown(f"{summary} is absent (reset not ready; reviewability unknown)")
    state = snapshot.json(summary).get("state")
    if state == "ready":
        return "ready", False, {"kind": "docker_reset", "attempt": f"attempt-{key}", "state": state}
    if state == "semantic_failed":
        return "semantic_failed", True, {"kind": "docker_reset", "attempt": f"attempt-{key}", "state": state}
    raise _Impossible(f"{summary} recorded {state!r} (not reviewable)")


def _non_docker_reset(snapshot: _Snapshot, env: dict) -> tuple[str, bool, dict]:
    binding = snapshot.json("harbor/binding.json")
    environment = binding.get("environment") if isinstance(binding, dict) else None
    kind = environment.get("kind") if isinstance(environment, dict) else None
    if kind not in {"none", "shellsim"}:
        raise _Impossible("non-Docker reset requires a none or shellsim binding (-> pending)")
    manifest = snapshot.json("harbor/manifest.json")
    steps = manifest.get("step_names") if isinstance(manifest, dict) else None
    if not isinstance(steps, list) or not steps or not all(isinstance(step, str) for step in steps):
        raise _Impossible("Harbor manifest has invalid step names (-> pending)")
    prompts = [
        f"harbor/steps/{name}/instruction.md" if len(steps) > 1 else "harbor/instruction.md"
        for name in steps
    ]
    for name in (*prompts, "harbor/task.toml", "harbor/specification.json", "harbor/renderings.json"):
        if not snapshot.listed(name):
            raise _Impossible(f"{name} is absent (non-Docker reset input -> pending)")
    portable = {
        "harbor_file_sha256": _file_tree_sha256(snapshot, "harbor/"),
        "binding_sha256": snapshot.files["harbor/binding.json"],
        "manifest_sha256": snapshot.files["harbor/manifest.json"],
        "task_toml_sha256": snapshot.files["harbor/task.toml"],
        "specification_sha256": snapshot.files["harbor/specification.json"],
        "renderings_sha256": snapshot.files["harbor/renderings.json"],
        "prompt_sha256": [snapshot.files[name] for name in prompts],
        "step_names": steps,
        "kind": kind,
    }
    matches = []
    for name in snapshot.under("diagnostics/non-docker-reset/"):
        parts = _parts(name)
        if len(parts) != 4 or parts[3] != "binding.json":
            continue
        frozen = snapshot.json(name).get("source")
        if not isinstance(frozen, dict):
            continue
        frozen_portable = {key: value for key, value in frozen.items() if key != "harbor_tree_sha256"}
        key = _sha256(json.dumps(frozen_portable, sort_keys=True).encode())[:20]
        if parts[2] != f"attempt-{key}":
            continue
        core = {key: value for key, value in frozen_portable.items()
                if key not in {"shellsim_bridge_sha256", "shellsim_overlay"}}
        if core != portable:
            continue
        if kind == "shellsim" and frozen_portable.get("shellsim_bridge_sha256") != env["bridge_sha256"]:
            continue
        matches.append(parts[2])
    if kind == "shellsim" and env["bridge_sha256"] is None:
        raise _Impossible("no pinned ShellSim bridge (non-Docker reset input -> pending)")
    if not matches:
        raise _Impossible("no frozen non-Docker reset attempt for the reviewed Harbor bytes (-> pending)")
    if len(matches) > 1:
        # Only the ShellSim overlay record (a controller constant) could differ.
        raise _Unknown(f"{len(matches)} non-Docker reset attempts match the reviewed bytes")
    summary = f"diagnostics/non-docker-reset/{matches[0]}/summary.json"
    if not snapshot.listed(summary):
        raise _Impossible(f"{summary} is absent (pending)")
    state = snapshot.json(summary).get("state")
    detail = {"kind": "non_docker_reset", "attempt": matches[0], "state": state}
    if state == "ready":
        return "ready", False, detail
    if state == "semantic_failed":
        return "semantic_failed", True, detail
    raise _Impossible(f"{summary} recorded {state!r} (not reviewable)")


# ---------------------------------------------------------------- decision


def _environment(identity: dict, controller: dict) -> dict:
    invocation = identity["invocation"]

    def sha(record: Any) -> str | None:
        if isinstance(record, dict) and record.get("provided") is True:
            return record.get("sha256")
        return None

    toolchain = identity.get("toolchain_source")
    return {
        "controller_files": controller.get("evaluation_controller_files") or {},
        "timeout": invocation.get("validation_timeout"),
        "helper_sha256": sha(invocation.get("daytona_helper")),
        "bridge_sha256": sha(invocation.get("shellsim_bridge")),
        "source_lock_sha256": toolchain.get("source_lock_sha256") if isinstance(toolchain, dict) else None,
    }


def _compose(snapshot: _Snapshot, accepted: dict, attempt: str, env: dict) -> dict:
    """_repeated_quality_diagnostics' (state, reviewable) for one environment."""
    state, reviewable, repeated = _repeated(snapshot, attempt)
    detail: dict[str, Any] = {"repeated": repeated}
    if state != "ready" and not reviewable:
        raise _Impossible(f"repeated {attempt} is pending and not reviewable")
    g_state, g_reviewable, g_unassessed, detail["grading"] = _grading(snapshot, accepted, attempt, env)
    if g_state not in {"not_applicable", "ready"} and not (g_state == "unsupported" and g_unassessed):
        reviewable = reviewable and g_reviewable and g_state == "semantic_failed"
        state = "pending"
    if state != "ready" and not reviewable:
        raise _Impossible("repeat and fixed-grading diagnostics are not reviewable")
    if accepted["proposal"].get("environment") == "container":
        r_state, r_reviewable, detail["reset"] = _docker_reset(snapshot, env)
    else:
        r_state, r_reviewable, detail["reset"] = _non_docker_reset(snapshot, env)
    if r_state != "ready":
        reviewable = r_state == "semantic_failed" and r_reviewable
        state = "pending"
    if state != "ready" and not reviewable:
        raise _Impossible("diagnostics are not reviewable after reset")
    detail["state"], detail["reviewable"] = state, reviewable
    return detail


def reconstruct(review_root: Path) -> dict[str, Any]:
    """Recompute the old post-review gate for ``quality/<item>/attempt-N``.

    Returns ``{"schema_version", "verdict": ready|not_ready|unknown, "reason",
    "candidates": [...], "eliminated": [...]}``.  Never raises for evidence
    problems; those are ``unknown``.
    """
    review_root = Path(review_root)
    item_name = review_root.parent.name
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "review_dir": f"quality/{item_name}/{review_root.name}",
        "verdict": UNKNOWN,
        "reason": "",
        "candidates": [],
        "eliminated": [],
    }
    try:
        snapshot = _Snapshot(review_root)
        accepted, derived, resources = _derived_identity(snapshot, item_name)
        if not isinstance(accepted.get("proposal"), dict) or "verification" not in accepted["proposal"]:
            raise _Unknown("contract/accepted.json lacks proposal.verification")
        names = sorted(
            {_parts(name)[1] for name in snapshot.under("diagnostics/")
             if len(_parts(name)) > 2 and _parts(name)[1].startswith("attempt-")},
            key=_attempt_name_key,
        )
        records = []
        for name in names:
            record: dict[str, Any] = {"name": name, "result": snapshot.listed(f"diagnostics/{name}/result.json")}
            manifest_name = f"diagnostics/{name}/inputs/manifest.json"
            if not snapshot.listed(manifest_name):
                record["manifest"] = "unreadable"
            else:
                try:
                    manifest = snapshot.json(manifest_name)
                except ValueError:
                    manifest = "unreadable"
                record["manifest"] = manifest
                if isinstance(manifest, dict):
                    identity, controller = manifest.get("item_identity"), manifest.get("controller")
                    record["schema"] = manifest.get("schema_version") == _DIAGNOSTICS_SCHEMA
                    record["env_key"] = digest({"item_identity": identity, "controller": controller})
                    record["derived"] = _derived_match(identity, derived, resources) and isinstance(controller, dict)
            records.append(record)
        environments: list[str] = []
        for record in records:
            if record["result"] and record.get("schema") and record.get("derived") and record["env_key"] not in environments:
                environments.append(record["env_key"])
        if not environments:
            raise _Unknown("no retained repeated-diagnostics attempt matches the reviewed task bytes")
        for env_key in environments:
            entry: dict[str, Any] = {"env": env_key[:16]}
            try:
                chosen = None
                for record in records:
                    if not record["result"]:
                        continue
                    if record["manifest"] == "unreadable":
                        raise _Impossible(f"diagnostics/{record['name']} metadata is unreadable (-> pending)")
                    if not isinstance(record["manifest"], dict):
                        raise _Unknown(f"diagnostics/{record['name']} manifest is not an object")
                    if record.get("schema") and record.get("env_key") == env_key:
                        chosen = record
                        break
                if chosen is None:
                    raise _Unknown("simulated attempt selection found no attempt")
                manifest = chosen["manifest"]
                env = _environment(manifest["item_identity"], manifest["controller"])
                entry["attempt"] = chosen["name"]
                entry.update(_compose(snapshot, accepted, chosen["name"], env))
                entry["outcome"] = READY if entry["state"] == "ready" else NOT_READY
                result["candidates"].append(entry)
            except _Impossible as error:
                entry["outcome"], entry["why"] = "impossible", str(error)
                result["eliminated"].append(entry)
        outcomes = {entry["outcome"] for entry in result["candidates"]}
        if not outcomes:
            raise _Unknown("no candidate environment can have reached review: "
                           + "; ".join(entry["why"] for entry in result["eliminated"]))
        if outcomes == {READY}:
            result["verdict"] = READY
            result["reason"] = "repeated diagnostics were ready when the review ran"
        elif outcomes == {NOT_READY}:
            result["verdict"] = NOT_READY
            result["reason"] = "review ran on reviewable but not ready diagnostics: " + "; ".join(
                _not_ready_reason(entry) for entry in result["candidates"]
            )
        else:
            raise _Unknown("candidate review-time environments disagree")
    except _Missing as error:
        result["verdict"], result["reason"] = UNKNOWN, f"missing evidence: {error}"
    except _Unknown as error:
        result["verdict"], result["reason"] = UNKNOWN, str(error)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        result["verdict"], result["reason"] = UNKNOWN, f"unreadable evidence: {type(error).__name__}: {error}"
    return result


def _not_ready_reason(entry: dict) -> str:
    reasons = []
    repeated = entry.get("repeated", {})
    if repeated.get("state") != "ready":
        reasons.append(f"repeated {repeated.get('attempt')} evaluation {repeated.get('evaluation_state')}")
    grading = entry.get("grading", {})
    if grading.get("state") == "semantic_failed":
        reasons.append(f"{grading.get('kind')} semantic_failed")
    reset = entry.get("reset", {})
    if reset.get("state") not in (None, "ready"):
        reasons.append(f"{reset.get('kind')} {reset.get('state')}")
    return ", ".join(reasons) or "not ready"

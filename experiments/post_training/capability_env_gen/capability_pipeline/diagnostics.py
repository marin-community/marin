"""Immutable repeated-runtime diagnostic orchestration.

This module stages an already lowered task into a new, self-contained
three-attempt evaluation input.  It deliberately does not decide task quality,
modify synthesis state, or reserve a repair round.  Existing evaluation code
executes all task/generated code through its remote runtime command.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from argparse import Namespace
from pathlib import Path

from . import evaluation

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_LOCK = _PROJECT_ROOT / "vendor" / "task_spec" / "source.lock.json"
_SCHEMA = "capability-runtime-evaluation-bundle-v1"
_RESULT_SCHEMA = "capability-repeated-diagnostics-result-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")


def _copy_tree(source: Path, destination: Path) -> None:
    evaluation.input_hash(source)
    shutil.copytree(source, destination)


def _copy_file(source: Path, destination: Path) -> None:
    evaluation.input_hash(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _toolchain_identity(toolchain_source: Path) -> dict[str, object]:
    """Bind only the maintained lock-listed source, never a local .venv/cache."""
    try:
        lock = json.loads(_SOURCE_LOCK.read_text())
        files = lock["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("pinned TaskCompendium source lock is unavailable") from error
    if not isinstance(files, dict) or not toolchain_source.is_dir():
        raise ValueError("diagnostics TaskCompendium source is unavailable")
    actual: dict[str, str] = {}
    for relative, expected in sorted(files.items()):
        path = toolchain_source / relative
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or not path.is_file()
        ):
            raise ValueError("pinned TaskCompendium source file is unavailable")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError("pinned TaskCompendium source lock differs")
        actual[relative] = observed
    overlay = {
        "composite_extension_lock": _sha256(
            _PROJECT_ROOT / "vendor" / "task_spec" / "composite_extension.lock.json"
        ),
        "composite_extension_patch": _sha256(
            _PROJECT_ROOT
            / "vendor"
            / "task_spec"
            / "patches"
            / "composite_required_extension.patch"
        ),
    }
    return {
        "source_lock_sha256": _sha256(_SOURCE_LOCK),
        "revision": lock.get("revision"),
        "package_root": lock.get("package_root"),
        "files": actual,
        "overlay": overlay,
    }


def _invocation_identity(
    validation_timeout: int,
    daytona_helper: Path | None,
    shellsim_bridge: Path | None,
    candidate_resources: Path | None,
) -> dict[str, object]:
    def file_identity(path: Path | None, label: str) -> dict[str, object]:
        if path is None:
            return {"provided": False}
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"diagnostics {label} is unavailable")
        return {"provided": True, "sha256": _sha256(path)}

    return {
        "validation_timeout": validation_timeout,
        "daytona_helper": file_identity(daytona_helper, "Daytona helper"),
        "shellsim_bridge": file_identity(shellsim_bridge, "ShellSim bridge"),
        "candidate_resources": file_identity(
            candidate_resources, "candidate resources"
        ),
    }


def _item_identity(
    item_root: Path,
    toolchain_source: Path,
    invocation: dict[str, object],
) -> dict[str, object]:
    accepted = item_root / "contract" / "accepted.json"
    try:
        accepted_data = json.loads(accepted.read_text())
    except (OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "diagnostics requires immutable contract/accepted.json"
        ) from error
    if not isinstance(accepted_data, dict) or not isinstance(
        accepted_data.get("proposal_hash"), str
    ):
        raise TypeError("accepted contract lacks proposal identity")
    harbor = item_root / "harbor"
    bundle = item_root / "workspace" / "task"
    required = (
        harbor / "manifest.json",
        bundle / "specification.json",
        bundle / "binding.json",
        bundle / "controls.json",
    )
    if any(not path.is_file() for path in required):
        raise ValueError("diagnostics requires lowered Harbor package and task bundle")
    return {
        "item_root_name": item_root.name,
        "proposal_hash": accepted_data["proposal_hash"],
        "accepted_contract_sha256": _sha256(accepted),
        "harbor_tree_sha256": evaluation.input_hash(harbor),
        "bundle_tree_sha256": evaluation.input_hash(bundle),
        "toolchain_source": _toolchain_identity(toolchain_source),
        "invocation": invocation,
        "specification_sha256": _sha256(bundle / "specification.json"),
        "binding_sha256": _sha256(bundle / "binding.json"),
        "controls_sha256": _sha256(bundle / "controls.json"),
        "runtime_evidence_sha256": (
            _sha256(item_root / "runtime-evidence.json")
            if (item_root / "runtime-evidence.json").is_file()
            else None
        ),
    }


def _controller_identity() -> dict[str, object]:
    return {
        "diagnostics_sha256": _sha256(Path(__file__)),
        "evaluation_controller_files": evaluation.controller_hashes(),
        "evaluation_policy": evaluation.runtime_policy(),
        "evaluation_unassessed_recipe_rows": evaluation.UNASSESSED,
    }


def _validated_primary_adversary(item_root: Path) -> dict[str, object]:
    """Bind repeat measurements to the already reviewed independent attacks."""
    from .attack_adjudication import validate_resolution
    from .synthesis import _attestation_issues

    evidence_path = item_root / "runtime-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    bundle = item_root / "workspace/task"
    issues = _attestation_issues(
        evidence, evidence_path, bundle, item_root / "harbor",
        bundle / "controls.json", _sha256(item_root / "harbor/specification.json"),
    )
    receipt = None
    if issues:
        adjudicable = {
            "attestation lacks an independent adversarial attack",
            "independent adversary report needs adjudication or retry",
        }
        if not set(issues) <= adjudicable:
            raise ValueError("primary runtime has unresolved non-adversarial issues")
        reviews = (
            item_root.parent.parent / "attack-adjudication" / item_root.name
        )
        for review_root in sorted(reviews.glob("attempt-*"), reverse=True):
            try:
                resolved = validate_resolution(item_root, review_root)
            except (OSError, ValueError, TypeError, KeyError):
                continue
            if resolved.get("state") == "resolved":
                receipt = review_root / "result.json"
                break
        if receipt is None:
            raise ValueError("primary rewarded attacks lack resolved GLM adjudication")
    return {
        "schema_version": "capability-primary-adversarial-gate-v1",
        "state": "resolved" if receipt else "passed",
        "runtime_evidence_sha256": _sha256(evidence_path),
        "adjudication_result_sha256": _sha256(receipt) if receipt else None,
        "new_attacks_in_repeated_measurement": False,
    }


def _manifest_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "manifest.json"
    }


def _manifest_directories(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()
    )


def _result_path(attempt: Path) -> Path:
    return attempt / "result.json"


def _public_result(result: dict[str, object]) -> dict[str, object]:
    """Expose quality-packet paths as Paths while persisting JSON safely."""
    exposed = dict(result)
    extra = result.get("extra_files")
    if isinstance(extra, dict):
        exposed["extra_files"] = {
            name: Path(path) if isinstance(path, str) else path
            for name, path in extra.items()
        }
    return exposed


def _evaluation_payload(output: Path) -> dict[str, str]:
    return {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }


def _validate_evaluation_output(
    attempt: Path, plan_path: Path, plan_sha256: str
) -> tuple[dict[str, object], dict[str, str]]:
    """Validate every retained evaluator artifact without trusting wrapper JSON."""
    output = attempt / "evaluation"
    if output.is_symlink() or any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("evaluation payload contains a symlink")
    matrix_path = output / "matrix.json"
    attestation_path = output / "matrix.attestation.json"
    matrix = json.loads(matrix_path.read_text())
    attestation = json.loads(attestation_path.read_text())
    plan, _ = evaluation.validate_plan(plan_path, plan_sha256)
    if (
        attestation.get("plan_sha256") != plan_sha256
        or attestation.get("matrix_sha256") != _sha256(matrix_path)
        or attestation.get("input_and_controller_hashes_unchanged_at_finish")
        is not True
        or attestation.get("initial_inputs") != plan["inputs"]
        or attestation.get("initial_controller") != plan["controller_files"]
        or attestation.get("final_controller") != plan["controller_files"]
        or attestation.get("final_input_sha256")
        != {key: value["sha256"] for key, value in plan["inputs"].items()}
    ):
        raise ValueError("evaluation matrix attestation is not bound to frozen inputs")
    expected_numbers = {1, 2, 3}
    cells = matrix.get("cells")
    receipts = attestation.get("attempt_receipts")
    if (
        not isinstance(cells, list)
        or not isinstance(receipts, dict)
        or set(receipts) != {"001", "002", "003"}
    ):
        raise ValueError("evaluation does not retain exactly three attempt receipts")
    by_attempt: dict[int, dict] = {}
    for cell in cells:
        if not isinstance(cell, dict) or type(cell.get("attempt")) is not int:
            raise ValueError("evaluation matrix has an invalid attempt inventory")
        by_attempt[cell["attempt"]] = cell
    if set(by_attempt) != expected_numbers or len(cells) != 3:
        raise ValueError("evaluation matrix attempt IDs are not exactly distinct 1..3")
    loaded_receipts = []
    for number in sorted(expected_numbers):
        root = output / "attempts" / f"{number:03d}"
        receipt_path = root / "receipt.json"
        if _sha256(receipt_path) != receipts[f"{number:03d}"]:
            raise ValueError("evaluation receipt hash differs from attestation")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("attempt") != number:
            raise ValueError("evaluation receipt attempt differs from its path")
        artifact_path = root / "artifacts.manifest.json"
        if receipt.get("artifact_manifest_sha256") != _sha256(artifact_path):
            raise ValueError("evaluation artifact manifest hash differs from receipt")
        if json.loads(artifact_path.read_text()) != evaluation.artifact_manifest(root):
            raise ValueError("evaluation payload differs from its attempt manifest")
        if receipt != by_attempt[number]:
            raise ValueError("evaluation matrix cell differs from retained receipt")
        loaded_receipts.append(receipt)
    recomputed = evaluation.aggregate(plan, loaded_receipts)
    for key in (
        "state",
        "complete_attempt_inventory",
        "gates",
        "counts",
        "reused_sandbox_ids",
        "cells",
    ):
        if matrix.get(key) != recomputed.get(key):
            raise ValueError("evaluation matrix differs from recomputed receipts")
    return matrix, _evaluation_payload(output)


def _result_from_evaluation(
    attempt: Path,
    manifest_path: Path,
    plan_path: Path,
    plan_sha256: str,
    *,
    reused: bool,
) -> dict[str, object]:
    matrix, payload = _validate_evaluation_output(attempt, plan_path, plan_sha256)
    matrix_path = attempt / "evaluation" / "matrix.json"
    attestation_path = attempt / "evaluation" / "matrix.attestation.json"
    reviewable = (
        matrix.get("complete_attempt_inventory") is True
        and isinstance(matrix.get("cells"), list)
        and len(matrix["cells"]) == 3
        and all(
            isinstance(cell, dict) and cell.get("state") == "valid"
            for cell in matrix["cells"]
        )
    )
    ready = matrix.get("state") == "repeated_runtime_passed" and reviewable
    extra_files: dict[str, str] = {
        "controller/diagnostics/input-manifest.json": str(manifest_path),
        "controller/diagnostics/evaluation-plan.json": str(plan_path),
        "controller/diagnostics/matrix.json": str(matrix_path),
        "controller/diagnostics/matrix.attestation.json": str(attestation_path),
    }
    for relative in payload:
        extra_files[f"controller/diagnostics/evaluation/{relative}"] = str(
            attempt / "evaluation" / relative
        )
    return {
        "schema_version": _RESULT_SCHEMA,
        "state": "ready" if ready else "pending",
        "reused": reused,
        "attempt": str(attempt),
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": _sha256(manifest_path),
        "evaluation_plan": str(plan_path),
        "evaluation_plan_sha256": plan_sha256,
        "evaluation_output": str(attempt / "evaluation"),
        "evaluation_matrix": str(matrix_path),
        "evaluation_matrix_sha256": _sha256(matrix_path),
        "evaluation_attestation": str(attestation_path),
        "evaluation_attestation_sha256": _sha256(attestation_path),
        "evaluation_state": matrix.get("state"),
        "evaluation_payload": payload,
        "reviewable": reviewable,
        "unassessed_recipe_rows": evaluation.UNASSESSED,
        "report_artifact": str(matrix_path),
        "report_sha256": _sha256(matrix_path),
        "extra_files": extra_files,
        **(
            {} if ready else {"issues": ["repeated runtime diagnostics are incomplete"]}
        ),
    }


def _validated_completed_attempt(
    attempt: Path, identity: dict[str, object], controller: dict[str, object]
) -> dict[str, object] | None:
    """Reuse only the evaluator evidence, never fields from editable result.json."""
    manifest_path = attempt / "inputs" / "manifest.json"
    if not _result_path(attempt).is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, TypeError, json.JSONDecodeError):
        return {
            "state": "pending",
            "issues": ["retained diagnostic metadata is unreadable"],
            "attempt": str(attempt),
            "reviewable": False,
        }
    if (
        manifest.get("schema_version") != _SCHEMA
        or manifest.get("item_identity") != identity
        or manifest.get("controller") != controller
    ):
        return None
    try:
        from scripts import restore_bundle

        restore_bundle.evaluation_members(attempt / "inputs")
        plan_path = attempt / "inputs" / "plan.json"
        return _result_from_evaluation(
            attempt,
            manifest_path,
            plan_path,
            _sha256(plan_path),
            reused=True,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {
            "state": "pending",
            "issues": ["retained diagnostic evidence failed validation"],
            "attempt": str(attempt),
            "reviewable": False,
        }


def _next_attempt(root: Path) -> Path:
    number = 1
    while (root / f"attempt-{number}").exists():
        number += 1
    return root / f"attempt-{number}"


def _stage_inputs(
    attempt: Path,
    identity: dict[str, object],
    controller: dict[str, object],
    *,
    item_root: Path,
    validation_timeout: int,
    daytona_helper: Path | None,
    shellsim_bridge: Path | None,
    candidate_resources: Path | None,
    primary_gate: dict[str, object] | None,
) -> tuple[Path, str]:
    inputs = attempt / "inputs"
    inputs.mkdir(parents=True)
    _copy_tree(item_root / "harbor", inputs / "package")
    _copy_tree(item_root / "workspace" / "task", inputs / "bundle")
    helper = None
    if daytona_helper is not None:
        if daytona_helper.name != "dt.py":
            raise ValueError("diagnostics Daytona helper must be dt.py")
        helper = inputs / "tools" / "dt.py"
        _copy_file(daytona_helper, helper)
    bridge = None
    if shellsim_bridge is not None:
        bridge = inputs / "shellsim-bridge.json"
        _copy_file(shellsim_bridge, bridge)
    resources = None
    if candidate_resources is not None:
        resources = inputs / "candidate-resources.json"
        _copy_file(candidate_resources, resources)
    primary_gate_path = None
    if primary_gate is not None:
        primary_gate_path = inputs / "primary-adversarial-gate.json"
        _write_new(primary_gate_path, primary_gate)
    plan_path = inputs / "plan.json"
    evaluation.make_plan(
        Namespace(
            package=inputs / "package",
            bundle=inputs / "bundle",
            controls=inputs / "bundle" / "controls.json",
            out=plan_path,
            attempt_timeout=validation_timeout,
            daytona_helper=helper,
            shellsim_bridge=bridge,
            candidate_resources=resources,
            primary_adversarial_gate=primary_gate_path,
        )
    )
    plan_sha256 = _sha256(plan_path)
    manifest = {
        "schema_version": _SCHEMA,
        "plan_sha256": plan_sha256,
        "item_identity": identity,
        "controller": controller,
        "files": _manifest_files(inputs),
        "directories": _manifest_directories(inputs),
    }
    _write_new(inputs / "manifest.json", manifest)
    from scripts import restore_bundle

    restore_bundle.evaluation_members(inputs)
    evaluation.validate_plan(plan_path, plan_sha256)
    return plan_path, plan_sha256


def run_repeated_diagnostics(
    item_root: Path,
    toolchain_source: Path,
    validation_timeout: int,
    daytona_helper: Path | None = None,
    shellsim_bridge: Path | None = None,
    candidate_resources: Path | None = None,
    primary_adversary_bound: bool = False,
) -> dict[str, object]:
    """Run or reuse a frozen repeated evaluation without changing repair state."""
    item_root = item_root.resolve()
    toolchain_source = toolchain_source.resolve()
    if type(validation_timeout) is not int or validation_timeout <= 0:
        raise ValueError("diagnostics validation_timeout must be positive")
    helper = daytona_helper.resolve() if daytona_helper else None
    bridge = shellsim_bridge.resolve() if shellsim_bridge else None
    resources = candidate_resources.resolve() if candidate_resources else None
    invocation = _invocation_identity(validation_timeout, helper, bridge, resources)
    primary_gate = _validated_primary_adversary(item_root) if primary_adversary_bound else None
    invocation["primary_adversarial_gate"] = primary_gate
    identity = _item_identity(item_root, toolchain_source, invocation)
    controller = _controller_identity()
    root = item_root / "diagnostics"
    if root.exists():
        for prior in sorted(root.glob("attempt-*")):
            retained = _validated_completed_attempt(prior, identity, controller)
            if retained is not None:
                return _public_result(retained)
    attempt = _next_attempt(root)
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        plan_path, plan_sha256 = _stage_inputs(
            attempt,
            identity,
            controller,
            item_root=item_root,
            validation_timeout=validation_timeout,
            daytona_helper=helper,
            shellsim_bridge=bridge,
            candidate_resources=resources,
            primary_gate=primary_gate,
        )
        output = attempt / "evaluation"
        evaluation.run_evaluation(
            Namespace(
                plan=plan_path,
                plan_sha256=plan_sha256,
                out=output,
                taskcompendium_source=str(toolchain_source),
                concurrency=3,
            )
        )
        # A controller/input change during remote work is evidence, not a
        # reason to reuse or overwrite the completed execution.
        post_invocation = _invocation_identity(
            validation_timeout, helper, bridge, resources
        )
        post_invocation["primary_adversarial_gate"] = (
            _validated_primary_adversary(item_root) if primary_adversary_bound else None
        )
        if (
            _item_identity(item_root, toolchain_source, post_invocation) != identity
            or _controller_identity() != controller
        ):
            raise ValueError("diagnostic source or controller changed during execution")
        result = _result_from_evaluation(
            attempt,
            attempt / "inputs" / "manifest.json",
            plan_path,
            plan_sha256,
            reused=False,
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        result = {
            "schema_version": _RESULT_SCHEMA,
            "state": "pending",
            "reused": False,
            "attempt": str(attempt),
            "issues": [
                (
                    "repeated diagnostics infrastructure or evidence is incomplete: "
                    f"{type(error).__name__}: {error}"
                )
            ],
            "reviewable": False,
            "unassessed_recipe_rows": evaluation.UNASSESSED,
        }
    _write_new(_result_path(attempt), result)
    return _public_result(result)

"""Frozen fixed-input replay of machine checks from repeated composite controls.

The outer controller consumes retained authored-control Harbor trials.  The inner
controller only invokes the pinned Daytona private grader; it never runs a
candidate, a generated script locally, or the native rubric judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from . import sandbox_provider
from .fixed_grading_capture import load_capture
from .runtime import provider_isolation_record, sha256, verifier_isolation_record

SCHEMA = "capability-composite-fixed-grading-v1"
REPEATS = 10


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or linked artifact: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"object artifact required: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, default=str)
        stream.write("\n")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _controller() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    names = (
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
    return {name: sha256(root / name) for name in names}


def _files(root: Path, *, exclude: set[str] = frozenset()) -> dict[str, str]:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("composite replay artifact contains a link")
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in exclude
    }


def _result(state: str, issues: list[str], attempt: Path | None = None, **fields: Any) -> dict:
    extra = {}
    if attempt is not None and attempt.is_dir():
        extra = {
            f"composite_grading/{attempt.name}/{path.relative_to(attempt).as_posix()}": path
            for path in sorted(attempt.rglob("*")) if path.is_file()
        }
    return {
        "schema_version": SCHEMA,
        "state": state,
        "reviewable": state in {"ready", "semantic_failed"},
        "issues": issues,
        "attempt": str(attempt) if attempt is not None else None,
        "extra_files": extra,
        **fields,
    }


def _expected_machine_delivery(inputs: Path, step: int, check: dict) -> tuple[bytes, bytes]:
    """Rebuild the exact synthetic verifier input from frozen composed source."""
    import msgspec
    from taskcompendium.models import (
        AssistantFinal,
        CodeAnswerVerifier,
        ContainerRuntime,
        FileSubmission,
        FinalState,
        TaskTroveVerifier,
    )
    from taskcompendium.serialization import from_json, renderings_from_json, to_json
    from tasktrove_verify.spec import Mode

    from .daytona_verifier import _embedded_specification

    specification = from_json((inputs / "composite-specification.json").read_bytes())
    renderings = renderings_from_json((inputs / "renderings.json").read_bytes())
    protocol = renderings[step]
    source_verifier = specification.steps[step].verifier
    machine = TaskTroveVerifier(
        Mode.SCRIPT,
        {"path": check["script_path"], "args": check.get("args", [])},
        runtime=ContainerRuntime(
            check["image"], timeout=check["timeout"],
            supervisor_python=check.get("supervisor_python", "python3"),
        ),
    )
    machine_contract = machine
    if isinstance(protocol.submission, AssistantFinal):
        machine_contract = CodeAnswerVerifier(machine, ".composite/candidate.txt")
    elif isinstance(protocol.submission, FileSubmission):
        paths = set(source_verifier.judge.view.files)
        paths.add(protocol.submission.path)
        protocol = msgspec.structs.replace(
            protocol, submission=FinalState(tuple(sorted(paths)))
        )
    synthetic = msgspec.structs.replace(
        specification,
        steps=(
            *specification.steps[:step],
            msgspec.structs.replace(specification.steps[step], verifier=machine_contract),
            *specification.steps[step + 1 :],
        ),
    )
    return to_json(_embedded_specification(synthetic, step)), msgspec.json.encode(protocol)


def _source(item_root: Path, repeated: dict, daytona_tools: Path | None = None) -> tuple[dict, Path, Path]:
    """Bind first full evaluator attempt and current authored task bytes."""
    if not repeated.get("reviewable"):
        raise ValueError("repeated evaluation is not complete and reviewable")
    output = Path(repeated["evaluation_output"])
    if not output.is_dir() or output.is_symlink():
        raise ValueError("repeated evaluation output is absent")
    from . import diagnostics

    attempt = output.parent
    plan = attempt / "inputs/plan.json"
    expected = repeated.get("evaluation_plan_sha256")
    if sha256(plan) != expected:
        raise ValueError("repeated evaluation plan changed")
    diagnostics._validate_evaluation_output(attempt, plan, expected)
    first = output / "attempts/001"
    evidence = first / "runtime-evidence.json"
    controls = attempt / "inputs/bundle/controls.json"
    task_binding = attempt / "inputs/bundle/binding.json"
    task = attempt / "inputs/package"
    resource = item_root / "workspace/task/candidate-resources.json"
    helper = item_root / "workspace/tools/daytona/dt.py"
    if not helper.is_file() and daytona_tools is not None:
        helper = Path(daytona_tools) / "dt.py"
    if helper.is_symlink() or not helper.is_file():
        raise ValueError("pinned Daytona helper is unavailable")
    source = {
        "repeated_plan_sha256": expected,
        "repeated_matrix_sha256": sha256(output / "matrix.json"),
        "first_receipt_sha256": sha256(first / "receipt.json"),
        "first_artifact_manifest_sha256": sha256(first / "artifacts.manifest.json"),
        "first_runtime_evidence_sha256": sha256(evidence),
        "first_authored_oracle_sha256": sha256(first / "authored-oracle.json"),
        "controls_sha256": sha256(controls),
        "task_binding_sha256": sha256(task_binding),
        "current_controls_sha256": sha256(item_root / "workspace/task/controls.json"),
        "current_harbor_manifest_sha256": sha256(item_root / "harbor/manifest.json"),
        "specification_sha256": sha256(task / "composite-specification.json") if (task / "composite-specification.json").is_file() else None,
        "renderings_sha256": sha256(task / "renderings.json"),
        "config_sha256": sha256(task / "composite-verifier.json"),
        "candidate_resources_sha256": sha256(resource) if resource.is_file() else None,
        "daytona_helper_sha256": sha256(helper),
    }
    if source["current_controls_sha256"] != source["controls_sha256"]:
        raise ValueError("current authored controls differ from repeated evaluation")
    for name in ("composite-specification.json", "renderings.json", "composite-verifier.json"):
        if sha256(item_root / "harbor" / name) != sha256(task / name):
            raise ValueError("current Harbor composite package differs from repeated evaluation")
    return source, first, task


def _freeze(attempt: Path, source: dict, first: Path, task: Path, binding: dict, helper: Path | None = None) -> None:
    attempt.mkdir(parents=True, exist_ok=False)
    inputs = attempt / "input"
    inputs.mkdir()
    for name in ("runtime-evidence.json", "authored-oracle.json", "receipt.json", "artifacts.manifest.json"):
        shutil.copy2(first / name, inputs / name)
    for name in ("composite-specification.json", "renderings.json", "composite-verifier.json"):
        if (task / name).is_file():
            shutil.copy2(task / name, inputs / name)
    if source.get("daytona_helper_sha256") is not None:
        if helper is None:
            raise ValueError("Daytona helper source is absent")
        (inputs / "tools").mkdir()
        shutil.copy2(helper, inputs / "tools/dt.py")
    # The raw source trial is retained byte-for-byte, including all capture
    # members and provider receipts.  This is the portable no-resampling input.
    evidence = _read(first / "runtime-evidence.json")
    oracle = _read(first / "authored-oracle.json")
    controls = _read(first.parents[2] / "inputs/bundle/controls.json")
    negative_ids = {case["id"] for case in controls["cases"] if case.get("class") != "positive"}
    positive_ids = {case["id"] for case in controls["cases"] if case.get("class") == "positive"}
    records = [
        {"case_id": row["id"], "step_index": row["step_index"], "grading_artifact": row["artifact"], "grading_sha256": row["artifact_sha256"], "trial_artifact": None, "trial_sha256": None, "result": row["result"], "source": "authored_negative"}
        for row in evidence.get("cases", []) if row.get("id") in negative_ids
    ]
    records += [
        {**row, "source": "authored_positive"}
        for row in oracle.get("cases", []) if row.get("case_id") in positive_ids
    ]
    if len(records) != len(controls["cases"]) or {row["case_id"] for row in records} != positive_ids | negative_ids:
        raise ValueError("first authored-control inventory is incomplete")
    copied = []
    for row in records:
        artifact = row.get("grading_artifact")
        relative_grade = PurePosixPath(artifact) if isinstance(artifact, str) else None
        if relative_grade is None or relative_grade.is_absolute() or relative_grade.parts[0] != "runtime-trials" or any(part in {"", ".", ".."} for part in relative_grade.parts):
            raise ValueError("authored control grading path is invalid")
        source_grade = first / artifact
        if sha256(source_grade) != row.get("grading_sha256"):
            raise ValueError("authored control grade differs from evidence")
        trial = source_grade.parents[1]
        # A multi-step verifier artifact lives under steps/<name>/verifier.
        while trial != first and not (trial / "result.json").is_file():
            trial = trial.parent
        if trial == first:
            raise ValueError("authored control Harbor trial is absent")
        if trial.is_symlink() or any(path.is_symlink() for path in trial.rglob("*")):
            raise ValueError("authored control Harbor trial contains a link")
        relative = trial.relative_to(first)
        if not relative.as_posix().startswith("runtime-trials/"):
            raise ValueError("authored control trial escaped runtime output")
        shutil.copytree(trial, inputs / relative)
        copied.append({"case_id": row["case_id"], "trial_path": relative.as_posix(), "grading_path": artifact, "source": row["source"]})
    _write(inputs / "controls.json", controls)
    shutil.copy2(first.parents[2] / "inputs/bundle/binding.json", inputs / "task-binding.json")
    _write(inputs / "selected-trials.json", {"trials": copied})
    _write(attempt / "binding.json", binding)
    _write(inputs / "files.json", {"files": _files(inputs, exclude={"files.json"})})


def _validate_input(attempt: Path, binding: dict, *, derive: bool = False) -> tuple[list[dict], dict]:
    if _read(attempt / "binding.json") != binding:
        raise ValueError("composite replay binding drifted")
    inputs = attempt / "input"
    frozen = _read(inputs / "files.json")["files"]
    if _files(inputs, exclude={"files.json"}) != frozen:
        raise ValueError("frozen composed input files changed")
    source = binding["source"]
    for name, key in (
        ("runtime-evidence.json", "first_runtime_evidence_sha256"),
        ("authored-oracle.json", "first_authored_oracle_sha256"),
        ("receipt.json", "first_receipt_sha256"),
        ("artifacts.manifest.json", "first_artifact_manifest_sha256"),
        ("controls.json", "controls_sha256"),
        ("task-binding.json", "task_binding_sha256"),
        ("composite-specification.json", "specification_sha256"),
        ("renderings.json", "renderings_sha256"),
        ("composite-verifier.json", "config_sha256"),
    ):
        if sha256(inputs / name) != source[key]:
            raise ValueError("frozen composed source hash changed")
    if source.get("daytona_helper_sha256") is not None and sha256(inputs / "tools/dt.py") != source["daytona_helper_sha256"]:
        raise ValueError("frozen Daytona helper changed")
    evidence = _read(inputs / "runtime-evidence.json")
    oracle = _read(inputs / "authored-oracle.json")
    selected = _read(inputs / "selected-trials.json")["trials"]
    source_manifest = _read(inputs / "artifacts.manifest.json")
    if not binding.get("probe_specific_source"):
        original_files = source_manifest.get("files")
        if not isinstance(original_files, dict):
            raise ValueError("first evaluator artifact inventory is absent")
        for picked in selected:
            relative = picked.get("trial_path")
            safe = PurePosixPath(relative) if isinstance(relative, str) else None
            if (safe is None or safe.as_posix() != relative or safe.is_absolute() or not safe.parts or safe.parts[0] != "runtime-trials"
                    or any(part in {"", ".", ".."} for part in safe.parts)):
                raise ValueError("selected original trial path is invalid")
            trial_tree = inputs / relative
            for path in trial_tree.rglob("*"):
                if not path.is_file():
                    continue
                key = path.relative_to(inputs).as_posix()
                entry = original_files.get(key)
                if (not isinstance(entry, dict) or entry.get("sha256") != sha256(path)
                        or entry.get("bytes") != path.stat().st_size):
                    raise ValueError("copied original trial differs from first evaluator closure")
    controls = _read(inputs / "controls.json")
    task_binding = _read(inputs / "task-binding.json")
    binding_kind = task_binding.get("environment", {}).get("kind") if isinstance(task_binding.get("environment"), dict) else None
    expected_ids = {case["id"] for case in controls["cases"]}
    if len(selected) != len(expected_ids) or {x["case_id"] for x in selected} != expected_ids:
        raise ValueError("frozen authored-control inventory changed")
    records = {
        row["id"]: {"case_id": row["id"], "step_index": row["step_index"], "grading_artifact": row["artifact"], "grading_sha256": row["artifact_sha256"], "result": row["result"], "source": "authored_negative"}
        for row in evidence["cases"] if row.get("id") in expected_ids and row.get("control_type") == "authored_adversarial_control"
    }
    records.update({row["case_id"]: {**row, "source": "authored_positive"} for row in oracle["cases"] if row.get("case_id") in expected_ids})
    if set(records) != expected_ids:
        raise ValueError("original authored-control records are incomplete")
    config = _read(inputs / "composite-verifier.json")
    from .composite_policy import validate_composite_config

    policies = validate_composite_config(
        config,
        specification_sha256=source["specification_sha256"],
        adapter_sha256=sha256(Path(__file__).with_name("composite_verifier.py")),
        policy_sha256=sha256(Path(__file__).with_name("composite_policy.py")),
        step_count=len(config["steps"]),
    )
    cells: list[dict] = []
    candidate_ids: set[str] = set()
    verifier_ids: set[str] = set()
    for picked in selected:
        row = records[picked["case_id"]]
        if picked.get("source") != row["source"]:
            raise ValueError("authored-control record source changed")
        grade_path = inputs / picked["grading_path"]
        relative_grade = PurePosixPath(picked["grading_path"])
        relative_trial = PurePosixPath(picked["trial_path"])
        if any(path.is_absolute() or not path.parts or path.parts[0] != "runtime-trials" or any(part in {"", ".", ".."} for part in path.parts) for path in (relative_grade, relative_trial)) or relative_grade.as_posix() != picked["grading_path"] or relative_trial.as_posix() != picked["trial_path"]:
            raise ValueError("frozen authored-control path is unsafe")
        if picked["grading_path"] != row["grading_artifact"] or sha256(grade_path) != row["grading_sha256"]:
            raise ValueError("original composed grade changed")
        grade = _read(grade_path)
        trial = inputs / picked["trial_path"]
        harbor = _read(trial / "result.json")
        if row.get("trial_sha256") is not None and sha256(trial / "result.json") != row["trial_sha256"]:
            raise ValueError("original authored Harbor trial changed")
        if harbor.get("exception_info") is not None or harbor.get("verifier_result", {}).get("rewards", {}).get("reward") != grade.get("reward"):
            raise ValueError("original composed Harbor result is incomplete")
        if (trial / "daytona-environment.json").is_file():
            provider_isolation_record(trial / "daytona-environment.json", inputs, candidate_ids)
        elif binding_kind == "docker":
            raise ValueError("Docker authored control lacks candidate provider receipt")
        detail = grade.get("detail")
        if not isinstance(detail, dict) or grade.get("status") != "graded" or type(grade.get("reward")) not in (int, float):
            raise ValueError("original composed result was not graded")
        if row.get("result") != grade:
            raise ValueError("original composed grade differs from control report")
        results = detail.get("machine_results")
        captures = detail.get("composite_machine_captures")
        step = row["step_index"]
        checks = policies[step]["machine_checks"]
        if not isinstance(results, list) or not isinstance(captures, list) or len(results) != len(checks) or len(captures) != len(checks):
            raise ValueError("historical composed trial lacks complete machine captures")
        if [x.get("id") for x in results] != [c["id"] for c in checks]:
            raise ValueError("original machine checks differ from config")
        step_root = grade_path.parent.parent
        for index, (check, machine, receipt) in enumerate(zip(checks, results, captures, strict=True)):
            if (receipt.get("check_id"), receipt.get("check_index"), receipt.get("step_index")) != (check["id"], index, step):
                raise ValueError("machine capture identity changed")
            relative = receipt.get("path")
            if relative != f"composite-machine-captures/check-{index:03d}":
                raise ValueError("machine capture path changed")
            capture_root = step_root / "verifier" / relative
            capture = load_capture(capture_root, expected_manifest_sha256=receipt.get("manifest_sha256"))
            if (receipt.get("source_specification_sha256") != source["specification_sha256"] or receipt.get("source_renderings_sha256") != source["renderings_sha256"] or receipt.get("config_sha256") != source["config_sha256"]):
                raise ValueError("machine capture source identity changed")
            manifest = capture["manifest"]
            if derive:
                expected_specification, expected_protocol = _expected_machine_delivery(inputs, step, check)
                if capture["specification"] != expected_specification or capture["protocol"] != expected_protocol:
                    raise ValueError("synthetic machine capture differs from frozen composed source")
            else:
                expected_specification, expected_protocol = capture["specification"], capture["protocol"]
            if (manifest["source_specification_sha256"] != hashlib.sha256(expected_specification).hexdigest() or manifest["source_renderings_sha256"] != hashlib.sha256(expected_protocol).hexdigest()):
                raise ValueError("synthetic machine capture manifest source differs")
            if machine.get("status") != "graded" or type(machine.get("reward")) not in (int, float) or machine.get("detail", {}).get("grading_input_fingerprint") != manifest["fingerprint"]:
                raise ValueError("original machine grade or fingerprint changed")
            isolation = verifier_isolation_record(machine, verifier_ids, candidate_ids, runtime_image=check["image"], supervisor_python=check.get("supervisor_python", "python3"))
            cells.append({
                "case_id": row["case_id"], "step_index": step, "check_index": index,
                "check_id": check["id"], "capture_path": capture_root.relative_to(inputs).as_posix(),
                "capture_manifest_sha256": capture["manifest_sha256"],
                "fingerprint": manifest["fingerprint"], "original_reward": machine["reward"],
                "original_isolation": isolation, "image": check["image"],
                "supervisor_python": check.get("supervisor_python", "python3"),
                "requested_resource_profile": machine["detail"].get("verifier_requested_resource_profile"),
            })
    if candidate_ids & verifier_ids:
        raise ValueError("original controls reused a candidate and private sandbox")
    return cells, {"candidate_ids": candidate_ids, "verifier_ids": verifier_ids}


def _inner(attempt: Path, binding_sha256: str) -> int:
    if os.environ.get("CAPABILITY_REMOTE_COMPOSITE_GRADE") != "1" or not sandbox_provider.credentials_present():
        raise RuntimeError("composite replay execution is remote-only")
    # The pinned dt.py lives in the immutable input tree. Importing it must
    # never create tools/__pycache__ and invalidate that closure mid-replay.
    sys.dont_write_bytecode = True
    if sha256(attempt / "binding.json") != binding_sha256:
        raise ValueError("composite replay binding hash changed")
    binding = _read(attempt / "binding.json")
    from .synthesis import SOURCE_LOCK

    if binding["controller"] != _controller() or binding["taskcompendium_source_lock_sha256"] != sha256(SOURCE_LOCK):
        raise ValueError("composite replay source changed")
    cells, _ = _validate_input(attempt, binding, derive=True)
    os.environ["CAPABILITY_DAYTONA_TOOLS"] = str((attempt / "input/tools").resolve())
    from .daytona_resources import profile_from_receipt
    from .daytona_verifier import grade_captured_in_daytona

    _write(attempt / "derivation-proof.json", {
        "schema_version": SCHEMA, "binding_sha256": binding_sha256,
        "cells_sha256": _digest(cells), "derived_checks": len(cells),
    })
    raw = attempt / "raw"
    raw.mkdir(exist_ok=False)
    for cell_index, cell in enumerate(cells):
        for repeat in range(1, REPEATS + 1):
            destination = raw / f"cell-{cell_index:03d}-repeat-{repeat:02d}.json"
            grade = grade_captured_in_daytona(
                attempt / "input" / cell["capture_path"],
                resource_profile=(
                    profile_from_receipt(cell["requested_resource_profile"])
                    if cell["requested_resource_profile"] is not None else None
                ),
                expected_manifest_sha256=cell["capture_manifest_sha256"],
            )
            _write(destination, {
                "schema_version": SCHEMA, "cell_index": cell_index, "repeat": repeat,
                "cell": cell, "status": grade.status.value, "reward": grade.reward,
                "detail": grade.detail,
            })
    return 0


def _stop(child: subprocess.Popen) -> None:
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def _runner(*, toolchain: Any, attempt: Path, timeout: int) -> int:
    if not sandbox_provider.credentials_present():
        raise RuntimeError("composite replay requires remote Daytona credentials")
    command = toolchain.runtime_command()
    index = command.index("python")
    command[index + 1:] = ["-m", "capability_pipeline.composite_grading_diagnostics", "--inner", "--attempt", str(attempt.resolve()), "--binding-sha256", sha256(attempt / "binding.json")]
    env = dict(os.environ)
    env["CAPABILITY_REMOTE_COMPOSITE_GRADE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if (attempt / "input/tools/dt.py").is_file():
        env["CAPABILITY_DAYTONA_TOOLS"] = str((attempt / "input/tools").resolve())
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    with (attempt / "controller-run.log").open("xb") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        finally:
            _stop(child)


def _classify(attempt: Path, binding: dict) -> tuple[str, list[str], dict]:
    cells, ids = _validate_input(attempt, binding)
    proof_path = attempt / "derivation-proof.json"
    if not proof_path.is_file() or _read(proof_path) != {
        "schema_version": SCHEMA, "binding_sha256": sha256(attempt / "binding.json"),
        "cells_sha256": _digest(cells), "derived_checks": len(cells),
    }:
        return "pending", ["pinned runtime did not prove synthetic machine derivation"], {"expected_checks": len(cells)}
    raw = attempt / "raw"
    expected = {f"cell-{i:03d}-repeat-{r:02d}.json" for i in range(len(cells)) for r in range(1, REPEATS + 1)}
    present = {path.name for path in raw.glob("*.json")} if raw.is_dir() else set()
    if present != expected:
        return "pending", ["composed fixed-input grade matrix is incomplete"], {"expected": len(expected), "observed": len(present)}
    semantic = []
    rows = []
    for index, cell in enumerate(cells):
        for repeat in range(1, REPEATS + 1):
            path = raw / f"cell-{index:03d}-repeat-{repeat:02d}.json"
            row = _read(path)
            if row.get("cell") != cell or row.get("cell_index") != index or row.get("repeat") != repeat:
                raise ValueError("private grade cell identity changed")
            if row.get("status") != "graded" or type(row.get("reward")) not in (int, float):
                return "pending", ["private grade did not complete"], {"expected": len(expected), "observed": len(present)}
            detail = row.get("detail")
            if not isinstance(detail, dict) or detail.get("fixed_grading_capture_manifest_sha256") != cell["capture_manifest_sha256"] or detail.get("grading_input_fingerprint") != cell["fingerprint"]:
                raise ValueError("private grade differs from frozen capture")
            if detail.get("verifier_requested_resource_profile") != cell.get("requested_resource_profile"):
                raise ValueError("private grade requested different verifier capacity")
            verifier_isolation_record(row, ids["verifier_ids"], ids["candidate_ids"], runtime_image=cell["image"], supervisor_python=cell["supervisor_python"])
            if row["reward"] != cell["original_reward"]:
                semantic.append(f"{cell['case_id']}/{cell['check_id']} repeat {repeat}: fixed-input score changed")
            rows.append({"cell_index": index, "repeat": repeat, "reward": row["reward"], "grade_sha256": sha256(path)})
    return ("semantic_failed" if semantic else "ready"), semantic, {"checks": len(cells), "expected": len(expected), "graded": len(rows), "rows": rows}


def _closure(attempt: Path) -> None:
    inventory = _files(attempt, exclude={"artifacts.manifest.json", "summary.json"})
    path = attempt / "artifacts.manifest.json"
    value = {"schema_version": SCHEMA, "binding_sha256": sha256(attempt / "binding.json"), "files": inventory}
    if path.exists():
        if _read(path) != value:
            raise ValueError("composed replay artifact closure changed")
    else:
        _write(path, value)


def run_composite_grading_diagnostics(
    item_root: Path, repeated_result: dict, toolchain: Any, timeout: int,
    daytona_tools: Path | None = None, *, runner: Callable[..., int] | None = None,
) -> dict:
    """Reuse the first authored controls and replay each captured check ten times."""
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("composite replay timeout must be positive")
    item_root = Path(item_root)
    try:
        source, first, task = _source(item_root, repeated_result, daytona_tools)
    except Exception as error:  # noqa: BLE001 - historical captures may be absent.
        return _result("pending", [f"repeated composed source: {type(error).__name__}: {error}"])
    from .synthesis import SOURCE_LOCK

    helper = item_root / "workspace/tools/daytona/dt.py"
    if not helper.is_file() and daytona_tools is not None:
        helper = Path(daytona_tools) / "dt.py"
    binding = {"schema_version": SCHEMA, "source": source, "controller": _controller(), "taskcompendium_source_lock_sha256": sha256(SOURCE_LOCK), "timeout_seconds": timeout, "repeats": REPEATS}
    key = _digest(source)[:20]
    attempt = item_root / "diagnostics/composite-grading" / f"attempt-{key}"
    if not attempt.exists():
        try:
            _freeze(attempt, source, first, task, binding, helper if source.get("daytona_helper_sha256") else None)
        except Exception as error:  # noqa: BLE001 - preserve partial freeze.
            return _result("pending", [f"composed input freeze: {type(error).__name__}: {error}"], attempt)
    try:
        if _source(item_root, repeated_result, daytona_tools)[0] != source:
            raise ValueError("authored composed source drifted")
        cells, _ = _validate_input(attempt, binding)
    except Exception as error:  # noqa: BLE001 - no replay on bad inputs.
        return _result("pending", [f"frozen composed input: {type(error).__name__}: {error}"], attempt)
    marker = attempt / "run-started.json"
    if not marker.exists() and not (attempt / "raw").exists() and not (attempt / "controller-run.log").exists():
        try:
            _write(marker, {"schema_version": SCHEMA, "binding_sha256": sha256(attempt / "binding.json")})
            (runner or _runner)(toolchain=toolchain, attempt=attempt, timeout=timeout)
        except Exception as error:  # noqa: BLE001 - marker prevents resampling.
            issues = [f"composed replay runner: {type(error).__name__}: {error}"]
            try:
                _closure(attempt)
            except Exception as closure_error:  # noqa: BLE001 - retain both failures.
                issues.append(f"partial composed closure: {type(closure_error).__name__}")
            return _result("pending", issues, attempt)
    try:
        if _read(marker) != {"schema_version": SCHEMA, "binding_sha256": sha256(attempt / "binding.json")}:
            raise ValueError("composed replay start marker changed")
        _closure(attempt)
        state, issues, summary = _classify(attempt, binding)
        if _source(item_root, repeated_result, daytona_tools)[0] != source:
            raise ValueError("authored composed source drifted during replay")
    except Exception as error:  # noqa: BLE001 - retain raw forensic evidence.
        return _result("pending", [f"composed replay evidence: {type(error).__name__}: {error}"], attempt)
    summary_path = attempt / "summary.json"
    if not summary_path.exists():
        _write(summary_path, {"schema_version": SCHEMA, "state": state, "issues": issues, "summary": summary})
    return _result(state, issues, attempt, summary=summary, report_artifact=str(summary_path), report_sha256=sha256(summary_path), checks=len(cells), expected_private_grades=len(cells) * REPEATS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner", action="store_true")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--binding-sha256")
    args = parser.parse_args(argv)
    if not args.inner or args.attempt is None or args.binding_sha256 is None:
        parser.error("only frozen remote --inner execution is supported")
    return _inner(args.attempt, args.binding_sha256)


if __name__ == "__main__":
    raise SystemExit(main())

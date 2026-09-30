"""Capture each authored delivery once and repeat private grading remotely."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

from capability_pipeline import sandbox_provider
from capability_pipeline.composite_extension import (
    BASE_VERIFIER_SHA256,
    PATCHED_VERIFIER_SHA256,
)
from capability_pipeline.runtime import (
    HARBOR_REVISION,
    authored_replay_agent_kwargs,
    authored_replay_inventory,
    environment_config,
    expected_extraction_transport,
    load_candidate_resources,
    provider_isolation_record,
    sha256,
    tree_sha256,
    verifier_isolation_record,
)

DEFAULT_REPEATS = 10
DEFAULT_PARALLELISM = 8
REGRADE_BUNDLE_SCHEMA = "capability-fixed-submission-regrade-bundle-v1"
NATIVE_DETERMINISTIC_MODES = frozenset({
    "mcq", "math", "numeric", "exact", "json-schema", "xml-elements", "csv-columns", "ifeval",
})


def _native_deterministic_verifier(verifier, runtime) -> bool:
    """Use the resolved TaskTrove verifier for direct and code-answer shapes."""
    return (
        verifier is not None
        and verifier.mode.value in NATIVE_DETERMINISTIC_MODES
        and runtime is None
        and verifier.judge is None
    )


def validate_controls(controls: dict, *, binding_kind: str = "none") -> list[dict]:
    """Validate the authored, single-step control matrix."""
    cases = controls.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("controls must contain at least one case")
    ids: set[str] = set()
    positives = 0
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("every control must be an object")
        case_id = case.get("id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in ids
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", case_id) is None
        ):
            raise ValueError("control ids must be unique safe strings")
        ids.add(case_id)
        if case.get("class") == "positive":
            positives += 1
        if case.get("step_index", 0) != 0:
            raise ValueError("regrade supports only step_index 0")
        if not isinstance(case.get("response"), str):
            raise TypeError("no-tool controls require a fixed response string")
        if binding_kind == "none" and (case.get("workspace") is not None or case.get("commands") not in (None, [])):
            raise ValueError("no-tool regrade rejects workspace and command replay")
        if binding_kind in {"docker", "shellsim"} and case.get("commands") is not None and (
            not isinstance(case["commands"], list)
            or any(not isinstance(command, str) for command in case["commands"])
        ):
            raise ValueError("container control commands must be strings")
        if "transcript" in case:
            raise ValueError("transcript replay is unsupported")
        expect = case.get("expect")
        if not isinstance(expect, dict) or expect.get("status") not in ("graded", "extraction_error"):
            raise ValueError("every control must declare graded or extraction_error expectations")
        if expect["status"] == "extraction_error":
            if set(expect) != {"status"}:
                raise ValueError("extraction_error expectations cannot declare reward bounds")
            if case.get("class") == "positive":
                raise ValueError("positive controls must declare graded expectations")
            continue
        low, high = expect.get("reward_min"), expect.get("reward_max")
        if (
            type(low) not in (int, float)
            or type(high) not in (int, float)
            or not 0 <= low <= high <= 1
        ):
            raise ValueError("control reward range must be numeric within [0, 1]")
    if positives < 1:
        raise ValueError("controls require at least one positive case")
    return cases


def validate_package_manifest(
    manifest: dict, *, specification_sha256: str, runtime_image: str | None
) -> None:
    """Reject packages outside the pinned single-step regrade surface."""
    runtimes = manifest.get("verifier_runtimes")
    if manifest.get("harbor_revision") != HARBOR_REVISION:
        raise RuntimeError("package does not pin the required Harbor revision")
    if manifest.get("specification_sha256") != specification_sha256:
        raise RuntimeError("package specification hash does not match the bundle")
    if manifest.get("step_names") != ["step-1"]:
        raise RuntimeError("regrade requires one canonical Harbor step")
    valid_runtime = (
        isinstance(runtimes, list)
        and len(runtimes) == 1
        and (
            runtimes == [None]
            if runtime_image is None
            else isinstance(runtimes[0], dict)
            and runtimes[0].get("kind") == "container"
            and runtimes[0].get("image") == runtime_image
        )
    )
    if not valid_runtime:
        raise RuntimeError("package verifier runtime does not match the specification")


def _file_identity(path: Path, label: str) -> dict:
    return {"label": label, "sha256": sha256(path)}


def _source_tree_sha256(root: Path) -> str:
    """Hash source files, excluding mutable local environments and caches."""
    excluded = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "target"}
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if excluded.intersection(relative.parts) or any(part.startswith(".venv") for part in relative.parts) or not path.is_file():
            continue
        digest.update(relative.as_posix().encode() + b"\0")
        digest.update(sha256(path).encode() + b"\n")
    return digest.hexdigest()


def validate_persisted_plan(plan: dict, path: Path, expected_sha256: str) -> bytes:
    """Require a previously persisted, hash-pinned plan before remote work."""
    if not path.is_file():
        raise FileNotFoundError("a persisted external regrade plan is required")
    data = path.read_bytes()
    if not isinstance(expected_sha256, str) or sha256(path) != expected_sha256:
        raise RuntimeError("persisted regrade plan hash mismatch")
    if json.loads(data) != plan:
        raise RuntimeError("persisted regrade plan does not match frozen inputs")
    return data


def _runtime_fields(bundle: Path) -> tuple[str | None, str | None, str]:
    binding = json.loads((bundle / "binding.json").read_text())
    kind = binding.get("environment", {}).get("kind")
    if kind not in {"none", "docker", "shellsim"}:
        raise ValueError("regrade supports no-tool, ShellSim and container bindings")
    if kind == "none" and binding.get("tools") != []:
        raise ValueError("regrade requires a no-tool binding")
    if (bundle / "composite-verifier.json").exists():
        raise ValueError("composite verification is unsupported")
    specification = json.loads((bundle / "specification.json").read_text())
    steps = specification.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        raise ValueError("multistep regrade is unsupported")
    step = steps[0]
    if not isinstance(step, dict):
        raise TypeError("regrade step is invalid")
    declared = step.get("verifier")
    if not isinstance(declared, dict):
        raise TypeError("regrade verifier is invalid")
    # Current TaskSpec packages place the TaskTrove verifier directly here.
    # The established code-answer wrapper is also supported, but only with its
    # exact outer kind and output contract. A malformed direct declaration must
    # not fall through to a nested value with different semantics.
    if declared.get("kind") == "tasktrove":
        verifier = declared
    elif (
        declared.get("kind") == "code_answer"
        and isinstance(declared.get("output_path"), str)
        and declared["output_path"]
    ):
        if declared.get("runtime") is not None or declared.get("judge") is not None:
            raise ValueError("code-answer wrapper cannot add runtime or judge")
        verifier = declared.get("verifier")
    else:
        raise ValueError("regrade requires a code ContainerRuntime verifier")
    if not isinstance(verifier, dict):
        raise TypeError("regrade verifier is invalid")
    if verifier.get("kind") != "tasktrove":
        raise ValueError("regrade requires a code ContainerRuntime verifier")
    if verifier.get("mode") in NATIVE_DETERMINISTIC_MODES:
        if verifier.get("runtime") is not None or verifier.get("judge") is not None:
            raise ValueError("native deterministic regrade requires no runtime or judge")
        return None, None, kind
    runtime = verifier.get("runtime")
    if verifier.get("mode") != "script" or not isinstance(runtime, dict) or runtime.get("kind") != "container":
        raise ValueError("regrade requires a code ContainerRuntime verifier")
    image, supervisor = runtime.get("image"), runtime.get("supervisor_python")
    if not isinstance(image, str) or not isinstance(supervisor, str):
        raise TypeError("regrade runtime fields are invalid")
    return image, supervisor, kind


def _bundle_args(root: Path, taskcompendium_source: Path, *, parallelism: int) -> Namespace:
    source = root / "input"
    resources = source / "candidate-resources.json"
    return Namespace(
        package=source / "package",
        bundle=source / "bundle",
        controls=source / "bundle" / "controls.json",
        daytona_tools=source / "tools",
        taskcompendium_source=taskcompendium_source,
        candidate_resources=resources if resources.exists() or resources.is_symlink() else None,
        repeats=DEFAULT_REPEATS,
        parallelism=parallelism,
    )


def _candidate_resource_request(args, binding_kind: str) -> tuple[dict | None, dict | None]:
    """Load the optional frozen Docker request without introducing defaults."""
    source = getattr(args, "candidate_resources", None)
    if source is None:
        return None, None
    if binding_kind != "docker":
        raise ValueError("candidate resources require a Docker regrade binding")
    path = Path(source)
    request = load_candidate_resources(path)
    return request, {
        "path": "candidate-resources.json",
        "sha256": sha256(path),
        "request": request,
    }


def validate_plan_bundle(root: Path, taskcompendium_source: Path, expected_plan_sha256: str) -> tuple[dict, Namespace]:
    """Check every transported input and the prior metadata-only plan."""
    from scripts import restore_bundle

    if not Path(restore_bundle.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise RuntimeError("regrade transport validator is outside the controller source")
    restore_bundle.regrade_members(root)
    plan_path = root / "plan.json"
    frozen = json.loads(plan_path.read_text())
    args = _bundle_args(root, taskcompendium_source, parallelism=frozen["parallelism"])
    image, supervisor, kind = _runtime_fields(Path(args.bundle))
    controls = json.loads(Path(args.controls).read_text())
    validate_package_manifest(
        json.loads((Path(args.package) / "manifest.json").read_text()),
        specification_sha256=sha256(Path(args.bundle) / "specification.json"),
        runtime_image=image,
    )
    current = build_plan(args, controls, runtime_image=image, supervisor_python=supervisor, binding_kind=kind)
    validate_persisted_plan(current, plan_path, expected_plan_sha256)
    return current, args


def create_plan_bundle(source: Path, taskcompendium_source: Path, output: Path, *, parallelism: int = DEFAULT_PARALLELISM) -> dict:
    """Build a self-contained input bundle without creating a remote runtime."""
    from scripts import restore_bundle

    source = source.resolve()
    output = output.resolve()
    if not Path(restore_bundle.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise RuntimeError("evaluation source validator is outside the controller source")
    restore_bundle.evaluation_members(source)
    if output.exists() or output.is_relative_to(source):
        raise FileExistsError("regrade plan output must be new and outside source")
    output.mkdir(parents=True)
    shutil.copytree(source, output / "input")
    args = _bundle_args(output, taskcompendium_source, parallelism=parallelism)
    image, supervisor, kind = _runtime_fields(Path(args.bundle))
    controls = json.loads(Path(args.controls).read_text())
    validate_package_manifest(
        json.loads((Path(args.package) / "manifest.json").read_text()),
        specification_sha256=sha256(Path(args.bundle) / "specification.json"),
        runtime_image=image,
    )
    plan = build_plan(args, controls, runtime_image=image, supervisor_python=supervisor, binding_kind=kind)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    files = {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted((output / "input").rglob("*")) if path.is_file()
    }
    files["plan.json"] = sha256(output / "plan.json")
    manifest = {
        "schema_version": REGRADE_BUNDLE_SCHEMA,
        "plan_sha256": files["plan.json"],
        "files": files,
        "directories": sorted(
            path.relative_to(output).as_posix()
            for path in output.rglob("*") if path.is_dir()
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_plan_bundle(output, taskcompendium_source, files["plan.json"])
    return {"bundle": str(output), "plan_sha256": files["plan.json"], "manifest_sha256": sha256(output / "manifest.json"), "cells": plan["cell_count"]}


def plan_command(args) -> int:
    from .synthesis import SOURCE_LOCK, OfficialToolchain

    OfficialToolchain._verify(
        Path(args.taskcompendium_source), json.loads(SOURCE_LOCK.read_text())
    )
    print(json.dumps(create_plan_bundle(Path(args.source), Path(args.taskcompendium_source), Path(args.out), parallelism=args.parallelism), indent=2))
    return 0


def run_command(args) -> int:
    source = Path(args.source)
    _, runtime_args = validate_plan_bundle(source, Path(args.taskcompendium_source), args.plan_sha256)
    if os.environ.get("CAPABILITY_REGRADE_INNER") != "1":
        from .synthesis import OfficialToolchain

        toolchain = OfficialToolchain.resolve(Path(args.out), args.taskcompendium_source)
        command = toolchain.runtime_command()
        python_index = command.index("python")
        command[python_index + 1 : python_index + 2] = [
            "-m", "capability_pipeline.cli", "regrade",
            "--source", str(source.resolve()), "--plan-sha256", args.plan_sha256,
            "--taskcompendium-source", str(Path(args.taskcompendium_source).resolve()),
            "--out", str(Path(args.out).resolve()),
        ]
        environment = dict(os.environ)
        environment["CAPABILITY_REGRADE_INNER"] = "1"
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + environment.get("PYTHONPATH", "")
        return subprocess.run(command, env=environment, check=False).returncode
    runtime_args.plan = source / "plan.json"
    runtime_args.plan_sha256 = args.plan_sha256
    runtime_args.output = Path(args.out)
    report = asyncio.run(execute(runtime_args))
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["state"] == "passed" else 2


def add_parser(subparsers) -> None:
    plan = subparsers.add_parser("regrade-plan", help="Freeze authored regrade inputs without remote execution")
    plan.add_argument("--source", required=True, help="immutable evaluation input bundle")
    plan.add_argument("--taskcompendium-source", required=True)
    plan.add_argument("--out", required=True, help="new regrade plan bundle")
    plan.add_argument("--parallelism", type=int, default=DEFAULT_PARALLELISM)
    plan.set_defaults(func=plan_command)
    run = subparsers.add_parser("regrade", help="Capture and grade a frozen matrix on remote Harbor/Daytona")
    run.add_argument("--source", required=True, help="restored regrade plan bundle")
    run.add_argument("--plan-sha256", required=True)
    run.add_argument("--taskcompendium-source", required=True)
    run.add_argument("--out", required=True)
    run.set_defaults(func=run_command)


def build_plan(args, controls: dict, *, runtime_image: str | None, supervisor_python: str | None, binding_kind: str = "none") -> dict:
    """Freeze the complete matrix and its controller/toolchain inputs before execution."""
    package, bundle, controls_path = map(
        Path, (args.package, args.bundle, args.controls)
    )
    cases = validate_controls(controls, binding_kind=binding_kind)
    _, candidate_resource_request = _candidate_resource_request(args, binding_kind)
    repeats = int(getattr(args, "repeats", DEFAULT_REPEATS))
    parallelism = int(getattr(args, "parallelism", DEFAULT_PARALLELISM))
    if repeats != DEFAULT_REPEATS:
        raise ValueError("regrade requires exactly 10 repeats")
    if not 1 <= parallelism <= DEFAULT_PARALLELISM:
        raise ValueError("regrade parallelism must be between 1 and 8")
    taskcompendium = Path(args.taskcompendium_source)
    helper = Path(args.daytona_tools) / "dt.py"
    lock_files = [taskcompendium / "pyproject.toml", taskcompendium / "uv.lock"]
    required = [
        package / "manifest.json",
        bundle / "binding.json",
        bundle / "renderings.json",
        bundle / "specification.json",
        controls_path,
        helper,
        *lock_files,
    ]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("regrade input or toolchain lock is missing")
    cells = [
        {
            "ordinal": ordinal,
            "case_id": case["id"],
            "repeat": repeat,
            "trial_name": f"regrade-{case['id']}-run-{repeat:02d}",
        }
        for ordinal, (case, repeat) in enumerate(
            ((case, repeat) for case in cases for repeat in range(1, repeats + 1)),
            1,
        )
    ]
    plan = {
        "schema_version": "capability-fixed-submission-regrade-plan-v1",
        "scope": "fixed-submission verifier determinism diagnostic; no task acceptance or repair budget effect",
        "binding_kind": binding_kind,
        "grading_strategy": "capture_once" if binding_kind in {"docker", "shellsim"} and runtime_image is not None else "harbor_replay",
        "require_fixed_grading_input": True,
        "replay_inventory": {
            case["id"]: authored_replay_inventory(case, {0: next(item for item in cases if item["class"] == "positive")}, 1, controls_path.parent)
            for case in cases
        },
        "repeats": repeats,
        "parallelism": parallelism,
        "case_count": len(cases),
        "cell_count": len(cells),
        "cells": cells,
        "runtime": {
            "image": runtime_image,
            "supervisor_python": supervisor_python,
        },
        "candidate_resource_request": candidate_resource_request,
        "identities": {
            "package_tree_sha256": tree_sha256(package),
            "bundle_tree_sha256": tree_sha256(bundle),
            "controls": _file_identity(controls_path, "controls"),
            "runtime": _file_identity(Path(__file__).with_name("runtime.py"), "capability_pipeline/runtime.py"),
            "regrade": _file_identity(Path(__file__), "capability_pipeline/regrade.py"),
            "verifier_adapter": _file_identity(
                Path(__file__).with_name("daytona_verifier.py"), "capability_pipeline/daytona_verifier.py"
            ),
            "grading_input_helper": _file_identity(
                Path(__file__).with_name("grading_input.py"), "capability_pipeline/grading_input.py"
            ),
            "fixed_grading_capture": _file_identity(
                Path(__file__).with_name("fixed_grading_capture.py"),
                "capability_pipeline/fixed_grading_capture.py",
            ),
            "candidate_environment": _file_identity(
                Path(__file__).with_name("daytona_environment.py"), "capability_pipeline/daytona_environment.py"
            ),
            "candidate_resources": _file_identity(
                Path(__file__).with_name("daytona_resources.py"), "capability_pipeline/daytona_resources.py"
            ),
            "candidate_telemetry": _file_identity(
                Path(__file__).with_name("daytona_telemetry.py"), "capability_pipeline/daytona_telemetry.py"
            ),
            "runtime_agents": _file_identity(
                Path(__file__).with_name("runtime_agents.py"), "capability_pipeline/runtime_agents.py"
            ),
            "verifier_policy": _file_identity(
                Path(__file__).with_name("daytona_policy.py"), "capability_pipeline/daytona_policy.py"
            ),
            "daytona_helper": _file_identity(helper, "daytona-tools/dt.py"),
            "taskcompendium_source_sha256": _source_tree_sha256(taskcompendium),
            "taskcompendium_locks": [_file_identity(path, path.name) for path in lock_files],
            "harbor_revision": HARBOR_REVISION,
        },
    }
    if runtime_image is None:
        if supervisor_python is not None:
            raise ValueError("native verifier has no supervisor python")
        source_verifier = taskcompendium / "src" / "taskcompendium" / "harbor" / "verifier.py"
        if sha256(source_verifier) != BASE_VERIFIER_SHA256:
            raise ValueError("native regrade requires the pinned base TaskCompendium verifier source")
        plan["verifier_surface"] = "native_deterministic"
        plan["identities"]["native_verifier"] = _file_identity(
            Path(__file__).with_name("native_verifier.py"),
            "capability_pipeline/native_verifier.py",
        )
        plan["identities"]["native_semantic_verifier"] = _file_identity(
            source_verifier,
            "taskcompendium/harbor/verifier.py (base source)",
        )
        plan["identities"]["native_semantic_verifier_runtime_sha256"] = PATCHED_VERIFIER_SHA256
    if binding_kind == "shellsim":
        from .shellsim_snapshot_extension import extension_record

        plan["identities"]["shellsim_overlay"] = extension_record()
        plan["identities"]["shellsim_capture_adapter"] = _file_identity(
            Path(__file__).with_name("shellsim_reset_environment.py"),
            "capability_pipeline/shellsim_reset_environment.py",
        )
        plan["identities"]["shellsim_snapshot_validator"] = _file_identity(
            Path(__file__).with_name("non_docker_reset.py"),
            "capability_pipeline/non_docker_reset.py",
        )
    return plan


def summarize(plan: dict, cells: list[dict], controls: dict) -> dict:
    """Compute determinism without removing failed cells from the denominator."""
    expected_cells = plan["cell_count"]
    by_case = {case["id"]: [] for case in validate_controls(controls, binding_kind=plan.get("binding_kind", "none"))}
    planned = {
        (cell["case_id"], cell["repeat"], cell["ordinal"], cell["trial_name"])
        for cell in plan["cells"]
    }
    observed = [
        (cell.get("case_id"), cell.get("repeat"), cell.get("ordinal"), cell.get("trial_name"))
        for cell in cells
    ]
    matrix_exact = len(observed) == len(planned) and set(observed) == planned
    for cell in cells:
        if cell.get("case_id") in by_case:
            by_case[cell["case_id"]].append(cell)
    cases = []
    all_deterministic = matrix_exact and len(cells) == expected_cells
    native = plan.get("verifier_surface") == "native_deterministic"
    captured = plan.get("grading_strategy") == "capture_once"
    expected_native_adapter = plan.get("identities", {}).get("native_verifier", {}).get("sha256")
    expected_semantic_verifier = plan.get("identities", {}).get("native_semantic_verifier_runtime_sha256")
    native_sources_bound = all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        for value in (expected_native_adapter, expected_semantic_verifier)
    )
    for case in controls["cases"]:
        rows = by_case[case["id"]]
        rewards = [row.get("reward") for row in rows]
        outcomes = [row.get("outcome_class") for row in rows]
        extraction_expected = case["expect"]["status"] == "extraction_error"
        exact = (
            len(rows) == plan["repeats"]
            and len(set(map(json.dumps, rewards))) == 1
            and len(set(outcomes)) == 1
            and all(outcome == ("extraction_error" if extraction_expected else "graded") for outcome in outcomes)
            and (not native or native_sources_bound)
            and all(
                (
                    row.get("private_verifier") is None
                    and isinstance(row.get("native_verifier_receipt"), dict)
                    and row["native_verifier_receipt"] == {
                        "schema_version": "capability-native-verifier-receipt-v1",
                        "adapter_sha256": expected_native_adapter,
                        "semantic_verifier_sha256": expected_semantic_verifier,
                    }
                    and row.get("verifier_result_present") is (not extraction_expected)
                    and (not extraction_expected or (
                        isinstance(row.get("exception"), dict)
                        and row["exception"].get("type") == "ExtractionError"
                        and isinstance(row.get("grading_sha256"), str)
                        and re.fullmatch(r"[0-9a-f]{64}", row["grading_sha256"]) is not None
                    ))
                ) if native else
                (
                    row.get("transport") == "captured_private_grade"
                    and row.get("trial_sha256") is None
                    and row.get("verifier_result_present") is None
                    and row.get("candidate_environment") is None
                    and row.get("exception") is None
                    and isinstance(row.get("capture_manifest_sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", row["capture_manifest_sha256"]) is not None
                    and isinstance(row.get("grading_sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", row["grading_sha256"]) is not None
                    and isinstance(row.get("private_verifier"), dict)
                ) if captured else
                (
                    row.get("private_verifier") is None
                    and row.get("verifier_result_present") is False
                    and isinstance(row.get("exception"), dict)
                    and row["exception"].get("type") == "ExtractionError"
                    and isinstance(row.get("grading_sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", row["grading_sha256"]) is not None
                ) if extraction_expected else isinstance(row.get("private_verifier"), dict)
                for row in rows
            )
            and (
                plan.get("binding_kind") not in {"docker", "shellsim"} or captured
                or all(isinstance(row.get("candidate_environment"), dict) for row in rows)
            )
        )
        if extraction_expected:
            expected = all(row.get("status") == "extraction_error" and row.get("reward") is None for row in rows)
        else:
            low, high = case["expect"]["reward_min"], case["expect"]["reward_max"]
            expected = all(
                row.get("status") == "graded"
                and type(row.get("reward")) in (int, float)
                and low <= row["reward"] <= high
                for row in rows
            )
        fingerprints = [row.get("grading_input_fingerprint") for row in rows]
        fingerprint_hashes = (
            "submission_sha256",
            "grading_input_sha256",
            "specification_sha256",
            "protocol_sha256",
            "transcript_sha256",
            "payload_sha256",
        )
        fixed_input = (
            len(rows) == plan["repeats"]
            and all(
                isinstance(value, dict)
                and value.get("schema_version") == "capability-grading-input-fingerprint-v1"
                and all(
                    isinstance(value.get(name), str)
                    and re.fullmatch(r"[0-9a-f]{64}", value[name]) is not None
                    for name in fingerprint_hashes
                )
                and type(value.get("workspace_file_count")) is int
                and value["workspace_file_count"] >= 0
                and type(value.get("step_index")) is int
                and value["step_index"] == 0
                for value in fingerprints
            )
            and len({value["grading_input_sha256"] for value in fingerprints}) == 1
        ) if plan.get("require_fixed_grading_input") or extraction_expected else True
        if captured:
            fixed_input = fixed_input and len({row.get("capture_manifest_sha256") for row in rows}) == 1
        fixed_input_assessment = (
            "verified" if fixed_input else
            "unassessed_missing_preverifier_fingerprint" if extraction_expected and all(value is None for value in fingerprints) else
            "failed"
        )
        passed = exact and expected and fixed_input
        all_deterministic = all_deterministic and passed
        cases.append(
            {
                "case_id": case["id"],
                "cell_count": len(rows),
                "outcome_classes": outcomes,
                "rewards": rewards,
                "reward_equal": exact,
                "expectation_met": expected,
                "fixed_grading_input": fixed_input,
                "fixed_grading_input_assessment": fixed_input_assessment,
                "grading_input_sha256s": [value.get("grading_input_sha256") if isinstance(value, dict) else None for value in fingerprints],
                "passed": passed,
            }
        )
    unresolved_input_only = (
        not all_deterministic
        and matrix_exact
        and len(cells) == expected_cells
        and all(case["reward_equal"] and case["expectation_met"] for case in cases)
        and any(case["fixed_grading_input_assessment"] == "unassessed_missing_preverifier_fingerprint" for case in cases)
        and all(case["fixed_grading_input_assessment"] != "failed" for case in cases)
    )
    return {
        "state": "passed" if all_deterministic else "unassessed" if unresolved_input_only else "failed",
        "denominator": expected_cells,
        "recorded_cells": len(cells),
        "all_cells_retained": len(cells) == expected_cells,
        "planned_matrix_exact": matrix_exact,
        "cases": cases,
    }


def attest_trial_isolation(
    cells: list[dict], trials: Path, *, kind: str, native: bool,
    runtime_image: str | None, supervisor_python: str | None,
    captures: list[dict] | None = None,
    shellsim_bridge_sha256: str | None = None,
) -> None:
    """Attest candidate sandboxes and only the private verifiers that exist."""
    candidate_ids: set[str] = set()
    verifier_ids: set[str] = set()
    captures = captures or []
    shellsim_sessions: set[str] = set()
    if kind == "shellsim":
        from .non_docker_reset import shellsim_candidate_record

        if shellsim_bridge_sha256 is None:
            raise ValueError("ShellSim bridge fingerprint is absent")
        for row in (captures if captures else cells):
            root = trials / row["trial_name"]
            try:
                row["candidate_environment"] = shellsim_candidate_record(
                    root, shellsim_sessions, shellsim_bridge_sha256,
                )
            except Exception as error:  # noqa: BLE001
                row["outcome_class"] = "invalid_candidate_evidence"
                row["isolation_error"] = {"type": type(error).__name__, "message": str(error)}
        snapshots = {
            row["candidate_environment"]["initial_snapshot_sha256"]
            for row in (captures if captures else cells)
            if isinstance(row.get("candidate_environment"), dict)
        }
        if len(snapshots) != 1:
            for row in (captures if captures else cells):
                row["outcome_class"] = "invalid_candidate_evidence"
                row["isolation_error"] = {"type": "ValueError", "message": "ShellSim initial VFS differs"}
    for capture in captures:
        capture_root = trials / capture["trial_name"]
        if kind == "docker":
            try:
                capture["candidate_environment"] = provider_isolation_record(
                    capture_root / "daytona-environment.json", trials, candidate_ids
                )
            except Exception as error:  # noqa: BLE001
                capture["outcome_class"] = "invalid_candidate_evidence"
                capture["isolation_error"] = {"type": type(error).__name__, "message": str(error)}
    for row in cells:
        trial_root = trials / row["trial_name"]
        if kind == "docker" and not captures:
            try:
                row["candidate_environment"] = provider_isolation_record(
                    trial_root / "daytona-environment.json", trials, candidate_ids
                )
            except Exception as error:  # noqa: BLE001
                row["outcome_class"] = "invalid_candidate_evidence"
                row["isolation_error"] = {"type": type(error).__name__, "message": str(error)}
    if native:
        return
    for capture in captures:
        artifact = trials / capture["trial_name"] / "verifier" / "taskcompendium-result.json"
        if not artifact.is_file():
            capture["outcome_class"] = "missing_capture_grade"
            continue
        try:
            capture["private_verifier"] = verifier_isolation_record(
                json.loads(artifact.read_text()), verifier_ids, candidate_ids,
                runtime_image=runtime_image, supervisor_python=supervisor_python,
            )
        except Exception as error:  # noqa: BLE001
            capture["outcome_class"] = "invalid_verifier_evidence"
            capture["isolation_error"] = {"type": type(error).__name__, "message": str(error)}
    for row in cells:
        trial_root = trials / row["trial_name"]
        artifact = trial_root / ("private-grade.json" if captures else "verifier/taskcompendium-result.json")
        if artifact.is_file():
            try:
                result = json.loads(artifact.read_text())
                if row.get("outcome_class") == "extraction_error":
                    continue
                isolation = verifier_isolation_record(
                    result,
                    verifier_ids,
                    candidate_ids,
                    runtime_image=runtime_image,
                    supervisor_python=supervisor_python,
                )
                row["private_verifier"] = isolation
            # Invalid private-isolation evidence is a recorded cell failure.
            except Exception as error:  # noqa: BLE001
                row["outcome_class"] = "invalid_verifier_evidence"
                row["isolation_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }


def actual_extraction_transport(result: dict, trial) -> bool:
    """Recognize a complete native extraction failure before checking its label."""
    exception = trial.exception_info
    return (
        result.get("status") == "extraction_error"
        and result.get("reward") is None
        and exception is not None
        and exception.exception_type == "ExtractionError"
        and trial.verifier_result is None
    )


async def execute(args) -> dict:
    """Run the frozen matrix remotely through real Harbor and Daytona."""
    if os.environ.get("CAPABILITY_REMOTE_REGRADE") != "1":
        raise RuntimeError("regrade is remote-only; CAPABILITY_REMOTE_REGRADE=1 is required")
    if not sandbox_provider.credentials_present():
        raise RuntimeError(f"regrade requires {sandbox_provider.credentials_hint()}")

    import msgspec
    from taskcompendium.execution import (
        HarborExecutionConfig,
        HarborLaunchConfig,
        HarborTaskBinding,
        HarnessToolBinding,
        NoEnvironment,
    )
    from taskcompendium.harbor.runner import run_trial
    from taskcompendium.lowering import resolve_harbor_execution
    from taskcompendium.models import (
        ContainerRuntime,
        tasktrove_verifier,
        verifier_runtime,
    )
    from taskcompendium.serialization import from_json, renderings_from_json
    from tasktrove_verify.spec import Mode

    package, bundle, controls_path, output = map(
        Path, (args.package, args.bundle, args.controls, args.output)
    )
    if output.exists():
        raise FileExistsError("regrade output must be a fresh path")
    binding = msgspec.json.decode(
        (bundle / "binding.json").read_bytes(), type=HarborTaskBinding
    )
    binding_kind = msgspec.to_builtins(binding.environment).get("kind")
    if binding_kind not in {"none", "docker", "shellsim"}:
        raise RuntimeError("regrade supports only no-tool, ShellSim or container bindings")
    if binding_kind == "none" and (not isinstance(binding.environment, NoEnvironment) or binding.tools):
        raise RuntimeError("regrade requires a no-tool binding")
    if (bundle / "composite-verifier.json").exists():
        raise RuntimeError("composite verification is unsupported")
    specification = from_json((bundle / "specification.json").read_bytes())
    if len(specification.steps) != 1:
        raise RuntimeError("multistep regrade is unsupported")
    verifier = tasktrove_verifier(specification.steps[0].verifier)
    runtime = verifier_runtime(specification.steps[0].verifier)
    native = _native_deterministic_verifier(verifier, runtime)
    if not native and (verifier is None or verifier.mode != Mode.SCRIPT or not isinstance(runtime, ContainerRuntime)):
        raise RuntimeError("regrade requires a deterministic native or code ContainerRuntime verifier")
    runtime_image = None if native else runtime.image
    supervisor_python = None if native else runtime.supervisor_python
    validate_package_manifest(
        json.loads((package / "manifest.json").read_text()),
        specification_sha256=sha256(bundle / "specification.json"),
        runtime_image=runtime_image,
    )

    controls = json.loads(controls_path.read_text())
    cases = validate_controls(controls, binding_kind=binding_kind)
    plan = build_plan(
        args,
        controls,
        runtime_image=runtime_image,
        supervisor_python=supervisor_python,
        binding_kind=binding_kind,
    )
    plan_path = Path(args.plan)
    plan_data = validate_persisted_plan(plan, plan_path, getattr(args, "plan_sha256", None))
    helper_root = Path(args.daytona_tools).resolve()
    os.environ["CAPABILITY_DAYTONA_TOOLS"] = str(helper_root)
    output.mkdir(parents=True)
    (output / "plan.json").write_bytes(plan_data)
    start_identities = plan["identities"]
    renderings = renderings_from_json((bundle / "renderings.json").read_bytes())
    candidate_resources, _ = _candidate_resource_request(args, binding_kind)
    env_config, kind = environment_config(
        binding, os.environ.get("TASKCOMPENDIUM_SHELLSIM_BRIDGE"), candidate_resources
    )
    if kind != binding_kind:
        raise AssertionError("validated binding resolved unexpectedly")
    shellsim_bridge_sha256 = (
        sha256(Path(os.environ["TASKCOMPENDIUM_SHELLSIM_BRIDGE"]))
        if kind == "shellsim" else None
    )
    positive = next(case for case in cases if case["class"] == "positive")
    case_by_id = {case["id"]: case for case in cases}
    execution_by_case = {}
    for case in cases:
        replay_binding = binding if kind == "none" else HarborTaskBinding(
            binding.environment, (HarnessToolBinding("terminal", kind),)
        )
        kwargs = authored_replay_agent_kwargs(
            case,
            {0: positive},
            1,
            controls_path.parent,
            terminal_available=kind in {"docker", "shellsim"},
            workspace_staging_available=kind in {"docker", "shellsim"},
        )
        execution = resolve_harbor_execution(
            renderings,
            HarborExecutionConfig(replay_binding, HarborLaunchConfig("replay")),
            env_config,
            agent_kwargs=kwargs,
        )
        execution["verifier"]["import_path"] = (
            "capability_pipeline.native_verifier:NativeDiagnosticSemanticVerifier"
            if native else "capability_pipeline.daytona_verifier:DaytonaSemanticVerifier"
        )
        execution_by_case[case["id"]] = execution

    trials = output / "runtime-trials"
    trials.mkdir()

    def shellsim_execution(execution: dict, name: str) -> dict:
        if kind != "shellsim":
            return execution
        value = dict(execution)
        value["environment"] = {
            **execution["environment"],
            "import_path": "capability_pipeline.shellsim_reset_environment:ShellSimResetEnvironment",
            "kwargs": {
                **execution["environment"].get("kwargs", {}),
                "snapshot_path": str((trials / name / "initial-snapshot.json").resolve()),
            },
        }
        return value
    semaphore = asyncio.Semaphore(plan["parallelism"])
    captures: list[dict] = []

    if plan["grading_strategy"] == "capture_once":
        from capability_pipeline.daytona_verifier import grade_captured_in_daytona
        from capability_pipeline.fixed_grading_capture import load_capture

        async def capture_case(case: dict) -> dict:
            name = f"capture-{case['id']}"
            trial_root = trials / name
            capture_root = trial_root / "verifier" / "fixed-grading-capture"
            artifact = trial_root / "verifier" / "taskcompendium-result.json"
            record = {
                "case_id": case["id"],
                "trial_name": name,
                "capture_path": str(capture_root.relative_to(output)),
                "trial_sha256": None,
                "grading_sha256": None,
                "capture_manifest_sha256": None,
                "candidate_environment": None,
                "private_verifier": None,
                "status": None,
                "reward": None,
                "exception": None,
                "verifier_result_present": None,
                "grading_input_fingerprint": None,
                "outcome_class": "capture_missing",
            }
            try:
                execution = dict(shellsim_execution(execution_by_case[case["id"]], name))
                execution["verifier"] = {
                    **execution["verifier"],
                    "import_path": "capability_pipeline.daytona_verifier:CapturingDaytonaSemanticVerifier",
                }
                async with semaphore:
                    trial = await run_trial(package, execution, trials, name)
                result = json.loads(artifact.read_text()) if artifact.is_file() else {}
                capture = load_capture(capture_root)
                manifest = capture["manifest"]
                if (
                    manifest.get("source_specification_sha256") != sha256(bundle / "specification.json")
                    or manifest.get("source_renderings_sha256") != sha256(bundle / "renderings.json")
                    or manifest.get("fingerprint") != (result.get("detail") or {}).get("grading_input_fingerprint")
                ):
                    raise RuntimeError("capture source or grading fingerprint differs from its frozen trial")
                record.update(
                    trial_sha256=sha256(trial_root / "result.json") if (trial_root / "result.json").is_file() else None,
                    grading_sha256=sha256(artifact) if artifact.is_file() else None,
                    capture_manifest_sha256=capture["manifest_sha256"],
                    status=result.get("status"),
                    reward=result.get("reward"),
                    exception=({"type": trial.exception_info.exception_type, "message": trial.exception_info.exception_message}
                               if trial.exception_info is not None else None),
                    verifier_result_present=trial.verifier_result is not None,
                    grading_input_fingerprint=manifest["fingerprint"],
                    outcome_class=("captured" if trial.exception_info is None and trial.verifier_result is not None
                                   and result.get("status") == "graded" else "capture_trial_failed"),
                )
            except Exception as error:  # noqa: BLE001
                record.update(
                    trial_sha256=sha256(trial_root / "result.json") if (trial_root / "result.json").is_file() else None,
                    grading_sha256=sha256(artifact) if artifact.is_file() else None,
                    exception={"type": type(error).__name__, "message": str(error)},
                    outcome_class="capture_exception",
                )
            return record

        captures = list(await asyncio.gather(*(capture_case(case) for case in cases)))
        capture_by_case = {capture["case_id"]: capture for capture in captures}

        async def run_captured_cell(cell: dict) -> dict:
            row = dict(cell)
            capture = capture_by_case[cell["case_id"]]
            trial_root = trials / cell["trial_name"]
            trial_root.mkdir()
            artifact = trial_root / "private-grade.json"
            row.update(
                transport="captured_private_grade",
                trial_sha256=None,
                grading_sha256=None,
                capture_manifest_sha256=capture["capture_manifest_sha256"],
                verifier_result_present=None,
                candidate_environment=None,
                exception=None,
                replay=authored_replay_inventory(
                    case_by_id[cell["case_id"]], {0: positive}, 1, controls_path.parent
                ),
            )
            try:
                if capture["outcome_class"] != "captured" or capture["capture_manifest_sha256"] is None:
                    raise RuntimeError("capture trial did not produce a verified grade and input closure")
                capture_root = output / capture["capture_path"]
                async with semaphore:
                    result = await asyncio.to_thread(
                        grade_captured_in_daytona,
                        capture_root,
                        expected_manifest_sha256=capture["capture_manifest_sha256"],
                    )
                grade = {
                    "schema_version": "capability-captured-private-grade-v1",
                    "cell": cell,
                    "capture_manifest_sha256": capture["capture_manifest_sha256"],
                    "status": result.status.value,
                    "reward": result.reward,
                    "detail": result.detail,
                }
                artifact.write_text(json.dumps(grade, sort_keys=True, allow_nan=False) + "\n")
                row.update(
                    status=grade["status"], reward=grade["reward"],
                    outcome_class=grade["status"],
                    grading_sha256=sha256(artifact),
                    grading_input_fingerprint=grade["detail"].get("grading_input_fingerprint"),
                )
            except Exception as error:  # noqa: BLE001
                row.update(
                    status=None, reward=None, outcome_class="capture_grade_exception",
                    exception={"type": type(error).__name__, "message": str(error)},
                    grading_input_fingerprint=None,
                )
            return row

        cells = list(await asyncio.gather(*(run_captured_cell(cell) for cell in plan["cells"])))
    else:

        async def run_cell(cell: dict) -> dict:
            row = dict(cell)
            trial_root = trials / cell["trial_name"]
            try:
                async with semaphore:
                    trial = await run_trial(
                        package,
                        shellsim_execution(execution_by_case[cell["case_id"]], cell["trial_name"]),
                        trials,
                        cell["trial_name"],
                    )
                artifact = trial_root / "verifier" / "taskcompendium-result.json"
                result = json.loads(artifact.read_text()) if artifact.is_file() else {}
                observed_extraction = (
                    actual_extraction_transport(result, trial)
                    if native else expected_extraction_transport(case_by_id[cell["case_id"]], result, trial)
                )
                row.update(
                    status=result.get("status"),
                    reward=result.get("reward"),
                    outcome_class=(
                        "extraction_error"
                        if observed_extraction
                        else "trial_exception"
                        if trial.exception_info is not None
                        else "missing_verifier_result"
                        if trial.verifier_result is None
                        else result.get("status", "missing_grade")
                    ),
                    exception=(
                        {
                            "type": trial.exception_info.exception_type,
                            "message": trial.exception_info.exception_message,
                        }
                        if trial.exception_info is not None
                        else None
                    ),
                    trial_sha256=(
                        sha256(trial_root / "result.json")
                        if (trial_root / "result.json").is_file()
                        else None
                    ),
                    grading_sha256=sha256(artifact) if artifact.is_file() else None,
                    verifier_result_present=trial.verifier_result is not None,
                    grading_input_fingerprint=(result.get("detail") or {}).get("grading_input_fingerprint"),
                    native_verifier_receipt=(result.get("detail") or {}).get("native_verifier_receipt") if native else None,
                    replay=authored_replay_inventory(
                        case_by_id[cell["case_id"]],
                        {0: positive},
                        1,
                        controls_path.parent,
                    ),
                )
            # Every launched cell remains in the denominator, including provider,
            # Harbor, serialization, and artifact failures.
            except Exception as error:  # noqa: BLE001
                row.update(
                    status=None,
                    reward=None,
                    outcome_class="runner_exception",
                    exception={"type": type(error).__name__, "message": str(error)},
                    trial_sha256=(
                        sha256(trial_root / "result.json")
                        if (trial_root / "result.json").is_file()
                        else None
                    ),
                    grading_sha256=None,
                    replay=authored_replay_inventory(
                        case_by_id[cell["case_id"]],
                        {0: positive},
                        1,
                        controls_path.parent,
                    ),
                )
            return row
    
        cells = list(await asyncio.gather(*(run_cell(cell) for cell in plan["cells"])))
    attest_trial_isolation(
        cells, trials, kind=kind, native=native,
        runtime_image=runtime_image, supervisor_python=supervisor_python,
        captures=captures,
        shellsim_bridge_sha256=shellsim_bridge_sha256,
    )

    end_plan = build_plan(
        args,
        controls,
        runtime_image=runtime_image,
        supervisor_python=supervisor_python,
        binding_kind=kind,
    )
    identities_stable = start_identities == end_plan["identities"] and (
        kind != "shellsim"
        or shellsim_bridge_sha256 == sha256(Path(os.environ["TASKCOMPENDIUM_SHELLSIM_BRIDGE"]))
    )
    summary = summarize(plan, cells, controls)
    if captures and any(capture["outcome_class"] != "captured" for capture in captures):
        summary["state"] = "failed"
    if not identities_stable:
        summary["state"] = "failed"
    report = {
        "schema_version": "capability-fixed-submission-regrade-v1",
        "scope": "diagnostic_only; never grants task acceptance",
        "plan_sha256": sha256(output / "plan.json"),
        "identities_stable": identities_stable,
        "summary": summary,
        "cells": cells,
        "captures": captures,
        "shellsim_bridge_sha256": shellsim_bridge_sha256,
    }
    (output / "regrade.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report

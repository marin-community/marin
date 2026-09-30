"""Execute artifact-bound control trials through pinned TaskCompendium Harbor."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import shlex
import stat
import sys
from pathlib import Path, PurePosixPath

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capability_pipeline import sandbox_provider
from capability_pipeline.daytona_policy import (
    verifier_bootstrap_sha256,
    verifier_snapshot_recipe,
)
from capability_pipeline.daytona_resources import profile_from_receipt, snapshot_name

HARBOR_REVISION = "93147ea9e07b04ec8d2eb5afd2916386f1aacc69"
_CONTROL_WORKSPACE_MAX_FILES = 256
_CONTROL_WORKSPACE_MAX_BYTES = 16 * 1024 * 1024
_CONTROL_WORKSPACE_CHUNK_BASE64_BYTES = 32 * 1024
_CONTROL_WORKSPACE_DENIED_PARTS = {
    ".env",
    "api_key",
    "credential",
    "credentials",
    "grader",
    "ground_truth",
    "password",
    "private",
    "secret",
    "secrets",
    "token",
    "verifier",
}


def normalize_openai_base(value: str) -> str:
    root = value.rstrip("/").removesuffix("/v1")
    if not root:
        raise ValueError("OpenAI-compatible base URL is empty")
    return root + "/v1"


def judge_step_indices(raw_specification: object) -> tuple[int, ...]:
    """Step indices whose verifier grades through a TaskTrove judge.

    A raw-JSON mirror of the parsed selection in ``run_controls``
    (``taskcompendium.models.tasktrove_verifier(step.verifier).mode ==
    Mode.JUDGE``), usable by the construction controller, which cannot import
    the pinned TaskCompendium toolchain. ``run_controls`` asserts that both
    selections agree so they cannot drift apart silently.
    """
    if not isinstance(raw_specification, dict):
        return ()
    steps = raw_specification.get("steps")
    if not isinstance(steps, list):
        return ()
    indices = []
    for index, step in enumerate(steps):
        verifier = step.get("verifier") if isinstance(step, dict) else None
        if isinstance(verifier, dict) and verifier.get("kind") == "code_answer":
            verifier = verifier.get("verifier")
        if (
            isinstance(verifier, dict)
            and verifier.get("kind") == "tasktrove"
            and verifier.get("mode") == "judge"
        ):
            indices.append(index)
    return tuple(indices)


def sha256(path: Path) -> str:
    # Runtime traces can be much larger than task specifications. Keep stage
    # handoff hashing bounded without changing the recorded digest contract.
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tree_sha256(root: Path) -> str:
    value = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        value.update(path.relative_to(root).as_posix().encode())
        value.update(b"\0" + sha256(path).encode() + b"\n")
    return value.hexdigest()


def persist_attack_report(attack: dict, output_root: Path) -> None:
    """Persist verifier annotations and rebind the adversary report digest."""
    adversary_path = output_root / attack["artifact"]
    adversary_path.write_text(
        json.dumps(attack["report"], indent=2, allow_nan=False) + "\n"
    )
    attack["artifact_sha256"] = sha256(adversary_path)


def extract_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
    raise ValueError("independent solver did not return a JSON object")


def expected_extraction_transport(case: dict, result: dict, trial_result) -> bool:
    exception = trial_result.exception_info
    return (
        case.get("expect", {}).get("status") == "extraction_error"
        and result.get("status") == "extraction_error"
        and result.get("reward") is None
        and exception is not None
        and exception.exception_type == "ExtractionError"
        and trial_result.verifier_result is None
    )


# Agent-side limits of the independent solver.  The verifier still grades the
# final state (measured 2026-09-29: all 47 turn-capped solver attempts on
# catalog-full-construct-003 carried a grade, 6 of them 1.0), so the grade is
# the attempt's outcome; aborting the whole gate on it discarded every control.
SOLVER_LIMIT_EXCEPTIONS = frozenset({"TurnCapExhaustedError"})


def solver_limit_graded(positive: bool, trial_result, artifact: Path) -> bool:
    exception = trial_result.exception_info
    return bool(
        positive
        and exception is not None
        and exception.exception_type in SOLVER_LIMIT_EXCEPTIONS
        and trial_result.verifier_result is not None
        and artifact.is_file()
    )


def authored_oracle_plan(
    controls: dict, step_count: int
) -> tuple[list[dict], dict[int, dict]]:
    """Retain every authored positive while selecting one seed per task step."""
    cases = controls.get("cases")
    if not isinstance(cases, list):
        raise TypeError("controls cases must be a list")
    positives = [case for case in cases if case.get("class") == "positive"]
    seeds_by_step: dict[int, dict] = {}
    for case in positives:
        step_index = case.get("step_index", 0)
        if type(step_index) is not int or not 0 <= step_index < step_count:
            raise RuntimeError("positive control has an invalid step index")
        seeds_by_step.setdefault(step_index, case)
    if set(seeds_by_step) != set(range(step_count)):
        raise RuntimeError("controls need one replayable positive seed per step")
    return positives, seeds_by_step


def _control_workspace_inventory(
    task_root: Path, case: dict
) -> tuple[Path, list[str], list[dict]]:
    declared = case.get("workspace")
    if not isinstance(declared, str) or not declared:
        raise RuntimeError("control workspace must be a nonempty relative path")
    relative = PurePosixPath(declared)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or str(relative) != declared
        or any(part in {"", "."} for part in relative.parts)
    ):
        raise RuntimeError("control workspace path is unsafe")
    if any(
        part.lower().replace("-", "_") in _CONTROL_WORKSPACE_DENIED_PARTS
        for part in relative.parts
    ):
        raise RuntimeError("control workspace path is private or credential-like")
    source = task_root.joinpath(*relative.parts)
    resolved_root = task_root.resolve()
    if task_root.is_symlink():
        raise RuntimeError("control task root cannot be a symlink")
    ancestor = task_root
    for part in relative.parts:
        ancestor /= part
        if ancestor.is_symlink():
            raise RuntimeError("control workspace path cannot traverse a symlink")
    try:
        resolved_source = source.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("control workspace is unavailable") from error
    if resolved_root not in resolved_source.parents or not resolved_source.is_dir():
        raise RuntimeError("control workspace must be a contained directory")
    if source.is_symlink():
        raise RuntimeError("control workspace cannot be a symlink")

    directories: set[PurePosixPath] = set()
    files: list[dict] = []
    total_bytes = 0
    for path in sorted(source.rglob("*")):
        submission_path = PurePosixPath(path.relative_to(source).as_posix())
        if path.is_symlink():
            raise RuntimeError("control workspace cannot contain symlinks")
        if any(
            part.lower().replace("-", "_") in _CONTROL_WORKSPACE_DENIED_PARTS
            for part in submission_path.parts
        ):
            raise RuntimeError(
                "control workspace contains a private or credential-like path"
            )
        if path.is_dir():
            directories.add(submission_path)
            continue
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode):
            raise RuntimeError("control workspace contains a non-regular file")
        total_bytes += path.stat().st_size
        files.append(
            {
                "path": submission_path.as_posix(),
                "sha256": sha256(path),
                "size": path.stat().st_size,
                "executable": bool(path.stat().st_mode & stat.S_IXUSR),
            }
        )
        directories.update(
            parent for parent in submission_path.parents if parent != PurePosixPath(".")
        )
    if (
        len(files) > _CONTROL_WORKSPACE_MAX_FILES
        or total_bytes > _CONTROL_WORKSPACE_MAX_BYTES
    ):
        raise RuntimeError("control workspace exceeds the trusted replay limit")
    return source, [item.as_posix() for item in sorted(directories)], files


def control_replay_inventory(task_root: Path, case: dict) -> dict | None:
    """Return the validated, credential-free inventory for a fixed submission."""
    if case.get("workspace") is None:
        return None
    _, directories, files = _control_workspace_inventory(task_root, case)
    chunk_count = sum(
        max(
            1,
            (4 * ((item["size"] + 2) // 3) + _CONTROL_WORKSPACE_CHUNK_BASE64_BYTES - 1)
            // _CONTROL_WORKSPACE_CHUNK_BASE64_BYTES,
        )
        for item in files
    )
    return {
        "action": "materialize_declared_fixed_submission",
        "transport": "trusted_replay_base64_sha256",
        "workspace": case["workspace"],
        "destination": "candidate_workdir",
        "materialization_action_count": len(directories) + 2 * len(files) + chunk_count,
        "directories": directories,
        "files": files,
    }


def trusted_workspace_replay_manifest(task_root: Path, case: dict) -> dict | None:
    """Describe controller-owned bytes for direct ShellSim staging.

    This is deliberately separate from the terminal/base64 replay inventory.  A
    ShellSim control workspace is controller-owned fixed evidence, so writing
    it through its VFS bridge must not consume the candidate's cumulative shell
    fuel or output budget.
    """
    if case.get("workspace") is None:
        return None
    source, directories, files = _control_workspace_inventory(task_root, case)
    return {
        "schema_version": "capability-trusted-shellsim-workspace-v1",
        "workspace": case["workspace"],
        "source_root": str(source),
        "directories": directories,
        "files": files,
    }


def authored_replay_inventory(
    case: dict,
    positive_by_step: dict[int, dict],
    step_count: int,
    task_root: Path,
) -> list[dict]:
    """Describe each fixed workspace staged for an authored replay."""
    target_step = case.get("step_index", 0)
    selected = (
        [case]
        if step_count == 1
        else [
            case if index == target_step else positive_by_step[index]
            for index in range(step_count)
        ]
    )
    return [
        {"step_index": index, **inventory}
        for index, selected_case in enumerate(selected)
        if (inventory := control_replay_inventory(task_root, selected_case)) is not None
    ]


def control_replay_commands(task_root: Path, case: dict) -> list[str]:
    """Encode a declared fixed submission as path-safe trusted replay actions."""
    declared = case.get("workspace")
    if declared is None:
        commands = case.get("commands", [])
        if not isinstance(commands, list) or any(
            not isinstance(command, str) for command in commands
        ):
            raise RuntimeError("control commands must be a list of strings")
        return list(commands)
    source, directory_names, files = _control_workspace_inventory(task_root, case)

    replay: list[str] = []
    directories = {PurePosixPath(item) for item in directory_names}

    def non_symlink_checks(path: PurePosixPath) -> str:
        checked = [
            parent for parent in reversed(path.parents) if parent != PurePosixPath(".")
        ]
        checked.append(path)
        # ShellSim's test builtin does not implement the three-argument
        # ``[ ! -L path ]`` form. Its persistent errexit state would then end
        # every later replay action. An explicit conditional has the same
        # fail-closed symlink check in both ShellSim and POSIX shells.
        return "; ".join(
            f"if [ -L {shlex.quote(f'./{item}')} ]; then exit 1; fi"
            for item in checked
        )

    for directory in sorted(directories, key=lambda item: (len(item.parts), str(item))):
        remote_directory = shlex.quote(f"./{directory}")
        replay.append(
            f"set -eu; {non_symlink_checks(directory)}; mkdir -p {remote_directory}"
        )
    for item in files:
        submission_path = PurePosixPath(item["path"])
        path = source.joinpath(*submission_path.parts)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        chunks = [
            encoded[index : index + _CONTROL_WORKSPACE_CHUNK_BASE64_BYTES]
            for index in range(0, len(encoded), _CONTROL_WORKSPACE_CHUNK_BASE64_BYTES)
        ] or [""]
        destination = shlex.quote(f"./{submission_path}")
        expected = item["sha256"]
        permission = "0755" if item["executable"] else "0644"
        replay.append(
            f"set -eu; {non_symlink_checks(submission_path)}; : > {destination}"
        )
        for chunk in chunks:
            replay.append(
                "set -eu; command -v base64 >/dev/null; "
                f"{non_symlink_checks(submission_path)}; "
                f"printf %s {shlex.quote(chunk)} | base64 -d >> {destination}"
            )
        replay.append(
            "set -eu; command -v sha256sum >/dev/null; "
            f"{non_symlink_checks(submission_path)}; "
            f'actual=$(sha256sum {destination}); [ "${{actual%% *}}" = {expected} ]; '
            f"chmod {permission} {destination}"
        )
    commands = case.get("commands", [])
    if not isinstance(commands, list) or any(
        not isinstance(command, str) for command in commands
    ):
        raise RuntimeError("control commands must be a list of strings")
    return [*replay, *commands]


def authored_replay_agent_kwargs(
    case: dict,
    positive_by_step: dict[int, dict],
    step_count: int,
    task_root: Path,
    *,
    terminal_available: bool,
    workspace_staging_available: bool,
    direct_workspace_staging: bool = False,
) -> dict:
    """Build a ReplayAgent attempt without silently dropping authored state."""

    def attempt(selected: dict) -> dict:
        if "transcript" in selected:
            raise RuntimeError("control transcript replay is unsupported")
        if selected.get("workspace") is not None and not workspace_staging_available:
            raise RuntimeError(
                "control workspace replay requires a Docker or ShellSim terminal"
            )
        trusted_workspace = (
            trusted_workspace_replay_manifest(task_root, selected)
            if direct_workspace_staging and selected.get("workspace") is not None
            else None
        )
        commands = (
            list(selected.get("commands", []))
            if trusted_workspace is not None
            else control_replay_commands(task_root, selected)
        )
        if any(not isinstance(command, str) for command in commands):
            raise RuntimeError("control commands must be a list of strings")
        if commands and not terminal_available:
            raise RuntimeError("control replay actions require a terminal")
        return {
            "response": selected.get("response", ""),
            **({"commands": commands} if terminal_available else {}),
            **(
                {"trusted_workspace": trusted_workspace}
                if trusted_workspace is not None
                else {}
            ),
        }
    if step_count == 1:
        return attempt(case)
    target_step = case.get("step_index", 0)
    if type(target_step) is not int or not 0 <= target_step < step_count:
        raise RuntimeError("control has an invalid step index")
    return {
        "steps": [
            attempt(case if index == target_step else positive_by_step[index])
            for index in range(step_count)
        ]
    }


def uses_trusted_workspace_staging(agent_kwargs: dict) -> bool:
    """Whether one replay configuration requires the ShellSim-only agent."""
    if agent_kwargs.get("trusted_workspace") is not None:
        return True
    return any(
        isinstance(step, dict) and step.get("trusted_workspace") is not None
        for step in agent_kwargs.get("steps", [])
    )


def runtime_gate_issues(controls: dict, evidence: dict) -> list[str]:
    from capability_pipeline.synthesis import _controls_pass

    _, issues = _controls_pass(controls, evidence, external=True)
    attestation = evidence.get("attestation", {})
    if attestation.get("oracle", {}).get("authored") is not True:
        issues.append("authored reference control did not pass")
    if attestation.get("solver", {}).get("state") != "passed":
        issues.append("independent solver needs adjudication after bounded retries")
    adversarial = attestation.get("adversarial", {})
    if adversarial.get("authored_controls_executed") is not True:
        issues.append("authored adversarial controls did not execute")
    if adversarial.get("independent_attack_executed") is not True:
        issues.append("independent attack suite needs adjudication or retry")
    return list(dict.fromkeys(issues))


def provider_isolation_record(
    artifact: Path, output_root: Path, seen_sandbox_ids: set[str]
) -> dict:
    provider = json.loads(artifact.read_text())
    required_strings = (
        "adapter",
        "daytona_sdk_version",
        "image",
        "sandbox_id",
        "snapshot",
    )
    if (
        provider.get("adapter") not in sandbox_provider.KNOWN_ADAPTERS
        or provider.get("network_block_all") is not True
        or any(
            not isinstance(provider.get(key), str) or not provider[key]
            for key in required_strings
        )
    ):
        raise RuntimeError("Daytona trial lacks enforced isolation evidence")
    sandbox_id = provider["sandbox_id"]
    if sandbox_id in seen_sandbox_ids:
        raise RuntimeError("Daytona controls did not receive fresh sandboxes")
    seen_sandbox_ids.add(sandbox_id)
    return {
        "provider_artifact": str(artifact.relative_to(output_root)),
        "provider_artifact_sha256": sha256(artifact),
        "sandbox_id": sandbox_id,
    }


def verifier_isolation_record(
    result: dict,
    seen_sandbox_ids: set[str],
    candidate_sandbox_ids: set[str],
    *,
    runtime_image: str | None = None,
    supervisor_python: str = "python3",
) -> dict:
    detail = result.get("detail")
    expected_adapter = sha256(Path(__file__).with_name("daytona_verifier.py"))
    expected_bootstrap = verifier_bootstrap_sha256()
    required = {
        # This grade was produced in this process, so it must claim the
        # provider that actually ran it, not merely any known one.
        "verifier_isolation": sandbox_provider.isolation_id(),
        "verifier_adapter_sha256": expected_adapter,
        "verifier_bootstrap_sha256": expected_bootstrap,
    }
    if not isinstance(detail, dict) or any(
        detail.get(key) != value for key, value in required.items()
    ):
        raise RuntimeError("container grade lacks bound sandbox verifier evidence")
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
        raise RuntimeError("private verifier cleanup is unconfirmed")
    profile = None
    if "verifier_requested_resource_profile" in detail:
        try:
            profile = profile_from_receipt(
                detail["verifier_requested_resource_profile"]
            )
        except ValueError as exc:
            raise RuntimeError(
                "container grade has malformed verifier resource profile"
            ) from exc
    recipe_sha256 = None
    if runtime_image is not None:
        recipe = verifier_snapshot_recipe(runtime_image, supervisor_python)
        recipe_sha256 = hashlib.sha256(recipe.encode()).hexdigest()
        if profile is not None:
            expected_snapshot = snapshot_name("cap-verifier", recipe, profile)
        else:
            # Old receipts did not carry a resource profile and therefore used
            # the recipe-only cache identity.  Do not permit this fallback for
            # a malformed present profile.
            expected_snapshot = f"cap-verifier-{recipe_sha256[:20]}"
        if (
            detail.get("verifier_supervisor_python") != supervisor_python
            or detail.get("verifier_bootstrap_command_sha256")
            != verifier_bootstrap_sha256(supervisor_python)
            or detail.get("verifier_snapshot_recipe_sha256") != recipe_sha256
            or detail.get("verifier_snapshot") != expected_snapshot
        ):
            raise RuntimeError(
                "container grade lacks its supervisor-aware snapshot recipe"
            )
    sandbox_id = detail.get("verifier_sandbox_id")
    snapshot = detail.get("verifier_snapshot")
    if (
        not isinstance(sandbox_id, str)
        or not sandbox_id
        or not isinstance(snapshot, str)
        or not snapshot
    ):
        raise RuntimeError("container grade lacks verifier sandbox identity")
    if sandbox_id in seen_sandbox_ids or sandbox_id in candidate_sandbox_ids:
        raise RuntimeError("container grading reused a verifier or candidate sandbox")
    seen_sandbox_ids.add(sandbox_id)
    record = {
        "sandbox_id": sandbox_id,
        "snapshot": snapshot,
        "adapter_sha256": expected_adapter,
        "bootstrap_sha256": expected_bootstrap,
    }
    if cleanup is not None:
        record["cleanup"] = cleanup
    if recipe_sha256 is not None:
        record.update(
            supervisor_python=supervisor_python,
            bootstrap_command_sha256=verifier_bootstrap_sha256(supervisor_python),
            snapshot_recipe_sha256=recipe_sha256,
        )
    return record


def composite_isolation_record(
    result: dict,
    checks: list[dict],
    config_sha256: str,
    seen_sandbox_ids: set[str],
    candidate_sandbox_ids: set[str],
    *,
    runtime_image: str | None = None,
    supervisor_python: str = "python3",
) -> dict:
    detail = result.get("detail")
    expected_adapter = sha256(Path(__file__).with_name("composite_verifier.py"))
    expected_policy = sha256(Path(__file__).with_name("composite_policy.py"))
    machine_results = (
        detail.get("machine_results") if isinstance(detail, dict) else None
    )
    if (
        not isinstance(detail, dict)
        or detail.get("composite_adapter_sha256") != expected_adapter
        or detail.get("composite_policy_sha256") != expected_policy
        or detail.get("composite_config_sha256") != config_sha256
        or not isinstance(machine_results, list)
        or [item.get("id") for item in machine_results if isinstance(item, dict)]
        != [check["id"] for check in checks]
    ):
        raise RuntimeError("composite grade lacks bound machine-check evidence")
    records = []
    for machine_result, check in zip(machine_results, checks, strict=True):
        supervisor_python = check.get("supervisor_python", "python3")
        machine_detail = machine_result.get("detail")
        strong_present = isinstance(machine_detail, dict) and any(
            key in machine_detail
            for key in (
                "verifier_supervisor_python",
                "verifier_bootstrap_command_sha256",
                "verifier_snapshot_recipe_sha256",
                "verifier_requested_resource_profile",
            )
        )
        record = verifier_isolation_record(
            machine_result,
            seen_sandbox_ids,
            candidate_sandbox_ids,
            **(
                {
                    "runtime_image": check["image"],
                    "supervisor_python": supervisor_python,
                }
                if supervisor_python != "python3" or strong_present
                else {
                    "runtime_image": runtime_image,
                    "supervisor_python": supervisor_python,
                }
            ),
        )
        records.append({"id": machine_result["id"], **record})
    return {
        "adapter_sha256": expected_adapter,
        "policy_sha256": expected_policy,
        "config_sha256": config_sha256,
        "machine_checks": records,
    }


def load_candidate_resources(path: Path) -> dict:
    """Read an explicit provider request; this does not prove enforced limits."""
    from capability_pipeline.daytona_resources import CANDIDATE_DEFAULT, resolve_profile

    if path.is_symlink() or not path.is_file():
        raise ValueError("candidate resources must be a regular JSON file")
    value = json.loads(path.read_text())
    if value is None:
        raise ValueError("candidate resources must be an explicit resource mapping")
    resolve_profile(value, default=CANDIDATE_DEFAULT)
    return value


def environment_config(binding, bridge: str | None, candidate_resources=None):
    from taskcompendium.execution import (
        DockerEnvironment,
        NoEnvironment,
        ShellSimEnvironment,
    )

    environment = binding.environment
    if candidate_resources is not None and not isinstance(
        environment, DockerEnvironment
    ):
        raise ValueError("candidate resources require a Docker environment")
    if isinstance(environment, NoEnvironment):
        return {
            "import_path": "taskcompendium.harbor.environments:NoToolEnvironment"
        }, "none"
    if isinstance(environment, ShellSimEnvironment):
        if not bridge or not Path(bridge).is_file():
            raise RuntimeError("ShellSim bridge is required for a real Harbor trial")
        return {
            "import_path": "taskcompendium.harbor.environments:ShellSimEnvironment",
            "kwargs": {
                "limits": {
                    "cpu": environment.max_steps,
                    "output": environment.max_output_bytes,
                },
                "bridge_path": str(Path(bridge).resolve()),
            },
        }, "shellsim"
    if isinstance(environment, DockerEnvironment):
        if not sandbox_provider.credentials_present():
            raise RuntimeError(f"container Harbor trials require {sandbox_provider.credentials_hint()}")
        return {
            "import_path": "capability_pipeline.daytona_environment:DaytonaHarborEnvironment",
            "kwargs": {
                "snapshot_prefix": os.environ.get(
                    "CAPABILITY_DAYTONA_SNAPSHOT_PREFIX", "cap-harbor"
                ),
                **(
                    {"resource_profile": candidate_resources}
                    if candidate_resources is not None
                    else {}
                ),
            },
        }, "docker"
    raise TypeError(f"unsupported Harbor environment: {type(environment).__name__}")


async def execute(args) -> dict:
    import msgspec
    from taskcompendium.execution import (
        HarborExecutionConfig,
        HarborLaunchConfig,
        HarborTaskBinding,
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

    from capability_pipeline.runtime_agents import (
        adversary_request_timeout,
        adversary_token_limits,
    )

    package, bundle, controls_path, output = map(
        Path, (args.package, args.bundle, args.controls, args.output)
    )
    binding = msgspec.json.decode(
        (bundle / "binding.json").read_bytes(), type=HarborTaskBinding
    )
    renderings = renderings_from_json((bundle / "renderings.json").read_bytes())
    specification = from_json((bundle / "specification.json").read_bytes())
    composite_path = bundle / "composite-verifier.json"
    composite_steps = {}
    composite_config_sha256 = None
    if composite_path.is_file():
        import taskcompendium.harbor.runner as native_runner
        import taskcompendium.harbor.verifier as native_verifier

        from capability_pipeline.composite_extension import (
            PATCHED_RUNNER_SHA256,
            PATCHED_VERIFIER_SHA256,
            runtime_attestation_bindings,
            validate_extension_marker,
        )
        from capability_pipeline.composite_policy import validate_composite_config

        composite_config_sha256 = sha256(composite_path)
        composite_adapter_sha256 = sha256(
            Path(__file__).with_name("composite_verifier.py")
        )
        composite_policy_sha256 = sha256(
            Path(__file__).with_name("composite_policy.py")
        )
        composite_steps = validate_composite_config(
            json.loads(composite_path.read_text()),
            specification_sha256=sha256(bundle / "specification.json"),
            adapter_sha256=composite_adapter_sha256,
            policy_sha256=composite_policy_sha256,
            step_count=len(specification.steps),
        )
        validate_extension_marker(
            package,
            adapter_sha256=composite_adapter_sha256,
            policy_sha256=composite_policy_sha256,
            config_sha256=composite_config_sha256,
            supported=True,
        )
        if sha256(Path(native_verifier.__file__)) != PATCHED_VERIFIER_SHA256:
            raise RuntimeError(
                "composite runtime lacks the pinned fail-closed TaskCompendium guard"
            )
        if sha256(Path(native_runner.__file__)) != PATCHED_RUNNER_SHA256:
            raise RuntimeError(
                "composite runtime lacks the pinned TaskCompendium runner extension"
            )
        if not sandbox_provider.credentials_present():
            raise RuntimeError(f"composite machine checks require {sandbox_provider.credentials_hint()}")
    container_runtimes = {
        index: runtime
        for index, step in enumerate(specification.steps)
        if isinstance((runtime := verifier_runtime(step.verifier)), ContainerRuntime)
    }
    container_steps = set(container_runtimes)
    if container_steps and not sandbox_provider.credentials_present():
        raise RuntimeError(f"container verifiers require {sandbox_provider.credentials_hint()}")
    from capability_pipeline.verifier_budget import verifier_override_seconds

    # Sandboxed graders pay provider provisioning inside Harbor's verifier
    # phase; budget it on top of the grader's own runtime timeout.
    verifier_timeout = verifier_override_seconds(
        package,
        {index: float(runtime.timeout) for index, runtime in container_runtimes.items()},
        json.loads(composite_path.read_text()) if composite_steps else None,
        frozenset(composite_steps),
    )
    semantic_verifiers = tuple(
        verifier
        for step in specification.steps
        if (verifier := tasktrove_verifier(step.verifier)) is not None
    )
    judge_verifiers = tuple(
        verifier for verifier in semantic_verifiers if verifier.mode == Mode.JUDGE
    )
    parsed_judge_steps = tuple(
        index
        for index, step in enumerate(specification.steps)
        if (verifier := tasktrove_verifier(step.verifier)) is not None
        and verifier.mode == Mode.JUDGE
    )
    if parsed_judge_steps != judge_step_indices(
        json.loads((bundle / "specification.json").read_bytes())
    ):
        raise RuntimeError(
            "controller judge-step detection disagrees with the parsed TaskSpec"
        )
    verifier_kwargs = None
    judge_policy_sha256 = None
    if judge_verifiers:
        if not os.environ.get("GLM_API_TOKEN"):
            raise RuntimeError("native judge trials require GLM_API_TOKEN")
        configured_base = os.environ.get("GLM_BASE_URL")
        expected_model = os.environ.get("CAPABILITY_JUDGE_MODEL", "glm-5.3")
        expected_provider = os.environ.get("CAPABILITY_JUDGE_PROVIDER", "glm")
        if not configured_base:
            raise RuntimeError("native judge trials require GLM_BASE_URL")
        expected_base = normalize_openai_base(configured_base)
        judge_policy_path = bundle / "judge-policy.json"
        judge_policy = json.loads(judge_policy_path.read_text())
        expected_policy = {
            "provider": expected_provider,
            "model": expected_model,
            "base_url": expected_base,
        }
        if judge_policy != expected_policy:
            raise RuntimeError(
                "frozen judge policy does not match the configured GLM policy"
            )
        judge_policy_sha256 = sha256(judge_policy_path)
        for verifier in judge_verifiers:
            policy = verifier.judge.policy
            if (
                normalize_openai_base(policy.base_url) != expected_base
                or policy.model != expected_model
                or policy.provider != expected_provider
            ):
                raise RuntimeError(
                    "judge policy must match the configured GLM provider, model, and base URL"
                )
        verifier_kwargs = {"judge_api_key_env": "GLM_API_TOKEN"}
    if not os.environ.get("GLM_BASE_URL") or not os.environ.get("GLM_API_TOKEN"):
        raise RuntimeError(
            "independent GLM solver requires GLM_BASE_URL and GLM_API_TOKEN"
        )
    adversary_limits = adversary_token_limits(
        os.environ.get("CAPABILITY_ADVERSARY_TOKEN_LIMITS")
    )
    adversary_timeout = adversary_request_timeout(
        os.environ.get("CAPABILITY_ADVERSARY_REQUEST_TIMEOUT")
    )
    resources_arg = getattr(args, "candidate_resources", None)
    resources_path = Path(resources_arg) if resources_arg else None
    resources_hash = sha256(resources_path) if resources_path else None
    candidate_resources = (
        load_candidate_resources(resources_path) if resources_path else None
    )
    env_config, kind = environment_config(
        binding, args.shellsim_bridge, candidate_resources
    )
    controls = json.loads(controls_path.read_text())
    manifest = json.loads((package / "manifest.json").read_text())
    step_names = manifest["step_names"]
    if len(step_names) != len(specification.steps):
        raise RuntimeError("Harbor manifest step list does not match TaskSpec")
    oracle_controls, positive_by_step = authored_oracle_plan(controls, len(step_names))
    trials = output.parent / "runtime-trials"
    trials.mkdir(parents=True, exist_ok=True)
    records = []
    raw_trials = []
    isolation_cases = []
    seen_sandbox_ids: set[str] = set()
    seen_verifier_sandbox_ids: set[str] = set()
    solver_transcripts = []
    solver_all_passed = True
    positive_execution = None
    oracle_cases = []
    for case in oracle_controls:
        replay_binding = binding
        if kind != "none":
            from taskcompendium.execution import HarnessToolBinding

            replay_binding = HarborTaskBinding(
                binding.environment, (HarnessToolBinding("terminal", kind),)
            )
        agent_kwargs = authored_replay_agent_kwargs(
            case,
            positive_by_step,
            len(step_names),
            controls_path.parent,
            terminal_available=kind != "none",
            workspace_staging_available=kind in {"docker", "shellsim"},
            direct_workspace_staging=kind == "shellsim",
        )
        oracle_execution = resolve_harbor_execution(
            renderings,
            HarborExecutionConfig(replay_binding, HarborLaunchConfig("replay")),
            env_config,
            agent_kwargs=agent_kwargs,
            verifier_kwargs=verifier_kwargs,
        )
        if uses_trusted_workspace_staging(agent_kwargs):
            oracle_execution["agent"]["import_path"] = (
                "capability_pipeline.runtime_agents:TrustedWorkspaceReplayAgent"
            )
        if container_steps:
            oracle_execution["verifier"]["import_path"] = (
                "capability_pipeline.daytona_verifier:DaytonaSemanticVerifier"
            )
        if composite_steps:
            oracle_execution["verifier"]["import_path"] = (
                "capability_pipeline.composite_verifier:CompositeSemanticVerifier"
            )
        if verifier_timeout is not None:
            oracle_execution["verifier"]["override_timeout_sec"] = verifier_timeout
        oracle_name = "oracle-" + "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in case["id"]
        )
        oracle_trial = await run_trial(package, oracle_execution, trials, oracle_name)
        oracle_root = trials / oracle_name
        oracle_step = case.get("step_index", 0)
        oracle_step_root = (
            oracle_root / "steps" / step_names[oracle_step]
            if len(step_names) > 1
            else oracle_root
        )
        oracle_artifact = oracle_step_root / "verifier" / "taskcompendium-result.json"
        if (
            oracle_trial.exception_info is not None
            or oracle_trial.verifier_result is None
        ):
            exception_type = (
                oracle_trial.exception_info.exception_type
                if oracle_trial.exception_info is not None
                else "MissingVerifierResult"
            )
            raise RuntimeError(
                f"authored reference {case['id']} failed in Harbor "
                f"({exception_type}); inspect "
                f"{(oracle_root / 'result.json').relative_to(output.parent)}"
            )
        if not oracle_artifact.is_file():
            raise RuntimeError(
                f"authored reference {case['id']} lacks a grading artifact; inspect "
                f"{(oracle_root / 'result.json').relative_to(output.parent)}"
            )
        oracle_result = json.loads(oracle_artifact.read_text())
        reward = oracle_result.get("reward")
        if (
            oracle_result.get("status") != "graded"
            or type(reward) not in (int, float)
            or not 0.8 <= reward <= 1.0
        ):
            raise RuntimeError(
                f"authored reference {case['id']} did not prove the grader; inspect "
                f"{(oracle_root / 'result.json').relative_to(output.parent)}"
            )
        oracle_private_verifier = None
        if oracle_step in composite_steps:
            oracle_private_verifier = composite_isolation_record(
                oracle_result,
                composite_steps[oracle_step]["machine_checks"],
                composite_config_sha256,
                seen_verifier_sandbox_ids,
                seen_sandbox_ids,
            )
        elif oracle_step in container_steps:
            oracle_private_verifier = verifier_isolation_record(
                oracle_result,
                seen_verifier_sandbox_ids,
                seen_sandbox_ids,
                runtime_image=container_runtimes[oracle_step].image,
                supervisor_python=container_runtimes[oracle_step].supervisor_python,
            )
        oracle_isolation = {
            "mechanism": {
                "none": "no-tool-host-boundary",
                "shellsim": "shellsim-no-host-access",
                "docker": sandbox_provider.isolation_id(),
            }[kind]
        }
        if kind == "docker":
            oracle_isolation.update(
                provider_isolation_record(
                    oracle_root / "daytona-environment.json",
                    output.parent,
                    seen_sandbox_ids,
                )
            )
        oracle_trial_artifact = oracle_root / "result.json"
        oracle_cases.append(
            {
                "case_id": case["id"],
                "step_index": oracle_step,
                "trial_artifact": str(oracle_trial_artifact.relative_to(output.parent)),
                "trial_sha256": sha256(oracle_trial_artifact),
                "grading_artifact": str(oracle_artifact.relative_to(output.parent)),
                "grading_sha256": sha256(oracle_artifact),
                "result": oracle_result,
                "control_replay": authored_replay_inventory(
                    case,
                    positive_by_step,
                    len(step_names),
                    controls_path.parent,
                ),
                "isolation": oracle_isolation,
                "private_verifier": oracle_private_verifier,
            }
        )
    oracle_artifact = output.parent / "authored-oracle.json"
    oracle_artifact.write_text(
        json.dumps(
            {
                "state": "passed",
                "independent": False,
                "source": "authored_reference_controls",
                "cases": oracle_cases,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    for case in controls["cases"]:
        positive = case["class"] == "positive"
        if positive:
            solver_binding = binding
            if kind != "none":
                from taskcompendium.execution import HarborTaskBinding, ShellToolBinding

                solver_binding = HarborTaskBinding(
                    binding.environment, (ShellToolBinding("shell", kind),)
                )
            execution_config = HarborExecutionConfig(
                solver_binding,
                HarborLaunchConfig("chat" if kind == "none" else "tool_chat"),
            )
            agent_kwargs = {
                "api_base": normalize_openai_base(os.environ["GLM_BASE_URL"]),
                "api_key_env": "GLM_API_TOKEN",
                "max_tokens": int(
                    os.environ.get("CAPABILITY_SOLVER_MAX_TOKENS", "32768")
                ),
                "request_timeout": float(
                    os.environ.get("CAPABILITY_SOLVER_REQUEST_TIMEOUT", "900")
                ),
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_effort": "high",
                },
            }
            if kind != "none":
                agent_kwargs["max_turns"] = int(
                    os.environ.get("CAPABILITY_SOLVER_MAX_TURNS", "128")
                )
        else:
            replay_binding = binding
            if kind != "none":
                from taskcompendium.execution import (
                    HarborTaskBinding,
                    HarnessToolBinding,
                )

                replay_binding = HarborTaskBinding(
                    binding.environment, (HarnessToolBinding("terminal", kind),)
                )
            execution_config = HarborExecutionConfig(
                replay_binding, HarborLaunchConfig("replay")
            )
            agent_kwargs = authored_replay_agent_kwargs(
                case,
                positive_by_step,
                len(step_names),
                controls_path.parent,
                terminal_available=kind != "none",
                workspace_staging_available=kind in {"docker", "shellsim"},
                direct_workspace_staging=kind == "shellsim",
            )
        execution = resolve_harbor_execution(
            renderings,
            execution_config,
            env_config,
            agent_kwargs=agent_kwargs,
            model_name=os.environ.get("CAPABILITY_SOLVER_MODEL", "glm-5.3")
            if positive
            else None,
            verifier_kwargs=verifier_kwargs,
        )
        if not positive and uses_trusted_workspace_staging(agent_kwargs):
            execution["agent"]["import_path"] = (
                "capability_pipeline.runtime_agents:TrustedWorkspaceReplayAgent"
            )
        if positive:
            execution["agent"]["import_path"] = (
                "capability_pipeline.runtime_agents:GLMChatAgent"
                if kind == "none"
                else "capability_pipeline.runtime_agents:GLMShellToolAgent"
            )
        if container_steps:
            execution["verifier"]["import_path"] = (
                "capability_pipeline.daytona_verifier:DaytonaSemanticVerifier"
            )
        if composite_steps:
            execution["verifier"]["import_path"] = (
                "capability_pipeline.composite_verifier:CompositeSemanticVerifier"
            )
        if verifier_timeout is not None:
            execution["verifier"]["override_timeout_sec"] = verifier_timeout
        if positive:
            positive_execution = execution
        trial_base = "control-" + "".join(
            c if c.isalnum() or c in "-_" else "-" for c in case["id"]
        )
        step_index = case.get("step_index", 0)
        attempt_limit = (
            max(1, int(os.environ.get("CAPABILITY_SOLVER_RETRIES", "2")))
            if positive
            else 1
        )
        private_verifier = None
        isolation = None
        passed_solver = False
        for attempt_number in range(1, attempt_limit + 1):
            trial_name = (
                f"{trial_base}-attempt-{attempt_number}" if positive else trial_base
            )
            trial_result = await run_trial(package, execution, trials, trial_name)
            trial_root = trials / trial_name
            step_root = (
                trial_root / "steps" / step_names[step_index]
                if len(step_names) > 1
                else trial_root
            )
            artifact = step_root / "verifier" / "taskcompendium-result.json"
            result = json.loads(artifact.read_text()) if artifact.is_file() else {}
            expected_extraction = expected_extraction_transport(
                case, result, trial_result
            )
            solver_limited = solver_limit_graded(positive, trial_result, artifact)
            harbor_failed = (
                trial_result.exception_info is not None
                or trial_result.verifier_result is None
            ) and not solver_limited
            if harbor_failed and not expected_extraction:
                exception_type = (
                    trial_result.exception_info.exception_type
                    if trial_result.exception_info is not None
                    else "MissingVerifierResult"
                )
                raise RuntimeError(
                    f"Harbor trial {trial_name} failed ({exception_type}); inspect "
                    f"{(trial_root / 'result.json').relative_to(output.parent)}"
                )
            if not artifact.is_file():
                raise RuntimeError(
                    f"Harbor trial {trial_name} lacks a grading artifact; inspect "
                    f"{(trial_root / 'result.json').relative_to(output.parent)}"
                )
            if (
                not harbor_failed
                and not solver_limited
                and result.get("status") == "extraction_error"
            ):
                raise RuntimeError(
                    f"Harbor trial {trial_name} did not preserve its extraction failure"
                )
            private_verifier = None
            if step_index in composite_steps:
                private_verifier = composite_isolation_record(
                    result,
                    composite_steps[step_index]["machine_checks"],
                    composite_config_sha256,
                    seen_verifier_sandbox_ids,
                    seen_sandbox_ids,
                )
            elif step_index in container_steps:
                private_verifier = verifier_isolation_record(
                    result,
                    seen_verifier_sandbox_ids,
                    seen_sandbox_ids,
                    runtime_image=container_runtimes[step_index].image,
                    supervisor_python=container_runtimes[step_index].supervisor_python,
                )
            isolation = {
                "case_id": case["id"],
                "mechanism": {
                    "none": "no-tool-host-boundary",
                    "shellsim": "shellsim-no-host-access",
                    "docker": sandbox_provider.isolation_id(),
                }[kind],
            }
            if kind == "docker":
                isolation.update(
                    provider_isolation_record(
                        trial_root / "daytona-environment.json",
                        output.parent,
                        seen_sandbox_ids,
                    )
                )
            if not positive:
                break
            transcript = step_root / "agent" / "transcript.json"
            if not transcript.is_file():
                raise RuntimeError(
                    f"Harbor trial {trial_name} completed without a solver transcript; "
                    f"inspect {(trial_root / 'result.json').relative_to(output.parent)}"
                )
            solver_transcripts.append(
                {
                    "case_id": case["id"],
                    "step_index": step_index,
                    "attempt": attempt_number,
                    "trial_artifact": str(
                        (trial_root / "result.json").relative_to(output.parent)
                    ),
                    "trial_sha256": sha256(trial_root / "result.json"),
                    "transcript_artifact": str(transcript.relative_to(output.parent)),
                    "transcript_sha256": sha256(transcript),
                    "grading_artifact": str(artifact.relative_to(output.parent)),
                    "grading_sha256": sha256(artifact),
                    "result": result,
                    "isolation": isolation,
                    "private_verifier": private_verifier,
                    **(
                        {"agent_limit": trial_result.exception_info.exception_type}
                        if solver_limited
                        else {}
                    ),
                }
            )
            reward = result.get("reward")
            passed_solver = (
                result.get("status") == "graded"
                and type(reward) in (int, float)
                and 0.8 <= reward <= 1.0
            )
            if passed_solver:
                break
        if positive and not passed_solver:
            solver_all_passed = False
        raw_trials.append(trial_root / "result.json")
        isolation_cases.append(isolation)
        records.append(
            {
                "id": case["id"],
                "control_type": "independent_solver"
                if positive
                else "authored_adversarial_control",
                "source_author": case["source_author"],
                "category": case["category"],
                "step_index": step_index,
                "artifact": str(artifact.relative_to(output.parent)),
                "artifact_sha256": sha256(artifact),
                "result": result,
                "control_replay": authored_replay_inventory(
                    case,
                    positive_by_step,
                    len(step_names),
                    controls_path.parent,
                ),
                "private_verifier": private_verifier,
            }
        )
    if not solver_transcripts:
        raise RuntimeError("controls contain no independent positive solver trial")
    if positive_execution is None:
        raise RuntimeError("independent solver execution configuration is absent")
    solver_artifact = output.parent / "solver-transcripts.json"
    solver_artifact.write_text(json.dumps(solver_transcripts, indent=2) + "\n")
    run_artifact = raw_trials[0]
    if getattr(args, "diagnostic_no_new_adversary", False):
        attack = {
            "report": {"state": "diagnostic_not_run", "cases": []},
            "artifact": None, "artifact_sha256": None,
        }
    else:
        from capability_pipeline.adversary import run_independent_attacks

        attack = await run_independent_attacks(
            package, positive_execution, trials, output.parent, kind,
            seen_sandbox_ids=seen_sandbox_ids, token_limits=adversary_limits,
            request_timeout=adversary_timeout,
        )
        for attack_case in attack["report"].get("cases", []):
            if attack_case.get("error"):
                continue
            for attack_step in attack_case.get("steps", []):
                attack_step_index = attack_step.get("step_index")
                if attack_step_index in composite_steps:
                    attack_step["private_verifier"] = composite_isolation_record(
                        attack_step.get("result", {}),
                        composite_steps[attack_step_index]["machine_checks"],
                        composite_config_sha256,
                        seen_verifier_sandbox_ids,
                        seen_sandbox_ids,
                    )
                elif attack_step_index in container_steps:
                    attack_step["private_verifier"] = verifier_isolation_record(
                        attack_step.get("result", {}),
                        seen_verifier_sandbox_ids,
                        seen_sandbox_ids,
                        runtime_image=container_runtimes[attack_step_index].image,
                        supervisor_python=container_runtimes[
                            attack_step_index
                        ].supervisor_python,
                    )
        if container_steps or composite_steps:
            persist_attack_report(attack, output.parent)
    evidence = {
        "attestation": {
            "kind": "harbor_control_run",
            "harbor_revision": HARBOR_REVISION,
            "specification_sha256": manifest["specification_sha256"],
            "package_sha256": tree_sha256(package),
            "package_manifest_sha256": sha256(package / "manifest.json"),
            "controls_sha256": sha256(controls_path),
            "candidate_resources_sha256": resources_hash,
            "candidate_resource_request": candidate_resources,
            "judge_policy_sha256": judge_policy_sha256,
            "daytona_environment_adapter_sha256": sha256(
                Path(__file__).with_name("daytona_environment.py")
            )
            if kind == "docker"
            else None,
            "daytona_verifier_adapter_sha256": sha256(
                Path(__file__).with_name("daytona_verifier.py")
            )
            if container_steps or composite_steps
            else None,
            "daytona_verifier_bootstrap_sha256": verifier_bootstrap_sha256()
            if container_steps or composite_steps
            else None,
            **(
                runtime_attestation_bindings(composite_config_sha256)
                if composite_steps
                else {}
            ),
            "isolation": {
                "network": "blocked",
                "fresh_environment_per_case": True,
                "cases": isolation_cases,
            },
            "solver": {
                "independent": True,
                "state": "passed" if solver_all_passed else "needs_adjudication",
                "run_id": sha256(solver_artifact)[:16],
                "retry_limit": max(
                    1, int(os.environ.get("CAPABILITY_SOLVER_RETRIES", "2"))
                ),
            },
            "oracle": {
                "authored": True,
                "run_id": sha256(oracle_artifact)[:16],
            },
            "oracle_artifact": str(oracle_artifact.relative_to(output.parent)),
            "oracle_artifact_sha256": sha256(oracle_artifact),
            "solver_policy": {
                "model": os.environ.get("CAPABILITY_SOLVER_MODEL", "glm-5.3"),
                "max_tokens": int(
                    os.environ.get("CAPABILITY_SOLVER_MAX_TOKENS", "32768")
                ),
                "max_turns": int(os.environ.get("CAPABILITY_SOLVER_MAX_TURNS", "128"))
                if kind != "none"
                else 1,
                "thinking": True,
            },
            "solver_artifact": str(solver_artifact.relative_to(output.parent)),
            "solver_artifact_sha256": sha256(solver_artifact),
            "adversarial": {
                "authored_controls_executed": any(
                    c["class"] in {"negative", "malformed", "partial"}
                    for c in controls["cases"]
                ),
                "independent_attack_executed": attack["report"].get("state")
                == "passed",
                "diagnostic_new_attacks": (
                    "not_run_primary_bound"
                    if getattr(args, "diagnostic_no_new_adversary", False)
                    else None
                ),
                "transport_policy": {
                    "token_limits": list(adversary_limits),
                    "request_timeout": adversary_timeout,
                    "full_context_semantics": "max_tokens:null",
                },
            },
            "adversary_artifact": attack["artifact"],
            "adversary_artifact_sha256": attack["artifact_sha256"],
            "run_artifact": str(run_artifact.relative_to(output.parent)),
            "run_artifact_sha256": sha256(run_artifact),
        },
        "cases": records,
    }
    if resources_path and sha256(resources_path) != resources_hash:
        raise RuntimeError("candidate resource request changed during runtime")
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostic-no-new-adversary", action="store_true")
    parser.add_argument(
        "--candidate-resources",
        help="Explicit Docker candidate request JSON: cpu, memory_gb, disk_gb",
    )
    parser.add_argument(
        "--shellsim-bridge", default=os.environ.get("TASKCOMPENDIUM_SHELLSIM_BRIDGE")
    )
    args = parser.parse_args(argv)
    evidence = asyncio.run(execute(args))
    issues = runtime_gate_issues(json.loads(Path(args.controls).read_text()), evidence)
    print(
        json.dumps(
            {
                "state": "passed" if not issues else "needs_adjudication_or_retry",
                "issues": issues,
                "evidence": str(Path(args.output).resolve()),
            },
            indent=2,
        )
    )
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())

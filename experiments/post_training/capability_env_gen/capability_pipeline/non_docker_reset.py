"""Frozen five-episode reset evidence for no-tool and ShellSim Harbor tasks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .runtime import sha256
from .shellsim_snapshot_extension import extension_record

SCHEMA = "capability-non-docker-reset-v1"
PREFIX = "non_docker_reset"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or linked artifact: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"artifact must be an object: {path}")
    return value


def _source(item: Path, shellsim_bridge: Path | None = None) -> tuple[dict, str]:
    from .reset_runner import _entries_sha256, _file_tree_sha256, _tree_entries

    harbor = item / "harbor"
    entries = _tree_entries(harbor)
    binding = _json(harbor / "binding.json")
    environment = binding.get("environment")
    kind = environment.get("kind") if isinstance(environment, dict) else None
    if kind not in {"none", "shellsim"}:
        raise ValueError("non-Docker reset requires a none or shellsim binding")
    if kind == "none":
        if binding.get("tools") != []:
            raise ValueError("no-tool binding must expose no tools")
        if (harbor / "environment/inputs").exists():
            raise ValueError("no-tool task cannot expose filesystem inputs")
    manifest = _json(harbor / "manifest.json")
    steps = manifest.get("step_names")
    if not isinstance(steps, list) or not steps or any(
        not isinstance(name, str) or name in {"", ".", ".."} or "/" in name for name in steps
    ) or len(set(steps)) != len(steps):
        raise ValueError("Harbor manifest has invalid step names")
    specification = _json(harbor / "specification.json")
    for step in [specification, *(specification.get("steps") or [])]:
        if not isinstance(step, dict):
            raise TypeError("TaskSpec step must be an object")
        resources = step.get("resources") or []
        if not isinstance(resources, list):
            raise TypeError("TaskSpec resources must be a list")
        if kind == "none" and any(
            isinstance(resource, dict) and "agent" in resource.get("roles", [])
            for resource in resources
        ):
            raise ValueError("no-tool task cannot declare agent-visible files")
    prompt_paths = [
        harbor / "steps" / name / "instruction.md" if len(steps) > 1
        else harbor / "instruction.md"
        for name in steps
    ]
    prompts = [sha256(path) for path in prompt_paths]
    identity = {
        "harbor_tree_sha256": _entries_sha256(entries),
        "harbor_file_sha256": _file_tree_sha256(harbor),
        "binding_sha256": sha256(harbor / "binding.json"),
        "manifest_sha256": sha256(harbor / "manifest.json"),
        "task_toml_sha256": sha256(harbor / "task.toml"),
        "specification_sha256": sha256(harbor / "specification.json"),
        "renderings_sha256": sha256(harbor / "renderings.json"),
        "prompt_sha256": prompts,
        "step_names": steps,
        "kind": kind,
    }
    if kind == "shellsim":
        bridge = Path(shellsim_bridge) if shellsim_bridge is not None else None
        if bridge is None or bridge.is_symlink() or not bridge.is_file():
            raise ValueError("pinned ShellSim snapshot bridge is required")
        identity["shellsim_bridge_sha256"] = sha256(bridge)
        identity["shellsim_overlay"] = extension_record()
    return identity, kind


def _portable(identity: dict) -> dict:
    return {key: value for key, value in identity.items() if key != "harbor_tree_sha256"}


def _result(state: str, issues: list[str], attempt: Path | None = None, summary: dict | None = None) -> dict:
    files = {}
    if attempt is not None and attempt.is_dir():
        files = {
            f"{PREFIX}/{attempt.name}/{path.relative_to(attempt).as_posix()}": path
            for path in sorted(attempt.rglob("*")) if path.is_file()
        }
    return {
        "schema_version": SCHEMA, "state": state,
        "reviewable": state == "semantic_failed", "issues": issues,
        "summary": summary or {}, "full_quality_reset_gate": "unassessed",
        "private_outside_public_root": "unassessed", "attempt": str(attempt) if attempt else None,
        "extra_files": files,
    }


def _freeze(attempt: Path, item: Path, identity: dict, toolchain: Any, timeout: int,
            shellsim_bridge: Path | None = None) -> None:
    from .reset_runner import _tree_entries

    attempt.mkdir(parents=True)
    shutil.copytree(item / "harbor", attempt / "input/harbor", symlinks=True)
    _write(attempt / "input/harbor-tree.json", {
        "schema_version": "capability-reset-input-tree-v1",
        "entries": _tree_entries(attempt / "input/harbor"),
    })
    _write(attempt / "binding.json", {
        "schema_version": SCHEMA, "source": identity,
        "source_lock_sha256": sha256(Path(__file__).resolve().parents[1] / "vendor/task_spec/source.lock.json"),
        "controller_sha256": sha256(Path(__file__)),
        "capture_verifier_sha256": sha256(Path(__file__).with_name("non_docker_reset_capture.py")),
        "shellsim_reset_adapter_sha256": sha256(Path(__file__).with_name("shellsim_reset_environment.py")),
        "timeout_seconds": timeout,
        "toolchain_package_root": str(getattr(toolchain, "package_root", "")),
    })
    if identity["kind"] == "shellsim":
        if shellsim_bridge is None:
            raise ValueError("ShellSim bridge is absent")
        shutil.copy2(shellsim_bridge, attempt / "input/shellsim-bridge")
    _validate_frozen(attempt, identity)


def _validate_frozen(attempt: Path, identity: dict) -> None:
    from .reset_runner import _restore_tree

    if attempt.is_symlink() or any(path.is_symlink() for path in attempt.rglob("*")):
        raise ValueError("frozen reset artifacts contain a link")
    _restore_tree(
        attempt / "input/harbor", _json(attempt / "input/harbor-tree.json"),
        identity["harbor_tree_sha256"],
    )
    frozen, _ = _source(
        attempt / "input",
        attempt / "input/shellsim-bridge" if identity["kind"] == "shellsim" else None,
    )
    if frozen != identity:
        raise ValueError("frozen Harbor package differs from source")
    binding = _json(attempt / "binding.json")
    if binding.get("source") != identity or binding.get("schema_version") != SCHEMA:
        raise ValueError("frozen reset binding differs")
    if binding.get("source_lock_sha256") != sha256(
        Path(__file__).resolve().parents[1] / "vendor/task_spec/source.lock.json"
    ) or binding.get("controller_sha256") != sha256(Path(__file__)) or binding.get(
        "capture_verifier_sha256"
    ) != sha256(Path(__file__).with_name("non_docker_reset_capture.py")) or binding.get(
        "shellsim_reset_adapter_sha256"
    ) != sha256(Path(__file__).with_name("shellsim_reset_environment.py")):
        raise ValueError("frozen reset controller differs")


async def _run_capture(attempt: Path) -> dict:
    import msgspec
    from taskcompendium.execution import (
        HarborExecutionConfig,
        HarborLaunchConfig,
        HarborTaskBinding,
    )
    from taskcompendium.harbor.runner import run_trial
    from taskcompendium.lowering import resolve_harbor_execution
    from taskcompendium.rendering import render_instruction
    from taskcompendium.serialization import from_json, renderings_from_json

    package = attempt / "input/harbor"
    binding = msgspec.json.decode((package / "binding.json").read_bytes(), type=HarborTaskBinding)
    specification = from_json((package / "specification.json").read_bytes())
    renderings = renderings_from_json((package / "renderings.json").read_bytes())
    if len(specification.steps) != len(renderings):
        raise ValueError("reset rendering count differs from TaskSpec")
    step_names = _json(package / "manifest.json")["step_names"]
    for index, step_name in enumerate(step_names):
        instruction = package / "steps" / step_name / "instruction.md" if len(step_names) > 1 else package / "instruction.md"
        if instruction.read_text() != render_instruction(specification, renderings[index], index):
            raise ValueError("exported instruction differs from canonical rendering")
    kind = _json(attempt / "binding.json")["source"]["kind"]
    trial_dir = attempt / "raw/trials"
    trial_dir.mkdir(parents=True)
    episodes = []
    baseline = None
    for cycle in range(0 if kind == "shellsim" else 1, 6):
        name = "baseline" if cycle == 0 else f"reset-{cycle:02d}"
        if kind == "shellsim":
            environment = {
                "import_path": "capability_pipeline.shellsim_reset_environment:ShellSimResetEnvironment",
                "kwargs": {
                    "bridge_path": str((attempt / "input/shellsim-bridge").resolve()),
                    "limits": {"cpu": binding.environment.max_steps,
                               "output": binding.environment.max_output_bytes},
                    "snapshot_path": str((trial_dir / name / "initial-snapshot.json").resolve()),
                    "calibrate": cycle == 0,
                },
            }
        else:
            environment = {"import_path": "taskcompendium.harbor.environments:NoToolEnvironment"}
        agent_kwargs = {"response": "", "steps": [
            {"response": "", "commands": []} for _ in renderings
        ] if len(renderings) > 1 else None}
        if kind == "shellsim":
            # This diagnostic is capture-only. The task's shell-tool binding
            # requires tool_chat for a solver launch, but ReplayAgent executes no
            # commands and lets Harbor perform the real environment startup.
            execution = {
                "environment": environment,
                "agent": {"import_path": "taskcompendium.harbor.agents:ReplayAgent",
                          "upload_agent_logs": False, "kwargs": agent_kwargs, "env": {}},
            }
        else:
            execution = resolve_harbor_execution(
                renderings, HarborExecutionConfig(binding, HarborLaunchConfig("replay")),
                environment, agent_kwargs=agent_kwargs,
            )
        execution["verifier"] = {
            "import_path": "capability_pipeline.non_docker_reset_capture:PromptCaptureVerifier", "kwargs": {},
        }
        result = await run_trial(package, execution, trial_dir, name)
        root = trial_dir / name
        prompts = []
        for index, step_name in enumerate(step_names):
            step = root / "steps" / step_name if len(step_names) > 1 else root
            transcript = step / "agent/transcript.json"
            turns = json.loads(transcript.read_text())
            users = [turn["content"] for turn in turns if turn.get("role") == "user"]
            if len(users) != index + 1 or not isinstance(users[-1], str):
                raise ValueError("capture transcript lacks expected user instruction")
            prompts.append(hashlib.sha256(users[-1].encode()).hexdigest())
        row = {
            "cycle": cycle, "trial": name, "prompts_sha256": prompts,
            "result_sha256": sha256(root / "result.json"),
            "exception": result.exception_info.exception_type if result.exception_info else None,
        }
        if cycle == 0:
            baseline = row
        else:
            episodes.append(row)
    return {"schema_version": SCHEMA, "source": _json(attempt / "binding.json")["source"],
            "baseline": baseline, "episodes": episodes}


def _inner(attempt: Path, binding_sha256: str) -> int:
    if os.environ.get("CAPABILITY_REMOTE_NON_DOCKER_RESET") != "1":
        raise RuntimeError("non-Docker reset execution is remote-only")
    if sha256(attempt / "binding.json") != binding_sha256:
        raise ValueError("frozen reset binding differs")
    binding = _json(attempt / "binding.json")
    if binding.get("controller_sha256") != sha256(Path(__file__)) or binding.get(
        "capture_verifier_sha256"
    ) != sha256(Path(__file__).with_name("non_docker_reset_capture.py")):
        raise ValueError("reset controller source differs")
    _validate_frozen(attempt, binding["source"])
    report = asyncio.run(_run_capture(attempt))
    _write(attempt / "raw/report.json", report)
    return 0


def _run_remote(*, toolchain: Any, attempt: Path, timeout: int) -> int:
    command = toolchain.runtime_command()
    index = command.index("python")
    command[index + 1:] = ["-m", "capability_pipeline.non_docker_reset", "--inner",
                           "--attempt", str(attempt.resolve()), "--binding-sha256", sha256(attempt / "binding.json")]
    env = dict(os.environ)
    env["CAPABILITY_REMOTE_NON_DOCKER_RESET"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    with (attempt / "controller-run.log").open("x") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def _shellsim_snapshot(path: Path) -> tuple[str, str, int, dict]:
    """Recompute the bridge's canonical digest from a retained complete reply."""
    record = _json(path)
    nonce, pid = record.get("session_nonce"), record.get("bridge_pid")
    if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{32}", nonce) or type(pid) is not int or pid <= 0:
        raise ValueError("ShellSim reset session identity is invalid")
    result = record.get("snapshot")
    if not isinstance(result, dict) or set(result) != {
        "snapshot", "snapshot_sha256", "entry_count", "total_file_bytes"
    }:
        raise ValueError("ShellSim snapshot reply is incomplete")
    snapshot = result["snapshot"]
    if not isinstance(snapshot, dict) or set(snapshot) != {"schema_version", "root", "limits", "entries"}:
        raise ValueError("ShellSim snapshot wire is incomplete")
    if snapshot["schema_version"] != "taskcompendium-shellsim-vfs-snapshot-v1" or snapshot["root"] != "/" or snapshot["limits"] != {
        "max_entries": 100_000, "max_file_bytes": 64 * 1024 * 1024,
        "max_response_bytes": 8 * 1024 * 1024,
    }:
        raise ValueError("ShellSim snapshot root or limits differ")
    entries = snapshot["entries"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 100_000 or result["entry_count"] != len(entries):
        raise ValueError("ShellSim snapshot entry count differs")
    paths = []
    file_bytes = 0
    ordered = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "kind", "mode", "size", "sha256", "target"}:
            raise ValueError("ShellSim snapshot entry is malformed")
        name, kind, mode = entry["path"], entry["kind"], entry["mode"]
        if not isinstance(name, str) or (name != "." and (
            not name or name.startswith("/") or any(part in {"", ".", ".."} for part in name.split("/"))
        )) or kind not in {"directory", "file", "symlink"} or type(mode) is not int or not 0 <= mode <= 0o7777:
            raise ValueError("ShellSim snapshot path, kind or mode is invalid")
        if kind == "file":
            size, digest = entry["size"], entry["sha256"]
            if type(size) is not int or size < 0 or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or entry["target"] is not None:
                raise ValueError("ShellSim snapshot file is malformed")
            file_bytes += size
        elif entry["size"] is not None or entry["sha256"] is not None or (
            kind == "directory" and entry["target"] is not None
        ) or (kind == "symlink" and not isinstance(entry["target"], str)):
            raise ValueError("ShellSim snapshot directory or symlink is malformed")
        paths.append(name)
        ordered.append({key: entry[key] for key in ("path", "kind", "mode", "size", "sha256", "target")})
    if paths[0] != "." or len(set(paths)) != len(paths) or paths[1:] != sorted(paths[1:]) or file_bytes != result["total_file_bytes"] or file_bytes > 64 * 1024 * 1024:
        raise ValueError("ShellSim snapshot inventory differs")
    wire = {"schema_version": snapshot["schema_version"], "root": snapshot["root"],
            "limits": {key: snapshot["limits"][key] for key in ("max_entries", "max_file_bytes", "max_response_bytes")},
            "entries": ordered}
    raw = json.dumps(wire, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    if result["snapshot_sha256"] != digest:
        raise ValueError("ShellSim snapshot canonical digest differs")
    return digest, nonce, pid, {entry["path"]: entry for entry in entries}


def _shellsim_stopped(root: Path, nonce: str) -> None:
    stop = _json(root / "session-stop.json")
    if stop != {"session_nonce": nonce, "process_exited": True}:
        raise ValueError("ShellSim reset session was not confirmed closed")


def shellsim_candidate_record(root: Path, seen: set[str], bridge_sha256: str) -> dict:
    """Attest a fresh, closed simulated candidate from retained bridge evidence."""
    digest, nonce, pid, _ = _shellsim_snapshot(root / "initial-snapshot.json")
    _shellsim_stopped(root, nonce)
    if nonce in seen or not re.fullmatch(r"[0-9a-f]{64}", bridge_sha256):
        raise ValueError("ShellSim candidate session or bridge identity differs")
    seen.add(nonce)
    return {
        "schema_version": "capability-shellsim-candidate-v1",
        "initial_snapshot_sha256": digest,
        "session_nonce": nonce,
        "bridge_pid": pid,
        "bridge_sha256": bridge_sha256,
        "process_exited": True,
    }


def _classify(attempt: Path, identity: dict) -> tuple[str, list[str], dict]:
    report = _json(attempt / "raw/report.json")
    if report.get("schema_version") != SCHEMA or report.get("source") != identity:
        raise ValueError("reset report source differs")
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 5:
        raise ValueError("reset report lacks five episodes")
    shellsim_digest = None
    session_nonces: set[str] = set()
    if identity["kind"] == "shellsim":
        baseline = report.get("baseline")
        if not isinstance(baseline, dict) or baseline.get("cycle") != 0 or baseline.get("trial") != "baseline":
            raise ValueError("ShellSim reset baseline identity differs")
        baseline_root = attempt / "raw/trials/baseline"
        if baseline.get("result_sha256") != sha256(baseline_root / "result.json") or baseline.get("exception") is not None:
            return "pending", ["ShellSim reset baseline trial is incomplete"], {"episodes": 5}
        baseline_result = _json(baseline_root / "result.json")
        if baseline_result.get("exception_info") is not None or any(
            not isinstance(step, dict) or step.get("exception_info") is not None
            for step in (baseline_result.get("step_results") or [])
        ):
            return "pending", ["ShellSim reset baseline trial failed"], {"episodes": 5}
        shellsim_digest, nonce, pid, initial = _shellsim_snapshot(baseline_root / "initial-snapshot.json")
        _shellsim_stopped(baseline_root, nonce)
        mutated_digest, mutated_nonce, mutated_pid, changed = _shellsim_snapshot(baseline_root / "mutated-snapshot.json")
        if (mutated_nonce, mutated_pid) != (nonce, pid) or mutated_digest == shellsim_digest or \
                "__capability_reset_marker" in initial or changed.get("__capability_reset_marker", {}).get("sha256") != hashlib.sha256(b"reset-mutation\n").hexdigest():
            return "pending", ["ShellSim calibration mutation was not demonstrated"], {"episodes": 5}
        session_nonces.add(nonce)
    elif report.get("baseline") is not None:
        raise ValueError("no-tool reset unexpectedly contains a baseline trial")
    names = set()
    failures = []
    vfs_failures = []
    for cycle, row in enumerate(episodes, 1):
        if not isinstance(row, dict) or row.get("cycle") != cycle or row.get("trial") != f"reset-{cycle:02d}":
            raise ValueError("reset episode identity differs")
        name = row["trial"]
        if name in names:
            raise ValueError("reset trial identity repeated")
        names.add(name)
        root = attempt / "raw/trials" / name
        if row.get("result_sha256") != sha256(root / "result.json"):
            raise ValueError("reset trial result changed")
        trial_result = _json(root / "result.json")
        if trial_result.get("exception_info") is not None:
            return "pending", [f"reset trial {name} failed before complete capture"], {"episodes": 5}
        steps = trial_result.get("step_results") or []
        if not isinstance(steps, list) or any(
            not isinstance(step, dict) or step.get("exception_info") is not None for step in steps
        ):
            return "pending", [f"reset trial {name} failed before complete capture"], {"episodes": 5}
        observed = []
        for index, step in enumerate(identity["step_names"]):
            path = root / "steps" / step / "agent/transcript.json" if len(identity["step_names"]) > 1 else root / "agent/transcript.json"
            turns = json.loads(path.read_text())
            users = [turn.get("content") for turn in turns if isinstance(turn, dict) and turn.get("role") == "user"]
            if len(users) != index + 1 or type(users[-1]) is not str:
                raise ValueError("reset trial transcript is incomplete")
            observed.append(hashlib.sha256(users[-1].encode()).hexdigest())
        if row.get("prompts_sha256") != observed:
            raise ValueError("reset episode summary differs from transcript")
        if row.get("exception") is not None:
            return "pending", [f"reset trial {name} failed before complete capture"], {"episodes": 5}
        if observed != identity["prompt_sha256"]:
            failures.append(cycle)
        if identity["kind"] == "shellsim":
            digest, nonce, _, entries = _shellsim_snapshot(root / "initial-snapshot.json")
            _shellsim_stopped(root, nonce)
            if nonce in session_nonces:
                raise ValueError("ShellSim reset session identity repeated")
            session_nonces.add(nonce)
            if "__capability_reset_marker" in entries or digest != shellsim_digest:
                vfs_failures.append(cycle)
    issues = []
    if failures:
        issues.append("fresh prompt differs from frozen export")
    if vfs_failures:
        issues.append("fresh ShellSim VFS differs from calibrated initial state")
    summary = {"episodes": 5, "mismatch_cycles": failures,
               "public_resources": "none" if identity["kind"] == "none" else "complete_shellsim_vfs"}
    if identity["kind"] == "shellsim":
        summary.update(baseline_snapshot_sha256=shellsim_digest, vfs_mismatch_cycles=vfs_failures,
                       fresh_sessions=len(session_nonces))
    return ("semantic_failed" if issues else "ready", issues, summary)


def run_frozen_non_docker_reset(
    item_root: Path, toolchain: Any, timeout: int, shellsim_bridge: Path | None = None,
    *, runner: Callable[..., int] | None = None,
) -> dict:
    """Run once, retain all raw episodes, and revalidate without resampling."""
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("reset timeout must be a positive integer")
    item = Path(item_root)
    try:
        identity, _ = _source(item, shellsim_bridge)
    except (OSError, ValueError, TypeError) as error:
        return _result("pending", [f"non-Docker reset input: {type(error).__name__}: {error}"])
    key = hashlib.sha256(json.dumps(_portable(identity), sort_keys=True).encode()).hexdigest()[:20]
    attempt = item / "diagnostics/non-docker-reset" / f"attempt-{key}"
    if not attempt.exists():
        try:
            _freeze(attempt, item, identity, toolchain, timeout, shellsim_bridge)
        except Exception as error:  # noqa: BLE001 - retain partial frozen attempt.
            return _result("pending", [f"reset input freeze: {type(error).__name__}: {error}"], attempt)
    try:
        binding = _json(attempt / "binding.json")
        frozen_identity = binding.get("source")
        if not isinstance(frozen_identity, dict) or _portable(frozen_identity) != _portable(identity):
            raise ValueError("frozen reset source bytes differ")
        _validate_frozen(attempt, frozen_identity)
    except Exception as error:  # noqa: BLE001 - immutable input failures remain pending.
        return _result("pending", [f"frozen reset input: {type(error).__name__}: {error}"], attempt)
    report = attempt / "raw/report.json"
    if (attempt / "raw").exists() or (attempt / "controller-run.log").exists():
        if not report.is_file():
            return _result("pending", ["incomplete reset attempt cannot be rerun"], attempt)
    else:
        try:
            (runner or _run_remote)(toolchain=toolchain, attempt=attempt, timeout=timeout)
        except Exception as error:  # noqa: BLE001 - retain runner failure without resampling.
            return _result("pending", [f"reset runner: {type(error).__name__}: {error}"], attempt)
    if not report.is_file():
        return _result("pending", ["reset report is absent"], attempt)
    try:
        if _portable(_source(item, shellsim_bridge)[0]) != _portable(frozen_identity):
            raise ValueError("reset source changed during attempt")
        if any(path.is_symlink() for path in attempt.rglob("*")):
            raise ValueError("reset artifacts contain a link")
        state, issues, summary = _classify(attempt, frozen_identity)
        inventory = {p.relative_to(attempt).as_posix(): sha256(p) for p in sorted(attempt.rglob("*"))
                     if p.is_file() and p.name not in {"artifacts.manifest.json", "summary.json"}}
        closure = {"schema_version": SCHEMA, "binding_sha256": sha256(attempt / "binding.json"), "files": inventory}
        closure_path = attempt / "artifacts.manifest.json"
        if closure_path.exists():
            if _json(closure_path) != closure:
                raise ValueError("reset artifact closure changed")
        else:
            _write(closure_path, closure)
    except Exception as error:  # noqa: BLE001 - raw evidence failure remains pending.
        return _result("pending", [f"reset artifact validation: {type(error).__name__}: {error}"], attempt)
    payload = _result(state, issues, attempt, summary)
    summary_path = attempt / "summary.json"
    if not summary_path.exists():
        _write(summary_path, {key: value for key, value in payload.items() if key != "extra_files"})
        payload["extra_files"][f"{PREFIX}/{attempt.name}/summary.json"] = summary_path
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner", action="store_true")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--binding-sha256")
    args = parser.parse_args(argv)
    if not args.inner or args.attempt is None or args.binding_sha256 is None:
        parser.error("only a frozen --inner invocation is supported")
    return _inner(args.attempt, args.binding_sha256)


if __name__ == "__main__":
    raise SystemExit(main())

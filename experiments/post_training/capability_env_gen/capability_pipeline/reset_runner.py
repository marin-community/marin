"""Frozen outer controller for task-bound reset diagnostics on remote Daytona."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import sandbox_provider
from .daytona_resources import DaytonaResourceProfile
from .reset_diagnostics import ResetDiagnosticsError, validate_policy_and_build_plan
from .runtime import sha256

SCHEMA = "capability-frozen-reset-wrapper-v1"
ATTEMPT_PREFIX = "reset_diagnostics"


def _json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or linked JSON artifact: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path.name}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")


def _tree_entries(root: Path) -> list[list[Any]]:
    """List file bytes, modes, and empty directories; reject links and devices."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("reset Harbor package must be an ordinary directory")
    entries: list[list[Any]] = []
    for path in (root, *sorted(root.rglob("*"))):
        mode = path.lstat().st_mode
        relative = "." if path == root else path.relative_to(root).as_posix()
        if stat.S_ISDIR(mode):
            entries.append([relative, "directory", stat.S_IMODE(mode)])
        elif stat.S_ISREG(mode):
            entries.append([relative, "file", stat.S_IMODE(mode), path.stat().st_size, sha256(path)])
        else:
            raise ValueError("reset Harbor package contains a linked or special member")
    return entries


def _entries_sha256(entries: list[list[Any]]) -> str:
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def _tree_sha256(root: Path) -> str:
    return _entries_sha256(_tree_entries(root))


def _file_tree_sha256(root: Path) -> str:
    entries = [
        [entry[0], entry[3], entry[4]]
        for entry in _tree_entries(root) if entry[1] == "file"
    ]
    return _entries_sha256(entries)


def _restore_tree(root: Path, manifest: dict, expected_sha256: str) -> None:
    """Restore only logical directory/mode metadata omitted by object storage."""
    entries = manifest.get("entries")
    if manifest.get("schema_version") != "capability-reset-input-tree-v1" or not isinstance(entries, list):
        raise ValueError("frozen reset tree manifest is invalid")
    if _entries_sha256(entries) != expected_sha256:
        raise ValueError("frozen reset tree manifest differs from source identity")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("frozen reset Harbor root is absent or linked")
    expected: dict[str, list] = {}
    for entry in entries:
        if not isinstance(entry, list) or len(entry) not in {3, 5}:
            raise ValueError("frozen reset tree member is malformed")
        relative, kind, mode = entry[:3]
        if (not isinstance(relative, str) or not relative or
                (relative != "." and (relative.startswith("/") or any(
                    part in {"", ".", ".."} for part in relative.split("/")
                ))) or relative in expected or type(mode) is not int or not 0 <= mode <= 0o7777):
            raise ValueError("frozen reset tree member path or mode is invalid")
        if (kind == "directory" and len(entry) != 3) or (kind == "file" and len(entry) != 5) or kind not in {"directory", "file"}:
            raise ValueError("frozen reset tree member kind is invalid")
        expected[relative] = entry
    if "." not in expected or expected["."][1] != "directory":
        raise ValueError("frozen reset tree lacks root directory")
    for relative, entry in sorted(expected.items(), key=lambda pair: (pair[0].count("/"), pair[0])):
        path = root if relative == "." else root.joinpath(*relative.split("/"))
        if path.is_symlink():
            raise ValueError("frozen reset tree contains a link")
        if entry[1] == "directory":
            if path.exists() and not path.is_dir():
                raise ValueError("frozen reset directory changed kind")
            path.mkdir(parents=True, exist_ok=True)
        else:
            if not path.is_file() or path.stat().st_size != entry[3] or sha256(path) != entry[4]:
                raise ValueError("frozen reset file bytes changed")
    actual = {"."} | {path.relative_to(root).as_posix() for path in root.rglob("*")}
    if actual != set(expected):
        raise ValueError("frozen reset tree has extra or missing members")
    for relative, entry in sorted(expected.items(), key=lambda pair: (pair[0].count("/"), pair[0]), reverse=True):
        path = root if relative == "." else root.joinpath(*relative.split("/"))
        os.chmod(path, entry[2])
    if _tree_sha256(root) != expected_sha256:
        raise ValueError("frozen reset tree failed logical restoration")


def _controller_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    names = (
        "capability_pipeline/reset_runner.py",
        "capability_pipeline/reset_diagnostics.py",
        "capability_pipeline/daytona_environment.py",
        "capability_pipeline/daytona_resources.py",
        "capability_pipeline/daytona_snapshot.py",
        "capability_pipeline/image_runtime_metadata.py",
        "capability_pipeline/provider_retry.py",
        "capability_pipeline/runtime.py",
        "capability_pipeline/synthesis.py",
        "scripts/run_reset_conformance.py",
    )
    return {name: sha256(root / name) for name in names}


def _source(item_root: Path, daytona_tools: Path | None) -> tuple[dict, dict]:
    harbor = item_root / "harbor"
    policy = item_root / "workspace/task/reset-policy.json"
    resources = item_root / "workspace/task/candidate-resources.json"
    helper = item_root / "workspace/tools/daytona/dt.py"
    if not helper.is_file() and daytona_tools is not None:
        helper = Path(daytona_tools) / "dt.py"
    if resources.is_symlink() or not resources.is_file():
        raise ResetDiagnosticsError("task/candidate-resources.json is required for reset capacity evidence")
    try:
        request = _json(resources)
    except (TypeError, ValueError) as error:
        raise ResetDiagnosticsError("task/candidate-resources.json is invalid") from error
    if set(request) != {"cpu", "memory_gb", "disk_gb"}:
        raise ResetDiagnosticsError("candidate-resources.json must contain cpu, memory_gb, disk_gb")
    try:
        profile = DaytonaResourceProfile(
            cpu=request["cpu"], memory_gb=request["memory_gb"], disk_gb=request["disk_gb"],
            source="adapter_kwargs",
        )
    except ValueError as error:
        raise ResetDiagnosticsError("task/candidate-resources.json has invalid capacity") from error
    if policy.is_symlink() or not policy.is_file():
        raise ResetDiagnosticsError("task/reset-policy.json is required for Docker reset evidence")
    try:
        _json(policy)
    except (TypeError, ValueError) as error:
        raise ResetDiagnosticsError("task/reset-policy.json is invalid") from error
    if helper.is_symlink() or not helper.is_file():
        raise ValueError("pinned Daytona dt.py helper is unavailable")
    source = {
        "harbor_tree_sha256": _tree_sha256(harbor),
        "harbor_file_sha256": _file_tree_sha256(harbor),
        "policy_sha256": sha256(policy),
        "candidate_resources_sha256": sha256(resources),
        "daytona_helper_sha256": sha256(helper),
    }
    paths = {"harbor": harbor, "policy": policy, "resources": resources, "helper": helper}
    return source, {"paths": paths, "resource_receipt": profile.receipt()}


def _attempt_key(source: dict) -> str:
    portable = {key: value for key, value in source.items() if key != "harbor_tree_sha256"}
    raw = json.dumps(portable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def _binding(source: dict, receipt: dict, timeout: int, toolchain: Any) -> dict:
    from .synthesis import SOURCE_LOCK

    return {
        "schema_version": SCHEMA,
        "source": source,
        "resource_receipt": receipt,
        "timeout_seconds": timeout,
        "taskcompendium_source_lock_sha256": sha256(SOURCE_LOCK),
        "controller": _controller_hashes(),
    }


def _freeze_inputs(attempt: Path, source: dict, details: dict, binding: dict) -> None:
    paths = details["paths"]
    attempt.mkdir(parents=True)
    inputs = attempt / "input"
    inputs.mkdir()
    shutil.copytree(paths["harbor"], inputs / "harbor", symlinks=False)
    shutil.copy2(paths["policy"], inputs / "reset-policy.json")
    shutil.copy2(paths["resources"], inputs / "candidate-resources.json")
    (inputs / "tools").mkdir()
    shutil.copy2(paths["helper"], inputs / "tools/dt.py")
    _write(inputs / "harbor-tree.json", {
        "schema_version": "capability-reset-input-tree-v1",
        "entries": _tree_entries(inputs / "harbor"),
    })
    _write(inputs / "resource-receipt.json", details["resource_receipt"])
    _write(attempt / "binding.json", binding)
    _restore_tree(inputs / "harbor", _json(inputs / "harbor-tree.json"), source["harbor_tree_sha256"])
    if _tree_sha256(inputs / "harbor") != source["harbor_tree_sha256"] or any(
        sha256(inputs / name) != source[label]
        for name, label in (
            ("reset-policy.json", "policy_sha256"),
            ("candidate-resources.json", "candidate_resources_sha256"),
            ("tools/dt.py", "daytona_helper_sha256"),
        )
    ):
        raise RuntimeError("reset inputs changed during immutable copy")


def _validate_frozen(attempt: Path, expected: dict) -> None:
    if attempt.is_symlink() or any(path.is_symlink() for path in attempt.rglob("*")):
        raise ValueError("frozen reset attempt contains a symlink")
    if _json(attempt / "binding.json") != expected:
        raise ValueError("frozen reset controller or source identity drifted")
    inputs = attempt / "input"
    source = expected["source"]
    _restore_tree(inputs / "harbor", _json(inputs / "harbor-tree.json"), source["harbor_tree_sha256"])
    if _tree_sha256(inputs / "harbor") != source["harbor_tree_sha256"] or any(
        sha256(inputs / name) != source[label]
        for name, label in (
            ("reset-policy.json", "policy_sha256"),
            ("candidate-resources.json", "candidate_resources_sha256"),
            ("tools/dt.py", "daytona_helper_sha256"),
        )
    ) or _json(inputs / "resource-receipt.json") != expected["resource_receipt"]:
        raise ValueError("frozen reset input bytes drifted")


def _inventory(attempt: Path) -> dict[str, str]:
    if any(path.is_symlink() for path in attempt.rglob("*")):
        raise ValueError("reset artifacts contain a symlink")
    return {
        path.relative_to(attempt).as_posix(): sha256(path)
        for path in sorted(attempt.rglob("*"))
        if path.is_file() and path.name not in {"artifacts.manifest.json", "summary.json"}
    }


def _seal_closure(attempt: Path) -> dict[str, str]:
    inventory = _inventory(attempt)
    closure_path = attempt / "artifacts.manifest.json"
    closure = {
        "schema_version": SCHEMA,
        "binding_sha256": sha256(attempt / "binding.json"),
        "files": inventory,
    }
    if closure_path.exists():
        if _json(closure_path) != closure:
            raise ValueError("retained reset artifact closure changed")
    else:
        _write(closure_path, closure)
    return inventory


def _stop(child: subprocess.Popen) -> None:
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def _default_runner(*, toolchain: Any, attempt: Path, timeout: int) -> int:
    if not sandbox_provider.credentials_present():
        raise RuntimeError("reset diagnostics require remote Daytona credentials")
    command = toolchain.runtime_command()
    index = command.index("python")
    command[index + 1:] = [
        "-m", "capability_pipeline.reset_runner", "--inner",
        "--attempt", str(attempt.resolve()),
        "--binding-sha256", sha256(attempt / "binding.json"),
    ]
    environment = dict(os.environ)
    environment["CAPABILITY_REMOTE_RESET"] = "1"
    environment["CAPABILITY_DAYTONA_TOOLS"] = str((attempt / "input/tools").resolve())
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get("PYTHONPATH", "")
    with (attempt / "controller-run.log").open("xb") as log:
        child = subprocess.Popen(command, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        finally:
            _stop(child)


def _inner(attempt: Path, binding_sha256: str) -> int:
    if os.environ.get("CAPABILITY_REMOTE_RESET") != "1" or not sandbox_provider.credentials_present():
        raise RuntimeError("reset execution is remote-only")
    attempt = attempt.resolve()
    if sha256(attempt / "binding.json") != binding_sha256:
        raise ValueError("frozen reset binding hash changed")
    binding = _json(attempt / "binding.json")
    from .synthesis import SOURCE_LOCK
    if binding.get("controller") != _controller_hashes() or binding.get("taskcompendium_source_lock_sha256") != sha256(SOURCE_LOCK):
        raise ValueError("reset controller source differs from frozen binding")
    _validate_frozen(attempt, binding)
    from .reset_diagnostics import run_reset_diagnostics

    inputs = attempt / "input"
    report = run_reset_diagnostics(
        inputs / "reset-policy.json", inputs / "harbor",
        inputs / "resource-receipt.json", attempt / "raw",
    )
    return 0 if report.get("reset_conformance") == "passed" else 2


def _cleanup_confirmed(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("verified_absent") is True
        and isinstance(value.get("observations"), list)
        and bool(value["observations"])
        and isinstance(value["observations"][-1], dict)
        and value["observations"][-1].get("state") == "not_found"
    )


def _observed_state(raw: dict, plan: Any, expected_files: dict[str, str]) -> dict:
    from scripts import run_reset_conformance as reset

    class Replay:
        def run(self, _sandbox, _command, _timeout):
            return {"exit": 0, "timed_out": False, "stdout": json.dumps(raw)}

    return reset._inspect(Replay(), object(), plan, expected_files)


def _binding_fingerprints(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "bundle_path", "binding_sha256", "inputs_sha256", "task_toml_sha256"
    }:
        raise ValueError("reset task binding has an invalid shape")
    return {key: value[key] for key in (
        "binding_sha256", "inputs_sha256", "task_toml_sha256"
    )}


def _classify_raw(attempt: Path, binding: dict) -> tuple[str, list[str], dict]:
    """Recompute baseline and five-cycle findings from retained probe evidence."""
    from scripts import run_reset_conformance as reset

    from .reset_diagnostics import _policy_observation

    raw = attempt / "raw"
    report = _json(raw / "report.json")
    baseline = _json(raw / "baseline.json")
    inputs = attempt / "input"
    preflight, additional_roots = validate_policy_and_build_plan(
        inputs / "reset-policy.json", inputs / "harbor", inputs / "resource-receipt.json"
    )
    if (
        report.get("schema_version") != "capability-reset-diagnostics-v1"
        or report.get("policy_sha256") != sha256(inputs / "reset-policy.json")
        or report.get("resource_receipt_sha256") != sha256(inputs / "resource-receipt.json")
        or _binding_fingerprints(report.get("task_binding")) != _binding_fingerprints(preflight["task_binding"])
        or report.get("requested_image") != preflight["image"]
        or report.get("requested_recipe_sha256") != preflight["recipe_sha256"]
        or report.get("requested_resource_profile") != preflight["resource_profile"]
        or report.get("unassessed_additional_roots") != list(additional_roots)
        or report.get("full_quality_reset_gate") != "unassessed"
        or report.get("baseline") != {"status": baseline.get("status"), "artifact": "baseline.json"}
    ):
        raise ValueError("reset report differs from frozen inputs or baseline")
    if not _cleanup_confirmed(baseline.get("cleanup")):
        return "pending", ["baseline sandbox deletion is unconfirmed"], {}
    if baseline.get("status") == "semantic_failed":
        reason = baseline.get("reason")
        if reason == "readiness_contract_failed":
            ready = baseline.get("readiness", {})
            if ready.get("timed_out") is not False or ready.get("exit") in (None, 0):
                raise ValueError("baseline readiness failure lacks raw exit evidence")
        elif reason == "baseline_policy_mismatch":
            observation = baseline.get("initial_observation")
            if not isinstance(observation, dict):
                raise ValueError("baseline policy mismatch lacks raw observation")
            checks = _policy_observation(observation, reset.validate_plan_payload(preflight))
            if checks != baseline.get("policy_checks") or (checks["process_matches"] and checks["inspector_environment_matches"]):
                raise ValueError("baseline policy mismatch differs from raw probe")
        elif reason == "mutation_contract_failed":
            mutation = baseline.get("mutation", {})
            observation = baseline.get("initial_observation")
            if (
                "mutation" not in _json(inputs / "reset-policy.json")
                or not isinstance(observation, dict)
                or not isinstance(mutation, dict)
                or mutation.get("timed_out") is not False
                or type(mutation.get("exit")) is not int
                or mutation["exit"] == 0
            ):
                raise ValueError("baseline mutation command failure lacks raw exit evidence")
            checks = _policy_observation(observation, reset.validate_plan_payload(preflight))
            if checks != baseline.get("policy_checks") or not checks["process_matches"] or not checks["inspector_environment_matches"]:
                raise ValueError("baseline pre-mutation checks differ from raw probe")
        elif reason == "mutation_did_not_change_public_state":
            mutation = baseline.get("mutation", {})
            initial = baseline.get("initial_observation")
            changed = baseline.get("mutation_observation")
            if (
                "mutation" not in _json(inputs / "reset-policy.json")
                or not isinstance(initial, dict)
                or not isinstance(changed, dict)
                or mutation.get("exit") != 0
                or mutation.get("timed_out") is not False
                or reset._hash_inventory(initial.get("files"), label="baseline initial", allow_empty=True)
                != reset._hash_inventory(changed.get("files"), label="baseline mutated", allow_empty=True)
            ):
                raise ValueError("baseline no-change claim differs from raw inventories")
            checks = _policy_observation(initial, reset.validate_plan_payload(preflight))
            if checks != baseline.get("policy_checks") or not checks["process_matches"] or not checks["inspector_environment_matches"]:
                raise ValueError("baseline pre-mutation checks differ from raw probe")
        elif reason == "task_startup_command_failed":
            if (
                baseline.get("error_type") != "TaskStartupError"
                or baseline.get("startup_phase") not in {"mkdir", "additional_directory", "setup_command"}
                or type(baseline.get("startup_exit")) is not int
                or baseline["startup_exit"] in {0, 124}
                or baseline.get("task_startup") == "completed"
            ):
                raise ValueError("baseline task startup failure lacks structured exit evidence")
        elif reason == "invalid_baseline_public_state":
            # The baseline adapter does not retain the rejected raw probe.
            # Its label alone cannot spend a semantic repair attempt.
            return "pending", ["rejected baseline probe bytes were not retained"], {}
        else:
            raise ValueError("unrecognized baseline semantic failure")
        return "semantic_failed", [f"reset baseline: {reason}"], {"baseline": reason}
    if baseline.get("status") != "complete":
        return "pending", ["reset baseline did not complete"], {}

    plan_payload = _json(raw / "plan.json")
    if plan_payload.get("schema_version") != "capability-reset-conformance-plan-v2":
        raise ValueError("reset plan is not task-bound")
    if _binding_fingerprints(plan_payload.get("task_binding")) != _binding_fingerprints(preflight["task_binding"]):
        raise ValueError("reset plan task binding differs")
    for key, value in preflight.items():
        if key in {"public_files", "mutation", "task_binding"}:
            continue
        if plan_payload.get(key) != value:
            raise ValueError(f"reset plan {key} differs from frozen preflight")
    policy = _json(inputs / "reset-policy.json")
    if "mutation" in policy:
        if not isinstance(plan_payload.get("mutation"), dict) or any(
            plan_payload["mutation"].get(key) != value for key, value in policy["mutation"].items()
        ):
            raise ValueError("reset mutation plan differs from policy")
    elif "mutation" in plan_payload:
        raise ValueError("reset plan added an unrequested mutation")
    validated_payload = json.loads(json.dumps(plan_payload))
    validated_payload["task_binding"]["bundle_path"] = str((inputs / "harbor").resolve())
    plan = reset.validate_plan_payload(validated_payload)
    initial_raw = baseline.get("initial_observation")
    if not isinstance(initial_raw, dict) or reset._hash_inventory(
        initial_raw.get("files"), label="baseline public files", allow_empty=True
    ) != plan.public_files:
        raise ValueError("baseline public inventory differs from frozen reset plan")
    checks = _policy_observation(initial_raw, plan)
    if checks != baseline.get("policy_checks") or not checks["process_matches"] or not checks["inspector_environment_matches"]:
        raise ValueError("baseline policy checks differ from raw probe")
    if baseline.get("readiness", {}).get("exit") != 0 or baseline.get("readiness", {}).get("timed_out") is not False:
        raise ValueError("completed baseline lacks successful readiness")
    if "mutation" in policy:
        mutated = baseline.get("mutation_observation")
        if not isinstance(mutated, dict) or reset._hash_inventory(
            mutated.get("files"), label="baseline mutated files", allow_empty=True
        ) != plan.mutation_files or mutated["files"] == initial_raw["files"]:
            raise ValueError("baseline mutation differs from raw probe")
        if baseline.get("mutation", {}).get("exit") != 0 or baseline.get("mutation", {}).get("timed_out") is not False:
            raise ValueError("baseline mutation command did not complete")

    five = _json(raw / "five-cycle/report.json")
    if (
        five.get("schema_version") != "capability-reset-conformance-report-v1"
        or five.get("plan_sha256") != sha256(raw / "plan.json")
        or five.get("controller_sha256") != binding["controller"]["scripts/run_reset_conformance.py"]
        or five.get("requested_image") != plan.image
        or five.get("requested_recipe_sha256") != plan.recipe_sha256
        or five.get("requested_resource_profile") != plan.profile.receipt()
        or five.get("cycles_required") != 5
        or five.get("startup_scope") != "task_bound"
        or five.get("task_binding_sha256") != plan.task_binding_sha256
        or five.get("task_inputs_sha256") != plan.task_inputs_sha256
        or five.get("task_toml_sha256") != plan.task_toml_sha256
        or five.get("task_bundle_unchanged_at_finish") is not True
    ):
        raise ValueError("five-cycle report differs from the bound plan")
    snapshot = five.get("snapshot", {})
    if snapshot.get("status") != "ready" or not isinstance(snapshot.get("name"), str):
        return "pending", ["reset snapshot was unavailable"], {}
    if baseline.get("snapshot") != snapshot["name"]:
        raise ValueError("baseline and cycle snapshots differ")
    trace_path = raw / "five-cycle-probes.jsonl"
    if trace_path.is_symlink() or not trace_path.is_file():
        raise ValueError("five-cycle raw probe trace is absent")
    traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
    if not all(isinstance(line, dict) for line in traces):
        raise ValueError("five-cycle raw trace contains invalid rows")
    by_sandbox: dict[str, list[dict]] = {}
    for trace in traces:
        sandbox_id = trace.get("sandbox_id")
        if not isinstance(sandbox_id, str) or not sandbox_id:
            raise ValueError("five-cycle trace lacks sandbox ID")
        if trace.get("exit") != 0 or trace.get("timed_out") is not False or not isinstance(trace.get("probe"), dict):
            return "pending", ["five-cycle raw probe transport is incomplete"], {}
        by_sandbox.setdefault(sandbox_id, []).append(trace)
    cycles = five.get("cycles")
    if not isinstance(cycles, list) or len(cycles) != 5 or [c.get("cycle") for c in cycles if isinstance(c, dict)] != list(range(1, 6)):
        return "pending", ["five-cycle denominator is incomplete"], {}
    baseline_id = baseline.get("sandbox_id")
    ids = [cycle.get("sandbox_id") for cycle in cycles]
    if not isinstance(baseline_id, str) or not baseline_id or any(
        not isinstance(value, str) or not value for value in ids
    ) or len(set(ids + [baseline_id])) != 6 or set(by_sandbox) != set(ids):
        raise ValueError("reset sandbox identities or probe inventory differ")
    semantic_failures = []
    for cycle in cycles:
        if cycle.get("snapshot") != snapshot["name"] or cycle.get("task_startup") != {"status": "completed"}:
            return "pending", ["cycle task startup is incomplete"], {}
        if not _cleanup_confirmed(cycle.get("cleanup")):
            return "pending", ["cycle sandbox deletion is unconfirmed"], {}
        if cycle["cleanup"].get("succeeded") is not True:
            return "pending", ["cycle sandbox cleanup is incomplete"], {}
        observed = by_sandbox[cycle["sandbox_id"]]
        if len(observed) != (2 if "mutation" in policy else 1):
            raise ValueError("cycle probe count differs from frozen policy")
        computed_initial = _observed_state(observed[0].get("probe"), plan, plan.public_files)
        if computed_initial != cycle.get("initial_state"):
            raise ValueError("cycle initial state differs from raw probe")
        ready = cycle.get("readiness", {})
        successful = ready.get("exit") == 0 and ready.get("timed_out") is False and computed_initial["status"] == "passed"
        if "mutation" in policy:
            computed_mutation = _observed_state(observed[1].get("probe"), plan, plan.mutation_files or {})
            mutation = cycle.get("mutation", {})
            if computed_mutation != mutation.get("state"):
                raise ValueError("cycle mutation state differs from raw probe")
            successful = successful and mutation.get("exit") == 0 and mutation.get("timed_out") is False and computed_mutation["status"] == "passed"
        if cycle.get("status") != ("passed" if successful else "failed"):
            raise ValueError("cycle status differs from raw checks")
        if not successful:
            semantic_failures.append(cycle["cycle"])
    recomputed = "passed" if not semantic_failures else "failed"
    if five.get("conformance") != recomputed or report.get("reset_conformance") != ("passed" if recomputed == "passed" else "semantic_failed"):
        raise ValueError("reset summary differs from raw cycle evidence")
    if report.get("five_cycle") != {"artifact": "five-cycle/report.json", "conformance": recomputed}:
        raise ValueError("reset report cycle reference differs")
    summary = {"cycles": 5, "baseline_sandbox_id": baseline_id,
               "cycle_sandbox_ids": ids, "semantic_failure_cycles": semantic_failures,
               "additional_roots": list(additional_roots)}
    if semantic_failures:
        return "semantic_failed", ["reset cycle state differs from baseline"], summary
    if additional_roots:
        return "pending", ["additional task roots remain unassessed"], summary
    return "ready", [], summary


def _result(
    state: str, issues: list[str], *, attempt: Path | None = None,
    summary: dict | None = None,
) -> dict:
    files: dict[str, Path] = {}
    if attempt is not None and attempt.is_dir():
        files = {
            f"{ATTEMPT_PREFIX}/{attempt.name}/{path.relative_to(attempt).as_posix()}": path
            for path in sorted(attempt.rglob("*")) if path.is_file()
        }
    return {
        "schema_version": SCHEMA,
        "state": state,
        "reviewable": state == "semantic_failed",
        "issues": issues,
        "summary": summary or {},
        "full_quality_reset_gate": "unassessed",
        "private_outside_public_root": "unassessed",
        "provider_network_blocking": "requested_not_proven",
        "attempt": str(attempt) if attempt is not None else None,
        "extra_files": files,
    }


def run_frozen_reset(
    item_root: Path, toolchain: Any, timeout: int,
    daytona_tools: Path | None,
    *, runner: Callable[..., int] | None = None,
) -> dict:
    """Run or validate one immutable task reset attempt without resampling it."""
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("reset timeout must be a positive integer")
    item_root = Path(item_root)
    if item_root.is_symlink() or not item_root.is_dir():
        return _result("pending", ["reset item root is absent or linked"])
    try:
        source, details = _source(item_root, daytona_tools)
    except (ResetDiagnosticsError, FileNotFoundError) as error:
        if "reset-policy.json" in str(error) or "candidate-resources.json" in str(error):
            return _result("semantic_failed", [str(error)])
        return _result("pending", [f"reset input: {type(error).__name__}"])
    except (ValueError, TypeError, OSError) as error:
        return _result("pending", [f"reset input: {type(error).__name__}"])
    binding = _binding(source, details["resource_receipt"], timeout, toolchain)
    attempt = item_root / "diagnostics/reset" / f"attempt-{_attempt_key(source)}"
    if not attempt.exists():
        try:
            _freeze_inputs(attempt, source, details, binding)
        except Exception as error:  # noqa: BLE001 - retain partial attempt; no resample.
            return _result("pending", [f"reset input freeze: {type(error).__name__}"], attempt=attempt)
    try:
        _validate_frozen(attempt, binding)
        validate_policy_and_build_plan(
            attempt / "input/reset-policy.json", attempt / "input/harbor",
            attempt / "input/resource-receipt.json",
        )
    except ResetDiagnosticsError as error:
        return _result("semantic_failed", [f"reset authoring: {error}"], attempt=attempt)
    except Exception as error:  # noqa: BLE001 - immutable/transport failures are pending.
        return _result("pending", [f"frozen reset input: {type(error).__name__}"], attempt=attempt)
    raw_report = attempt / "raw/report.json"
    marker = attempt / "run-started.json"
    if marker.exists() or (attempt / "raw").exists() or (attempt / "controller-run.log").exists():
        try:
            if _json(marker) != {"schema_version": SCHEMA, "binding_sha256": sha256(attempt / "binding.json")}:
                raise ValueError("reset started marker differs from binding")
            _seal_closure(attempt)
        except Exception as error:  # noqa: BLE001 - partial retained closure is pending.
            return _result("pending", [f"reset partial artifact integrity: {type(error).__name__}"], attempt=attempt)
        if not raw_report.is_file():
            return _result("pending", ["incomplete reset attempt cannot be rerun"], attempt=attempt)
    else:
        try:
            _write(marker, {"schema_version": SCHEMA, "binding_sha256": sha256(attempt / "binding.json")})
            (runner or _default_runner)(toolchain=toolchain, attempt=attempt, timeout=timeout)
        except Exception as error:  # noqa: BLE001 - provider or timeout evidence is pending.
            issues = [f"reset runner: {type(error).__name__}"]
            try:
                _seal_closure(attempt)
            except Exception as closure_error:  # noqa: BLE001 - preserve both failures.
                issues.append(f"partial reset closure: {type(closure_error).__name__}")
            return _result("pending", issues, attempt=attempt)
    if not raw_report.is_file():
        issues = ["reset report is absent"]
        try:
            _seal_closure(attempt)
        except Exception as error:  # noqa: BLE001 - preserve incomplete evidence.
            issues.append(f"partial reset closure: {type(error).__name__}")
        return _result("pending", issues, attempt=attempt)
    try:
        if _source(item_root, daytona_tools)[0] != source:
            raise ValueError("reset authoring source drifted during the attempt")
        _validate_frozen(attempt, binding)
        inventory = _seal_closure(attempt)
        state, issues, summary = _classify_raw(attempt, binding)
        if _inventory(attempt) != inventory:
            raise ValueError("reset artifacts changed while validating")
    except Exception as error:  # noqa: BLE001 - raw evidence drift is pending.
        return _result("pending", [f"reset artifact validation: {type(error).__name__}: {error}"], attempt=attempt)
    payload = _result(state, issues, attempt=attempt, summary=summary)
    summary_path = attempt / "summary.json"
    if not summary_path.exists():
        _write(summary_path, {key: value for key, value in payload.items() if key != "extra_files"})
        payload["extra_files"][f"{ATTEMPT_PREFIX}/{attempt.name}/summary.json"] = summary_path
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner", action="store_true")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--binding-sha256")
    args = parser.parse_args(argv)
    if not args.inner or args.attempt is None or args.binding_sha256 is None:
        parser.error("reset_runner CLI only supports a frozen remote --inner invocation")
    return _inner(args.attempt, args.binding_sha256)


if __name__ == "__main__":
    raise SystemExit(main())

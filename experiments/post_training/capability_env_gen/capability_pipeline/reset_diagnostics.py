"""Freeze a task-bound reset baseline and run five fresh Daytona cycles."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from scripts import run_reset_conformance as reset

from .daytona_resources import profile_from_receipt
from .image_runtime_metadata import derive_daytona_recipe
from .inference import atomic_json

POLICY_SCHEMA = "capability-reset-policy-v1"
REPORT_SCHEMA = "capability-reset-diagnostics-v1"


class ResetDiagnosticsError(ValueError):
    """A proposed reset policy or frozen input is unsuitable for this task."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ResetDiagnosticsError(f"required reset input missing or linked: {path.name}")
    try:
        return json.loads(path.read_bytes())
    except ValueError as error:
        raise ResetDiagnosticsError(f"invalid reset JSON input: {path.name}") from error


def _policy(value: Any, *, workdir: str) -> dict[str, Any]:
    required = frozenset({"schema_version", "public_root", "readiness", "process_policy",
                          "environment_name_policy"})
    if not isinstance(value, Mapping) or frozenset(value) not in {required, required | {"mutation"}}:
        raise ResetDiagnosticsError("reset policy has an invalid shape")
    if value.get("schema_version") != POLICY_SCHEMA:
        raise ResetDiagnosticsError("reset policy schema is unsupported")
    if value.get("public_root") != workdir:
        raise ResetDiagnosticsError("reset policy public_root must equal the bound workdir")
    if "mutation" in value:
        mutation = value["mutation"]
        if not isinstance(mutation, Mapping) or set(mutation) != {"command", "timeout_seconds"}:
            raise ResetDiagnosticsError("reset mutation policy has an invalid shape")
        reset._command(mutation["command"], label="mutation command")
        reset._positive_int(mutation["timeout_seconds"], label="mutation timeout_seconds")
    # Let the existing plan validator enforce exact command/name-policy syntax.
    return dict(value)


def _package_inputs(package: Path) -> tuple[str, str, dict[str, str], tuple[str, ...]]:
    if not package.is_absolute() or any(path.is_symlink() for path in (package, *package.parents)):
        raise ResetDiagnosticsError("Harbor package must be an ordinary absolute path")
    binding = _json(package / "binding.json")
    environment = binding.get("environment") if isinstance(binding, dict) else None
    if not isinstance(environment, dict) or environment.get("kind") != "docker":
        raise ResetDiagnosticsError("reset diagnostics require a Docker task binding")
    image, workdir = environment.get("image"), environment.get("workdir", "/app")
    if not isinstance(image, str) or not isinstance(workdir, str):
        raise ResetDiagnosticsError("Docker task binding lacks image or workdir")
    additional = environment.get("additional_directories", [])
    if not isinstance(additional, list) or any(not isinstance(path, str) for path in additional):
        raise ResetDiagnosticsError("Docker additional directories are malformed")
    inputs = package / "environment" / "inputs"
    fingerprints = {
        "binding_sha256": _sha(package / "binding.json"),
        "task_toml_sha256": _sha(package / "task.toml"),
        "inputs_sha256": reset._inputs_sha256(inputs),
    }
    return image, workdir, fingerprints, tuple(additional)


def validate_policy_and_build_plan(
    policy_path: Path, harbor_package: Path, resource_receipt: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Read-only plan preflight; the baseline fills public file inventories."""
    policy_path, harbor_package, resource_receipt = map(Path, (
        policy_path, harbor_package, resource_receipt,
    ))
    image, workdir, fingerprints, additional = _package_inputs(harbor_package)
    policy = _policy(_json(policy_path), workdir=workdir)
    profile = profile_from_receipt(_json(resource_receipt))
    if profile.source == "compatible_default_no_pinned_profile":
        raise ResetDiagnosticsError("reset requires an explicit requested resource profile")
    recipe = derive_daytona_recipe(image)
    plan = {
        "schema_version": "capability-reset-conformance-plan-v2",
        "image": image,
        "recipe": recipe,
        "recipe_sha256": hashlib.sha256(recipe.encode()).hexdigest(),
        "resource_profile": profile.receipt(),
        "public_root": workdir,
        "public_files": {},
        "process_policy": policy["process_policy"],
        "environment_name_policy": policy["environment_name_policy"],
        "readiness": policy["readiness"],
        "cycles": 5,
        "task_binding": {"bundle_path": str(harbor_package), **fingerprints},
    }
    reset.validate_plan_payload(plan)
    root = PurePosixPath(workdir)
    unassessed = tuple(directory for directory in additional
                       if PurePosixPath(directory) != root and
                       root not in PurePosixPath(directory).parents)
    return plan, unassessed


def _probe(adapter: Any, sandbox: Any, root: str) -> dict[str, Any]:
    result = adapter.run(sandbox, reset._probe_command(root), 60)
    if result.get("timed_out") is True or result.get("exit") != 0:
        raise RuntimeError("reset public-state probe failed")
    try:
        raw = json.loads(result.get("stdout", ""))
    except (TypeError, ValueError) as error:
        raise RuntimeError("reset public-state probe returned invalid JSON") from error
    if not isinstance(raw, dict) or set(raw) != {
        "files", "symlinks", "process_comm_counts", "environment_names"
    }:
        raise RuntimeError("reset public-state probe returned an invalid shape")
    files = reset._hash_inventory(raw["files"], label="baseline public files", allow_empty=True)
    if raw["symlinks"] != []:
        raise ResetDiagnosticsError("baseline public workdir contains symlinks")
    if not isinstance(raw["process_comm_counts"], dict):
        raise TypeError("baseline process inventory is malformed")
    reset._name_set(raw["environment_names"], label="baseline inspector environment")
    return {"files": files, "raw": raw}


def _policy_observation(raw: dict[str, Any], plan: reset.ResetPlan) -> dict[str, Any]:
    counts = raw["process_comm_counts"]
    if any(not isinstance(name, str) or not name or type(count) is not int or count <= 0
           for name, count in counts.items()):
        raise RuntimeError("baseline process inventory is malformed")
    names = set(raw["environment_names"])
    process_ok = (set(counts) <= plan.process_allowed and
                  sum(counts.values()) <= plan.process_max_count)
    environment_ok = (names <= plan.environment_allowed and
                      plan.environment_required <= names and
                      not names & plan.environment_forbidden)
    return {"process_matches": process_ok, "inspector_environment_matches": environment_ok,
            "process_comm_counts": counts, "inspector_environment_names": sorted(names)}


class _TraceAdapter:
    """Retain the full public probe response for each five-cycle observation."""

    def __init__(self, inner: Any, path: Path):
        self.inner, self.path = inner, path

    def ensure_snapshot(self, plan):
        return self.inner.ensure_snapshot(plan)

    def create_sandbox(self, snapshot):
        return self.inner.create_sandbox(snapshot)

    def start_task(self, sandbox, plan):
        return self.inner.start_task(sandbox, plan)

    def run(self, sandbox, command, timeout):
        result = self.inner.run(sandbox, command, timeout)
        if command.startswith("python3 -I -c "):
            try:
                raw = json.loads(result.get("stdout", ""))
            except (TypeError, ValueError):
                raw = None
            trace = {
                "sandbox_id": getattr(sandbox, "id", None),
                "exit": result.get("exit"),
                "timed_out": result.get("timed_out") is True,
                "probe": raw if isinstance(raw, dict) and set(raw) == {
                    "files", "symlinks", "process_comm_counts", "environment_names"
                } else None,
            }
            with self.path.open("a") as stream:
                stream.write(json.dumps(trace, sort_keys=True) + "\n")
        return result

    def delete(self, sandbox):
        return self.inner.delete(sandbox)


def _cycle_classification(report: dict[str, Any]) -> str:
    if report.get("conformance") == "passed":
        return "passed"
    if report.get("task_bundle_unchanged_at_finish") is False:
        return "pending_infrastructure"
    cycles = report.get("cycles")
    if not isinstance(cycles, list) or len(cycles) != 5:
        return "pending_infrastructure"
    for cycle in cycles:
        if cycle.get("cleanup", {}).get("verified_absent") is not True:
            return "pending_infrastructure"
        if cycle.get("task_startup", {}).get("status") != "completed":
            return "pending_infrastructure"
        if cycle.get("readiness", {}).get("timed_out") is True:
            return "pending_infrastructure"
        state = cycle.get("initial_state", {})
        if state.get("status") not in {"passed", "failed"}:
            return "pending_infrastructure"
        mutation = cycle.get("mutation")
        if isinstance(mutation, dict) and mutation.get("timed_out") is True:
            return "pending_infrastructure"
        if isinstance(mutation, dict) and mutation.get("state", {}).get("status") not in {"passed", "failed"}:
            return "pending_infrastructure"
    return "semantic_failed"


def run_reset_diagnostics(
    policy_path: Path, harbor_package: Path, resource_receipt: Path, output: Path,
    *, adapter: Any | None = None,
) -> dict[str, Any]:
    """Freeze one baseline, delete it, then compare five fresh task starts."""
    plan_payload, unassessed_roots = validate_policy_and_build_plan(
        policy_path, harbor_package, resource_receipt
    )
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("reset diagnostics output already exists")
    output.mkdir(parents=True)
    policy = _json(Path(policy_path))
    adapter = adapter or reset.DaytonaResetAdapter()
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "policy_sha256": _sha(Path(policy_path)),
        "resource_receipt_sha256": _sha(Path(resource_receipt)),
        "task_binding": plan_payload["task_binding"],
        "requested_image": plan_payload["image"],
        "requested_recipe_sha256": plan_payload["recipe_sha256"],
        "requested_resource_profile": plan_payload["resource_profile"],
        "unassessed_additional_roots": list(unassessed_roots),
        "environment_name_scope": "remote inspector process only",
        "private_outside_public_root": "unassessed",
        "provider_network_blocking": "requested_not_proven",
        "full_quality_reset_gate": "unassessed",
        "reset_conformance": "pending_infrastructure",
    }
    plan = reset.validate_plan_payload(plan_payload)
    sandbox = None
    baseline: dict[str, Any] = {"status": "running"}
    try:
        snapshot = adapter.ensure_snapshot(plan)
        baseline["snapshot"] = snapshot
        sandbox, attempts = adapter.create_sandbox(snapshot)
        baseline["sandbox_id"] = getattr(sandbox, "id", None)
        baseline["provisioning_attempts"] = attempts
        adapter.start_task(sandbox, plan)
        baseline["task_startup"] = "completed"
        readiness = reset._run_exit(adapter, sandbox, plan.readiness_command,
                                    plan.readiness_timeout_seconds)
        baseline["readiness"] = readiness
        if readiness["timed_out"] is True:
            raise RuntimeError("baseline readiness timed out")
        if readiness["exit"] != 0:
            baseline.update(status="semantic_failed", reason="readiness_contract_failed")
        else:
            observed = _probe(adapter, sandbox, plan.public_root)
            baseline["initial_observation"] = observed["raw"]
            checks = _policy_observation(observed["raw"], plan)
            baseline["policy_checks"] = checks
            if not checks["process_matches"] or not checks["inspector_environment_matches"]:
                baseline.update(status="semantic_failed", reason="baseline_policy_mismatch")
            else:
                plan_payload["public_files"] = observed["files"]
                if "mutation" in policy:
                    command = reset._command(policy["mutation"].get("command"), label="mutation command")
                    timeout = reset._positive_int(policy["mutation"].get("timeout_seconds"), label="mutation timeout_seconds")
                    changed = reset._run_exit(adapter, sandbox, command, timeout)
                    baseline["mutation"] = changed
                    if changed["timed_out"] is True:
                        raise RuntimeError("baseline mutation timed out")
                    if changed["exit"] != 0:
                        baseline.update(status="semantic_failed", reason="mutation_contract_failed")
                    else:
                        mutated = _probe(adapter, sandbox, plan.public_root)
                        baseline["mutation_observation"] = mutated["raw"]
                        mutated_checks = _policy_observation(mutated["raw"], plan)
                        baseline["mutation_policy_checks"] = mutated_checks
                        if (not mutated_checks["process_matches"] or
                                not mutated_checks["inspector_environment_matches"]):
                            baseline.update(status="semantic_failed", reason="mutation_policy_mismatch")
                        elif mutated["files"] == observed["files"]:
                            baseline.update(status="semantic_failed", reason="mutation_did_not_change_public_state")
                        else:
                            plan_payload["mutation"] = {**policy["mutation"],
                                                        "expected_files": mutated["files"]}
                if baseline["status"] == "running":
                    baseline["status"] = "complete"
    except ResetDiagnosticsError as error:
        baseline.update(status="semantic_failed", error_type=type(error).__name__,
                        reason="invalid_baseline_public_state")
    except Exception as error:  # noqa: BLE001 - provider boundary, redact message
        code = getattr(error, "return_code", None)
        phase = getattr(error, "phase", None)
        if (phase in {"mkdir", "additional_directory", "setup_command"}
                and type(code) is int and code not in {0, 124}):
            baseline.update(status="semantic_failed", error_type=type(error).__name__,
                            reason="task_startup_command_failed", startup_phase=phase,
                            startup_exit=code)
        else:
            baseline.update(status="pending_infrastructure", error_type=type(error).__name__)
    finally:
        if sandbox is None:
            baseline["cleanup"] = {"attempted": False, "verified_absent": False}
        else:
            try:
                deletion = adapter.delete(sandbox)
                baseline["cleanup"] = deletion
                if not isinstance(deletion, Mapping) or deletion.get("verified_absent") is not True:
                    baseline["status"] = "pending_infrastructure"
            except Exception as error:  # noqa: BLE001 - provider boundary, redact message
                baseline["cleanup"] = {"attempted": True, "verified_absent": False,
                                       "error_type": type(error).__name__}
                baseline["status"] = "pending_infrastructure"
        atomic_json(output / "baseline.json", baseline)
    report["baseline"] = {"status": baseline["status"], "artifact": "baseline.json"}
    if baseline["status"] != "complete":
        report["reset_conformance"] = baseline["status"]
        atomic_json(output / "report.json", report)
        return report
    try:
        frozen = reset.validate_plan_payload(plan_payload)
    except Exception as error:  # noqa: BLE001 - changed local inputs stay pending
        report["reset_conformance"] = "pending_infrastructure"
        report["input_error_type"] = type(error).__name__
        atomic_json(output / "report.json", report)
        return report
    atomic_json(output / "plan.json", plan_payload)
    trace = _TraceAdapter(adapter, output / "five-cycle-probes.jsonl")
    cycles = reset.run_plan(frozen, _sha(output / "plan.json"), output / "five-cycle",
                            adapter=trace)
    report["five_cycle"] = {"artifact": "five-cycle/report.json",
                            "conformance": cycles["conformance"]}
    report["reset_conformance"] = _cycle_classification(cycles)
    report["baseline_public_files"] = len(frozen.public_files)
    report["baseline_probes_artifact"] = "five-cycle-probes.jsonl"
    report["finished_unix"] = time.time()
    atomic_json(output / "report.json", report)
    return report

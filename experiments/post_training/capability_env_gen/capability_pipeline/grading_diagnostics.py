"""Frozen fixed-response grading diagnostics for supported evaluation bundles."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from capability_pipeline import regrade, sandbox_provider
from capability_pipeline.runtime import (
    provider_isolation_record,
    sha256,
    verifier_isolation_record,
)

DIAGNOSTIC_SCHEMA = "capability-grading-diagnostics-v1"


def _controller_identity() -> dict[str, str]:
    from capability_pipeline.evaluation import controller_hashes

    return {
        **controller_hashes(),
        "capability_pipeline/grading_diagnostics.py": sha256(Path(__file__)),
    }


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _source_identity(source: Path) -> dict:
    from scripts import restore_bundle

    restore_bundle.evaluation_members(source)
    manifest = source / "manifest.json"
    return {
        "evaluation_manifest_sha256": sha256(manifest),
        "evaluation_plan_sha256": sha256(source / "plan.json"),
    }


def _unsupported(source: Path) -> str | None:
    """Return an explicit unassessed reason without writing a partial plan."""
    try:
        _, _, binding_kind = regrade._runtime_fields(source / "bundle")
        regrade.validate_controls(
            _read_json(source / "bundle/controls.json"), binding_kind=binding_kind
        )
    except Exception as error:  # noqa: BLE001 - unsupported inputs are unassessed.
        return str(error)
    return None


def _binding(
    source: Path, plan_bundle: Path, plan: dict, timeout_seconds: int, parallelism: int
) -> dict:
    return {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "source": _source_identity(source),
        "plan_bundle_manifest_sha256": sha256(plan_bundle / "manifest.json"),
        "plan_sha256": sha256(plan_bundle / "plan.json"),
        "plan_identities": plan["identities"],
        "candidate_resource_request": plan.get("candidate_resource_request"),
        "parallelism": parallelism,
        "timeout_seconds": timeout_seconds,
        "controller": _controller_identity(),
    }


def _write(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _inventory(root: Path) -> dict[str, str]:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("retained output contains a symlink")
    return {
        path.relative_to(root.parent).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _artifact_issues(
    plan: dict, report: dict, output: Path, controls: dict
) -> list[str]:
    """Verify raw trial artifacts; report summary/state fields are untrusted."""
    if plan.get("grading_strategy") == "capture_once":
        from .captured_grading_diagnostics import artifact_issues

        return artifact_issues(plan, report, output, controls)
    issues: list[str] = []
    cells = report.get("cells")
    if not isinstance(cells, list):
        return ["regrade report lacks raw cells"]
    planned_cells = {cell["trial_name"]: cell for cell in plan["cells"]}
    control_cases = {case["id"]: case for case in controls["cases"]}
    native = plan.get("verifier_surface") == "native_deterministic"
    planned = set(planned_cells)
    names = [row.get("trial_name") for row in cells if isinstance(row, dict)]
    if (
        len(cells) != len(plan["cells"])
        or set(names) != planned
        or len(set(names)) != len(names)
    ):
        issues.append("regrade cell inventory differs from frozen plan")
    candidate_ids: set[str] = set()
    verifier_ids: set[str] = set()
    actual_rows: list[tuple[dict, Path, dict]] = []
    for row in cells:
        if not isinstance(row, dict) or not isinstance(row.get("trial_name"), str):
            continue
        if row["trial_name"] not in planned:
            issues.append("raw cell path is outside frozen plan")
            continue
        expected_cell = planned_cells[row["trial_name"]]
        expected = control_cases[expected_cell["case_id"]]["expect"]
        extraction = (
            row.get("status") == "extraction_error"
            and (native or expected["status"] == "extraction_error")
        )
        exception = row.get("exception")
        if (
            any(
                row.get(key) != expected_cell[key]
                for key in ("ordinal", "case_id", "repeat", "trial_name")
            )
            or row.get("outcome_class")
            != ("extraction_error" if extraction else "graded")
            or (not extraction and exception is not None)
            or (
                extraction
                and (
                    not isinstance(exception, dict)
                    or exception.get("type") != "ExtractionError"
                    or row.get("reward") is not None
                    or row.get("verifier_result_present") is not False
                )
            )
        ):
            issues.append(
                f"{row['trial_name']}: outcome differs from frozen completed cell"
            )
        root = output / "runtime-trials" / row["trial_name"]
        if root.is_symlink() or not root.resolve().is_relative_to(
            (output / "runtime-trials").resolve()
        ):
            issues.append(f"{row['trial_name']}: unsafe trial path")
            continue
        trial = root / "result.json"
        grading = root / "verifier" / "taskcompendium-result.json"
        for field, path in (("trial_sha256", trial), ("grading_sha256", grading)):
            expected = row.get(field)
            if expected is None:
                if row.get("status") in {"graded", "extraction_error"}:
                    issues.append(f"{row['trial_name']}: completed cell lacks {field}")
                continue
            if (
                not isinstance(expected, str)
                or not path.is_file()
                or sha256(path) != expected
            ):
                issues.append(
                    f"{row['trial_name']}: {field} does not bind retained artifact"
                )
        if trial.is_file():
            try:
                trial_result = _read_json(trial)
                trial_exception = trial_result.get("exception_info")
                verifier_result = trial_result.get("verifier_result")
                if extraction:
                    if (
                        not isinstance(trial_exception, dict)
                        or trial_exception.get("exception_type") != "ExtractionError"
                        or "verifier_result" not in trial_result
                        or verifier_result is not None
                    ):
                        raise ValueError(
                            "retained trial does not prove expected extraction transport"
                        )
                else:
                    if (
                        "exception_info" not in trial_result
                        or trial_exception is not None
                    ):
                        raise ValueError("retained trial records an exception")
                    if (
                        not isinstance(verifier_result, dict)
                        or not isinstance(verifier_result.get("rewards"), dict)
                        or verifier_result["rewards"].get("reward") != row.get("reward")
                    ):
                        raise ValueError(
                            "retained Harbor trial lacks the reported reward"
                        )
            except (OSError, TypeError, ValueError) as error:
                issues.append(f"{row['trial_name']}: {error}")
        if (
            row.get("status") not in {"graded", "extraction_error"}
            or not grading.is_file()
        ):
            continue
        try:
            actual = _read_json(grading)
            detail = actual.get("detail")
            fingerprint = (
                detail.get("grading_input_fingerprint")
                if isinstance(detail, dict)
                else None
            )
            if (
                actual.get("status") != row.get("status")
                or actual.get("reward") != row.get("reward")
                or fingerprint != row.get("grading_input_fingerprint")
            ):
                raise ValueError(
                    "row grade fields differ from retained verifier result"
                )
            actual_rows.append((row, root, actual))
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            issues.append(f"{row['trial_name']}: {error}")
    candidates: dict[str, dict] = {}
    if plan.get("binding_kind") == "docker":
        for row, root, _actual in actual_rows:
            try:
                candidates[row["trial_name"]] = provider_isolation_record(
                    root / "daytona-environment.json",
                    output / "runtime-trials",
                    candidate_ids,
                )
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                issues.append(f"{row['trial_name']}: {error}")
    elif plan.get("binding_kind") == "shellsim":
        from .non_docker_reset import shellsim_candidate_record

        sessions: set[str] = set()
        bridge_sha = report.get("shellsim_bridge_sha256")
        for row, root, _actual in actual_rows:
            try:
                candidates[row["trial_name"]] = shellsim_candidate_record(
                    root, sessions, bridge_sha,
                )
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                issues.append(f"{row['trial_name']}: {error}")
        if len({receipt["initial_snapshot_sha256"] for receipt in candidates.values()}) != 1:
            issues.append("ShellSim initial VFS differs across fixed grading cells")
    for row, _root, actual in actual_rows:
        try:
            if plan.get("binding_kind") in {"docker", "shellsim"} and row.get(
                "candidate_environment"
            ) != candidates.get(row["trial_name"]):
                raise ValueError(
                    "row candidate isolation differs from retained receipt"
                )
            if native:
                expected_receipt = {
                    "schema_version": "capability-native-verifier-receipt-v1",
                    "adapter_sha256": plan["identities"]["native_verifier"]["sha256"],
                    "semantic_verifier_sha256": plan["identities"][
                        "native_semantic_verifier_runtime_sha256"
                    ],
                }
                if (
                    actual.get("detail", {}).get("native_verifier_receipt")
                    != expected_receipt
                    or row.get("native_verifier_receipt") != expected_receipt
                    or row.get("private_verifier") is not None
                    or row.get("verifier_result_present")
                    is not (row.get("status") == "graded")
                ):
                    raise ValueError(
                        "native verifier source receipt differs from frozen plan"
                    )
                continue
            if row.get("outcome_class") == "extraction_error":
                if row.get("private_verifier") is not None:
                    raise ValueError(
                        "pre-verifier extraction cannot claim a private grader receipt"
                    )
                continue
            verifier = verifier_isolation_record(
                actual,
                verifier_ids,
                candidate_ids,
                runtime_image=plan["runtime"]["image"],
                supervisor_python=plan["runtime"]["supervisor_python"],
            )
            if (
                plan.get("binding_kind") in {"docker", "shellsim"}
                and row.get("candidate_environment")
                != candidates.get(row["trial_name"])
                or row.get("private_verifier") != verifier
            ):
                raise ValueError("row isolation fields differ from retained receipts")
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            issues.append(f"{row['trial_name']}: {error}")
    return issues


def _classify(plan: dict, report: dict, output: Path) -> tuple[str, list[str], dict]:
    controls = _read_json(
        output.parent / "plan-bundle" / "input" / "bundle" / "controls.json"
    )
    issues = _artifact_issues(plan, report, output, controls)
    cells = report.get("cells") if isinstance(report.get("cells"), list) else []
    summary = regrade.summarize(plan, cells, controls)
    complete = (
        not issues
        and len(cells) == plan["cell_count"]
        and all(
            isinstance(row, dict)
            and row.get("status") in {"graded", "extraction_error"}
            for row in cells
        )
    )
    fixed_inputs = complete and all(
        case.get("fixed_grading_input") is True for case in summary.get("cases", [])
    )
    if not complete:
        return (
            "pending",
            issues or ["regrade has incomplete infrastructure evidence"],
            summary,
        )
    if not fixed_inputs:
        return "pending", ["grading input bytes differ or are incomplete"], summary
    if summary.get("state") == "passed":
        return "ready", [], summary
    return (
        "semantic_failed",
        ["complete fixed-input grading disagrees with expectations"],
        summary,
    )


def _stop_process_group(child: subprocess.Popen) -> None:
    """Stop the controller and its pinned-runtime child even if the leader exited."""
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


def _default_runner(
    *,
    plan_bundle: Path,
    plan_sha256: str,
    output: Path,
    taskcompendium_source: Path,
    timeout_seconds: int,
) -> int:
    """Use a bounded subprocess from synthesis threads, with remote-only task code."""
    if os.environ.get("CAPABILITY_REMOTE_REGRADE") != "1":
        raise RuntimeError("grading diagnostics require the remote worker")
    if not sandbox_provider.credentials_present():
        raise RuntimeError("grading diagnostics require Daytona credentials")
    environment = dict(os.environ)
    environment.pop("CAPABILITY_REGRADE_INNER", None)
    environment["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[1])
        + os.pathsep
        + environment.get("PYTHONPATH", "")
    )
    command = [
        sys.executable,
        "-m",
        "capability_pipeline.cli",
        "regrade",
        "--source",
        str(plan_bundle.resolve()),
        "--plan-sha256",
        plan_sha256,
        "--taskcompendium-source",
        str(taskcompendium_source.resolve()),
        "--out",
        str(output.resolve()),
    ]
    # This is controller output only. Generated candidate and verifier programs
    # are launched by regrade in their separate remote Daytona sandboxes.
    with (output.parent / "controller-run.log").open("xb") as log:
        child = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return child.wait(timeout=timeout_seconds)
        finally:
            _stop_process_group(child)


def run_grading_diagnostics(
    evaluation_bundle: Path,
    toolchain_source: Path,
    out: Path,
    *,
    parallelism: int = 8,
    timeout: int = 3600,
    runner: Callable[..., int] | None = None,
) -> dict:
    """Prepare/run/reuse a 10-repeat regrade without changing task admission state."""
    if Path(out).is_symlink() or Path(evaluation_bundle).is_symlink():
        return {
            "state": "pending",
            "reviewable": False,
            "issues": ["diagnostic root is a symlink"],
            "extra_files": {},
        }
    evaluation_bundle = Path(evaluation_bundle).resolve()
    toolchain_source = Path(toolchain_source).resolve()
    out = Path(out).resolve()
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    timeout_seconds = timeout
    try:
        _source_identity(evaluation_bundle)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        return {
            "state": "pending",
            "reviewable": False,
            "issues": [f"evaluation input integrity: {error}"],
            "extra_files": {},
        }
    unsupported = _unsupported(evaluation_bundle)
    if unsupported is not None:
        return {
            "state": "unsupported",
            "reviewable": False,
            "unassessed": True,
            "issues": [unsupported],
            "extra_files": {},
        }
    plan_bundle = out / "plan-bundle"
    result_path = out / "grading-diagnostics.json"
    if not out.exists():
        try:
            out.mkdir(parents=True)
            created = regrade.create_plan_bundle(
                evaluation_bundle,
                toolchain_source,
                plan_bundle,
                parallelism=parallelism,
            )
            plan_sha256 = created["plan_sha256"]
        except Exception as error:  # noqa: BLE001 - creation failures are pending.
            return {
                "state": "pending",
                "reviewable": False,
                "issues": [f"regrade plan: {type(error).__name__}"],
                "extra_files": {},
            }
    elif not plan_bundle.is_dir() or not result_path.parent.is_dir():
        return {
            "state": "pending",
            "reviewable": False,
            "issues": ["diagnostic output is partial and cannot be resumed"],
            "extra_files": {},
        }
    else:
        try:
            manifest = _read_json(plan_bundle / "manifest.json")
        except (OSError, ValueError, TypeError):
            return {
                "state": "pending",
                "reviewable": False,
                "issues": ["retained plan metadata is invalid"],
                "extra_files": {},
            }
        plan_sha256 = manifest.get("plan_sha256")
        if not isinstance(plan_sha256, str):
            return {
                "state": "pending",
                "reviewable": False,
                "issues": ["plan hash is invalid"],
                "extra_files": {},
            }
    try:
        plan, _ = regrade.validate_plan_bundle(
            plan_bundle, toolchain_source, plan_sha256
        )
        if _source_identity(evaluation_bundle)["evaluation_manifest_sha256"] != sha256(
            plan_bundle / "input" / "manifest.json"
        ):
            raise RuntimeError(
                "frozen plan input differs from requested evaluation bundle"
            )
        binding = _binding(
            evaluation_bundle, plan_bundle, plan, timeout_seconds, parallelism
        )
    except Exception as error:  # noqa: BLE001 - frozen evidence failures remain pending.
        return {
            "state": "pending",
            "reviewable": False,
            "issues": [str(error)],
            "extra_files": {},
        }
    binding_path = out / "binding.json"
    if binding_path.exists():
        try:
            if _read_json(binding_path) != binding:
                return {
                    "state": "pending",
                    "reviewable": False,
                    "issues": ["frozen controller or input identity drifted"],
                    "extra_files": {},
                }
        except (ValueError, TypeError) as error:
            return {
                "state": "pending",
                "reviewable": False,
                "issues": [str(error)],
                "extra_files": {},
            }
    else:
        _write(binding_path, binding)

    output = out / "regrade"
    report_path = output / "regrade.json"
    if (
        output.exists() or result_path.exists() or (out / "controller-run.log").exists()
    ) and not report_path.is_file():
        return {
            "state": "pending",
            "reviewable": False,
            "issues": ["incomplete regrade output cannot be safely overwritten"],
            "extra_files": {},
        }
    if not report_path.is_file():
        try:
            (runner or _default_runner)(
                plan_bundle=plan_bundle,
                plan_sha256=plan_sha256,
                output=output,
                taskcompendium_source=toolchain_source,
                timeout_seconds=timeout_seconds,
            )
        except Exception as error:  # noqa: BLE001 - provider failures are pending evidence.
            payload = {
                "state": "pending",
                "reviewable": False,
                "issues": [f"regrade runner: {type(error).__name__}"],
                "extra_files": {},
            }
            _write(result_path, payload)
            return payload
    if not report_path.is_file():
        payload = {
            "state": "pending",
            "reviewable": False,
            "issues": ["regrade output is absent"],
            "extra_files": {},
        }
        _write(result_path, payload)
        return payload
    closure_path = out / "artifacts.manifest.json"
    try:
        inventory = _inventory(output)
        log_path = out / "controller-run.log"
        if log_path.is_symlink():
            raise ValueError("retained controller log is a symlink")
        if log_path.is_file():
            inventory[log_path.name] = sha256(log_path)
        closure = {"binding_sha256": sha256(binding_path), "files": inventory}
        if closure_path.exists():
            if closure_path.is_symlink() or _read_json(closure_path) != closure:
                raise ValueError("retained grading artifact inventory changed")
        else:
            _write(closure_path, closure)
        post_plan, _ = regrade.validate_plan_bundle(
            plan_bundle, toolchain_source, plan_sha256
        )
        if (
            _binding(
                evaluation_bundle, plan_bundle, post_plan, timeout_seconds, parallelism
            )
            != binding
        ):
            raise RuntimeError("controller or input identity changed during regrade")
        report = _read_json(report_path)
        if (
            report.get("plan_sha256") != plan_sha256
            or report.get("identities_stable") is not True
        ):
            raise ValueError("regrade report plan hash differs")
        if sha256(output / "plan.json") != plan_sha256:
            raise ValueError("retained execution plan differs from frozen plan")
        state, issues, summary = _classify(plan, report, output)
        final_inventory = _inventory(output)
        if log_path.is_file():
            final_inventory[log_path.name] = sha256(log_path)
        if final_inventory != inventory:
            raise ValueError("grading artifacts changed while validating evidence")
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        state, issues, summary = "pending", [str(error)], {}
        inventory = {}
    payload = {
        "state": state,
        "reviewable": state in {"ready", "semantic_failed"},
        "unassessed": False,
        "issues": issues,
        "plan_sha256": plan_sha256,
        "report_artifact": str(report_path),
        "report_sha256": sha256(report_path),
        "summary": summary,
        "artifact_inventory": inventory,
        "extra_files": {
            "grading_diagnostics/binding.json": binding_path,
            "grading_diagnostics/regrade.json": report_path,
            "grading_diagnostics/plan.json": plan_bundle / "plan.json",
            "grading_diagnostics/plan-manifest.json": plan_bundle / "manifest.json",
            **(
                {"grading_diagnostics/artifacts.manifest.json": closure_path}
                if closure_path.is_file()
                else {}
            ),
            **(
                {"grading_diagnostics/controller-run.log": out / "controller-run.log"}
                if (out / "controller-run.log").is_file()
                else {}
            ),
        },
    }
    if not result_path.exists():
        _write(result_path, payload)
    return payload

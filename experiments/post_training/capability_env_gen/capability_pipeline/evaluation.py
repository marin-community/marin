"""Frozen, repeated Harbor evaluations. This command never admits a task."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import sandbox_provider
from .runtime import (
    control_replay_inventory,
    load_candidate_resources,
    sha256,
    tree_sha256,
)

SCHEMA = "repeated-runtime-evaluation-v1"
POLICY = {
    "CAPABILITY_SOLVER_RETRIES": "1",
    "CAPABILITY_SOLVER_MODEL": "glm-5.3",
    "CAPABILITY_SOLVER_MAX_TOKENS": "32768",
    "CAPABILITY_SOLVER_MAX_TURNS": "128",
    "CAPABILITY_SOLVER_REQUEST_TIMEOUT": "900",
    "CAPABILITY_ADVERSARY_TOKEN_LIMITS": "131072,remaining_context",
    "CAPABILITY_ADVERSARY_REQUEST_TIMEOUT": "3600",
    "CAPABILITY_JUDGE_MODEL": "glm-5.3",
    "CAPABILITY_JUDGE_PROVIDER": "glm",
}
PROVIDER_KEY = "CAPABILITY_SANDBOX_PROVIDER"


def runtime_policy() -> dict:
    """The frozen runtime environment, bound to the sandbox provider that ran it.

    Attempts run with every CAPABILITY_* variable stripped and this policy put
    back, so a diagnostic runs exactly as planned.  The provider must travel
    inside the policy: stripped, runtime.py fell back to its Daytona default
    and every repeated-diagnostic attempt on silo died with "container
    verifiers require DAYTONA_API_KEY" (2026-09-22).  Binding it here also
    makes each evaluation's attestation name the provider it ran on.
    """
    return {**POLICY, PROVIDER_KEY: sandbox_provider.provider()}


def policy_matches(recorded) -> bool:
    """Accept only the current provider's policy.

    Plans written before the provider was recorded carry exactly POLICY and
    were all made on Daytona, so they stay valid only under Daytona.
    """
    if recorded == runtime_policy():
        return True
    return recorded == POLICY and sandbox_provider.provider() == sandbox_provider.DAYTONA


UNASSESSED = [
    "provenance_license",
    "semantic_alignment",
    "build_reproducibility",
    "reset_determinism",
    "extraction_taxonomy",
    "reward_determinism_10_regrades",
    "code_mutation",
    "judge_calibration",
    "resource_envelope",
    "split_contamination",
    "outcome_failure_injections",
    "full_environment_conformance",
    "critical_negative_inventory_completeness",
]


def read(path: Path):
    return json.loads(path.read_text())


def write_new(path: Path, value) -> None:
    """Refuse overwrites, including an existing result from an interrupted run."""
    with path.open("x") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def input_hash(path: Path) -> str:
    if path.is_symlink() or (
        path.is_dir() and any(p.is_symlink() for p in path.rglob("*"))
    ):
        raise ValueError("evaluation inputs must not contain symlinks")
    if path.is_dir():
        return tree_sha256(path)
    if path.is_file():
        return sha256(path)
    raise ValueError("evaluation input is missing")


def controller_hashes() -> dict:
    root = Path(__file__).resolve().parent
    paths = [*root.glob("*.py"), *root.glob("*.json")]
    paths += list((root.parent / "vendor" / "task_spec").rglob("*.json"))
    paths += list((root.parent / "vendor" / "task_spec" / "patches").rglob("*.patch"))
    return {str(p.relative_to(root.parent)): sha256(p) for p in sorted(paths)}


def critical_control_ids(cases: list[dict]) -> list[str]:
    """Preserve declared zero-reward cases without inventing criticality."""
    selected = []
    for case in cases:
        expected = case.get("expect", {})
        zero = (
            type(expected.get("reward_max")) in (int, float)
            and expected["reward_max"] == 0
        )
        if "critical" in case and type(case["critical"]) is not bool:
            raise ValueError("control critical flag must be boolean")
        if case.get("critical") is True and (
            case["class"] not in {"negative", "malformed"}
            or expected.get("status") != "graded"
            or not zero
            or expected.get("reward_min", 0) != 0
        ):
            raise ValueError("critical control must declare a graded exact-zero range")
        if case["class"] != "positive" and zero:
            selected.append(case["id"])
    return sorted(selected)


def needs_daytona_helper(bundle: Path) -> bool:
    def container(value):
        if isinstance(value, dict):
            return value.get("kind") in {"container", "docker"} or any(
                container(v) for v in value.values()
            )
        return isinstance(value, list) and any(container(v) for v in value)

    return any(
        container(read(bundle / name))
        for name in ("binding.json", "specification.json", "composite-verifier.json")
        if (bundle / name).is_file()
    )


def replay_inventory(controls_path: Path) -> dict:
    """Validate relative replay assets before spending remote runtime work."""
    return {
        case["id"]: control_replay_inventory(controls_path.parent, case)
        for case in read(controls_path)["cases"]
    }


def _validate_primary_gate(value: dict) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {
            "schema_version", "state", "runtime_evidence_sha256",
            "adjudication_result_sha256", "new_attacks_in_repeated_measurement",
        }
        or value["schema_version"] != "capability-primary-adversarial-gate-v1"
        or value["state"] not in {"passed", "resolved"}
        or value["new_attacks_in_repeated_measurement"] is not False
    ):
        raise ValueError("primary adversarial gate is invalid")
    digests = (value["runtime_evidence_sha256"], value["adjudication_result_sha256"])
    if (
        not isinstance(digests[0], str)
        or len(digests[0]) != 64
        or any(character not in "0123456789abcdef" for character in digests[0])
        or (value["state"] == "passed") != (digests[1] is None)
        or (digests[1] is not None and (
            not isinstance(digests[1], str)
            or len(digests[1]) != 64
            or any(character not in "0123456789abcdef" for character in digests[1])
        ))
    ):
        raise ValueError("primary adversarial gate digest is invalid")


def make_plan(args) -> int:
    destination = Path(args.out).resolve()
    if destination.exists():
        raise ValueError("evaluation plan already exists")
    inputs = {
        key: Path(getattr(args, key)).resolve()
        for key in ("package", "bundle", "controls")
    }
    helper = getattr(args, "daytona_helper", None)
    if helper:
        helper = Path(helper).resolve()
        if helper.name != "dt.py" or not helper.is_file():
            raise ValueError("Daytona helper must be the maintained dt.py file")
        inputs["daytona_helper"] = helper
    if needs_daytona_helper(inputs["bundle"]) and "daytona_helper" not in inputs:
        raise ValueError("Daytona execution requires a frozen --daytona-helper dt.py")
    binding_kind = read(inputs["bundle"] / "binding.json")["environment"]["kind"]
    resources = getattr(args, "candidate_resources", None)
    if resources:
        if binding_kind != "docker":
            raise ValueError("candidate resources require a Docker environment")
        load_candidate_resources(Path(resources))
        inputs["candidate_resources"] = Path(resources).resolve()
    if binding_kind == "shellsim":
        bridge = getattr(args, "shellsim_bridge", None)
        if not bridge or not Path(bridge).is_file():
            raise ValueError("ShellSim evaluation requires a pinned bridge file")
        inputs["shellsim_bridge"] = Path(bridge).resolve()
    primary_gate = getattr(args, "primary_adversarial_gate", None)
    if primary_gate:
        inputs["primary_adversarial_gate"] = Path(primary_gate).resolve()
        _validate_primary_gate(read(inputs["primary_adversarial_gate"]))
    for key in ("package", "bundle"):
        if not inputs[key].is_dir() or destination.is_relative_to(inputs[key]):
            raise ValueError("plan must be outside input directories")
    controls = read(inputs["controls"])
    cases = controls.get("cases", [])
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("controls require distinct case IDs")
    if not any(c["class"] == "positive" for c in cases):
        raise ValueError("evaluation requires an authored positive control")
    specification = inputs["bundle"] / "specification.json"
    lowered = inputs["package"] / "specification.json"
    if not lowered.is_file():
        raise ValueError("lowered package is missing its specification")
    # TaskCompendium lowering re-serializes the specification, so the package
    # manifest binds the LOWERED bytes and those bytes need not equal the
    # authored bundle's.  Check the package's own integrity against the bytes
    # it actually names, then check that the lowered document still says the
    # same thing as the authored one.  Comparing authored bytes to a manifest
    # that never described them rejects sound packages.
    if read(inputs["package"] / "manifest.json")["specification_sha256"] != sha256(
        lowered
    ):
        raise ValueError("package manifest does not bind the lowered specification")
    if read(lowered) != read(specification):
        raise ValueError("lowered package specification differs from the authored bundle")
    plan = {
        "schema_version": SCHEMA,
        "scope": (
            "three fresh oracle, solver and control measurements; primary adversarial review bound"
            if primary_gate else "three fresh full runtime attempts; not training admission"
        ),
        "purpose": "primary_bound_repeatability" if primary_gate else "full_runtime",
        "attempts": 3,
        "provider_seed": None,
        "seed_semantics": "fresh attempts; provider seed is unsupported and not claimed",
        "solver_successes_required": 2,
        "critical_control_ids": critical_control_ids(cases),
        "replay_inventory": replay_inventory(inputs["controls"]),
        "attempt_timeout_seconds": args.attempt_timeout,
        "inputs": {
            key: {
                "path": os.path.relpath(path, destination.parent),
                "sha256": input_hash(path),
            }
            for key, path in inputs.items()
        },
        "specification_sha256": sha256(specification),
        # The runtime executes the LOWERED Harbor package, so its attestation
        # names these bytes.  Carry both: the authored hash re-verifies the
        # bundle, the lowered hash checks the attestation.
        "lowered_specification_sha256": sha256(lowered),
        "controller_files": controller_hashes(),
        "runtime_environment": runtime_policy(),
        "unassessed_recipe_rows": UNASSESSED,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_new(destination, plan)
    print(json.dumps({"plan": str(destination), "sha256": sha256(destination)}))
    return 0


def validate_plan(path: Path, expected_sha256: str) -> tuple[dict, dict[str, Path]]:
    if sha256(path) != expected_sha256:
        raise ValueError("evaluation plan fingerprint mismatch")
    plan = read(path)
    if (
        plan.get("schema_version") != SCHEMA
        or plan.get("attempts") != 3
        or plan.get("provider_seed") is not None
        or plan.get("solver_successes_required") != 2
        or type(plan.get("attempt_timeout_seconds")) is not int
        or plan["attempt_timeout_seconds"] <= 0
        or not policy_matches(plan.get("runtime_environment"))
        or plan.get("controller_files") != controller_hashes()
        or plan.get("purpose") not in {"full_runtime", "primary_bound_repeatability"}
        or plan.get("unassessed_recipe_rows") != UNASSESSED
        or not {"package", "bundle", "controls"} <= set(plan.get("inputs", {}))
        or not set(plan.get("inputs", {}))
        <= {
            "package",
            "bundle",
            "controls",
            "shellsim_bridge",
            "daytona_helper",
            "candidate_resources",
            "primary_adversarial_gate",
        }
        or ("primary_adversarial_gate" in plan.get("inputs", {})) != (
            plan.get("purpose") == "primary_bound_repeatability"
        )
    ):
        raise ValueError("unsupported or changed evaluation contract/controller")
    inputs = {}
    for key, record in plan["inputs"].items():
        inputs[key] = (path.parent / record["path"]).resolve()
        if input_hash(inputs[key]) != record["sha256"]:
            raise ValueError(f"evaluation input fingerprint mismatch: {key}")
    if "primary_adversarial_gate" in inputs:
        _validate_primary_gate(read(inputs["primary_adversarial_gate"]))
    if sha256(inputs["bundle"] / "specification.json") != plan["specification_sha256"]:
        raise ValueError("evaluation specification fingerprint mismatch")
    if needs_daytona_helper(inputs["bundle"]) and "daytona_helper" not in inputs:
        raise ValueError("Daytona execution lacks a frozen helper")
    if "candidate_resources" in inputs:
        if read(inputs["bundle"] / "binding.json")["environment"]["kind"] != "docker":
            raise ValueError("candidate resources require a Docker environment")
        load_candidate_resources(inputs["candidate_resources"])
    if "daytona_helper" in inputs and (
        inputs["daytona_helper"].name != "dt.py"
        or not inputs["daytona_helper"].is_file()
    ):
        raise ValueError("invalid frozen Daytona helper")
    if (
        read(inputs["bundle"] / "binding.json")["environment"]["kind"] == "shellsim"
    ) != ("shellsim_bridge" in inputs):
        raise ValueError("ShellSim bridge binding mismatch")
    if plan.get("critical_control_ids") != critical_control_ids(
        read(inputs["controls"])["cases"]
    ):
        raise ValueError("critical control inventory differs from frozen controls")
    if plan.get("replay_inventory") != replay_inventory(inputs["controls"]):
        raise ValueError("control replay inventory differs from frozen plan")
    return plan, inputs


def sandbox_ids(value) -> set[str]:
    """Deduplicate repeated references within one run, then compare across runs."""
    ids = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"sandbox_id", "verifier_sandbox_id"} and isinstance(child, str):
                ids.add(child)
            else:
                ids.update(sandbox_ids(child))
    elif isinstance(value, list):
        for child in value:
            ids.update(sandbox_ids(child))
    return ids


def model_provenance(root: Path) -> dict:
    identities = set()
    missing = 0
    for path in root.rglob("glm-requests.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            identity = (event.get("model"), event.get("system_fingerprint"))
            if not all(isinstance(value, str) and value for value in identity):
                missing += 1
            identities.add(identity)
    return {
        "returned_identities": [
            {"model": model, "system_fingerprint": fingerprint}
            for model, fingerprint in sorted(identities, key=repr)
        ],
        "requests_without_full_identity": missing,
        "revision_claim": "provider-returned metadata only; missing fingerprints remain unknown",
    }


def artifact_manifest(root: Path) -> dict:
    """Explicit payload inventory excludes its own manifest and receipt."""
    return {
        "excluded": ["artifacts.manifest.json", "receipt.json"],
        "files": {
            str(path.relative_to(root)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.relative_to(root).as_posix()
            not in {"artifacts.manifest.json", "receipt.json"}
        },
    }


def inspect_attempt(root: Path, inputs: dict, plan: dict) -> dict:
    from .synthesis import _attestation_issues, _controls_pass

    evidence_path = root / "runtime-evidence.json"
    if not evidence_path.is_file():
        return {
            "state": "incomplete",
            "issues": ["runtime evidence is absent"],
            "sandbox_ids": [],
        }
    evidence = read(evidence_path)
    primary_bound = plan.get("purpose") == "primary_bound_repeatability"
    lowered_sha256 = plan.get("lowered_specification_sha256")
    if not isinstance(lowered_sha256, str) or not lowered_sha256:
        # Falling back to the authored hash here is what made three separate
        # runs report an unbound attestation; refuse instead of guessing.
        raise ValueError("evaluation plan does not record the lowered specification hash")
    issues = _attestation_issues(
        evidence,
        evidence_path,
        inputs["bundle"],
        inputs["package"],
        inputs["controls"],
        lowered_sha256,
        require_solver_pass=False,
        require_independent_adversary=not primary_bound,
    )
    attestation = evidence.get("attestation", {})
    expected_resources = inputs.get("candidate_resources")
    if attestation.get("candidate_resources_sha256") != (
        sha256(expected_resources) if expected_resources else None
    ) or attestation.get("candidate_resource_request") != (
        load_candidate_resources(expected_resources) if expected_resources else None
    ):
        issues.append("candidate resource request differs from frozen plan")
    if attestation.get("solver", {}).get("retry_limit") != 1:
        issues.append(
            "evaluation requires exactly one solver attempt per positive case"
        )
    if attestation.get("solver", {}).get("state") not in {
        "passed",
        "needs_adjudication",
    }:
        issues.append("unknown solver outcome state")
    controls = read(inputs["controls"])
    definitions = controls["cases"]
    records = evidence.get("cases", [])
    # Exact global coverage must hold before slicing solver/negative results.
    if len(records) != len(definitions) or {c.get("id") for c in records} != {
        c["id"] for c in definitions
    }:
        issues.append("runtime evidence does not cover the controls exactly")
    results = {}
    for group, positive in (("solver", True), ("authored_controls", False)):
        selected = [c for c in definitions if (c["class"] == "positive") == positive]
        ids = {c["id"] for c in selected}
        passed, group_issues = _controls_pass(
            {"cases": selected},
            {"cases": [r for r in records if r.get("id") in ids]},
            external=True,
        )
        # Preserve preregistered noncritical partial-rubric credit. Completeness
        # of the critical inventory requires separate semantic review.
        if not positive:
            negative_ids = set(plan["critical_control_ids"])
            for record in records:
                if (
                    record.get("id") in negative_ids
                    and record.get("result", {}).get("status") == "graded"
                    and record.get("result", {}).get("reward") != 0
                ):
                    group_issues.append(
                        f"{record['id']}: negative received nonzero reward"
                    )
        results[group] = {"passed": passed and not group_issues, "issues": group_issues}
    # Evidence integrity: the attested solver state must follow from the
    # recorded grades by the runtime's own rule (every positive case graded
    # within [0.8, 1.0]).  Authored expectations stricter than that (e.g.
    # reward_min 1.0 or criterion assertions) are a graded solver result for
    # the gates below, not an evidence disagreement: comparing against them
    # marked 39 valid cells invalid_evidence on catalog-full-construct-003.
    solver_passed = results["solver"]["passed"]
    positive_ids = {c["id"] for c in definitions if c["class"] == "positive"}
    runtime_solver_passed = all(
        record.get("result", {}).get("status") == "graded"
        and type(record.get("result", {}).get("reward")) in (int, float)
        and 0.8 <= record["result"]["reward"] <= 1.0
        for record in records
        if record.get("id") in positive_ids
    )
    if (attestation.get("solver", {}).get("state") == "passed") != runtime_solver_passed:
        issues.append("solver state disagrees with recorded grades")
    referenced = [evidence]
    attack_issues = []
    if not issues:
        for key in (
            "oracle_artifact", "solver_artifact",
            *(("adversary_artifact",) if not primary_bound else ()),
        ):
            referenced.append(read(root / attestation[key]))
        if not primary_bound:
            from .adversary import assess_attack_results

            attack_issues = assess_attack_results(referenced[-1]["cases"], max_reward=0.0)
    return {
        "state": "valid" if not issues else "invalid_evidence",
        "issues": issues,
        "oracle_passed": not issues,
        "solver_passed": solver_passed and not issues,
        "authored_controls_passed": results["authored_controls"]["passed"]
        and not issues,
        "control_results": results,
        "independent_attacks_passed": (
            not issues and not attack_issues and not primary_bound
            and attestation.get("adversarial", {}).get("independent_attack_executed") is True
        ),
        "primary_adversarial_review_bound": not issues and primary_bound,
        "independent_attack_issues": attack_issues,
        "sandbox_ids": sorted(sandbox_ids(referenced)),
        "evidence_sha256": sha256(evidence_path),
    }


def aggregate(plan: dict, cells: list[dict]) -> dict:
    expected = set(range(1, plan["attempts"] + 1))
    observed = [cell["attempt"] for cell in cells]
    complete = len(observed) == len(expected) and set(observed) == expected
    seen, reused = set(), set()
    for cell in cells:
        ids = set(cell.get("sandbox_ids", []))
        reused.update(seen & ids)
        seen.update(ids)
    valid = complete and not reused and all(c.get("state") == "valid" for c in cells)
    counts = {
        key: sum(c.get(key) is True for c in cells)
        for key in (
            "oracle_passed",
            "solver_passed",
            "authored_controls_passed",
            "independent_attacks_passed",
            "primary_adversarial_review_bound",
        )
    }
    gates = {
        "oracle_3_of_3": valid and counts["oracle_passed"] == 3,
        "solver_at_least_2_of_3": valid and counts["solver_passed"] >= 2,
        "authored_controls_all_attempts": valid
        and counts["authored_controls_passed"] == 3,
    }
    if plan.get("purpose") == "primary_bound_repeatability":
        gates["primary_adversarial_review_bound"] = (
            valid and counts["primary_adversarial_review_bound"] == 3
        )
    else:
        gates["independent_attacks_all_attempts"] = (
            valid and counts["independent_attacks_passed"] == 3
        )
    return {
        "schema_version": SCHEMA,
        "state": "repeated_runtime_passed" if all(gates.values()) else "needs_review",
        "training_admission": "not_assessed",
        "complete_attempt_inventory": complete,
        "provider_seed": None,
        "gates": gates,
        "counts": counts,
        "reused_sandbox_ids": sorted(reused),
        "sandbox_reuse_check_scope": "reported Daytona candidate and private-verifier IDs only",
        "cells": cells,
        "unassessed_recipe_rows": plan["unassessed_recipe_rows"],
        "measurement_limit": "controller elapsed time and artifact bytes are not sandbox resource measurements",
        "model_provenance": [c.get("model_provenance") for c in cells],
        "backend_homogeneity": "not_assessed; inspect returned model identities and missing fingerprints",
        "critical_control_ids": plan["critical_control_ids"],
        "critical_inventory_completeness_assessed": False,
    }


def run_evaluation(args) -> int:
    from .synthesis import OfficialToolchain, _run

    plan_path = Path(args.plan).resolve()
    plan, inputs = validate_plan(plan_path, args.plan_sha256)
    output = Path(args.out).resolve()
    if any(
        output.is_relative_to(path) or path.is_relative_to(output)
        for path in inputs.values()
    ):
        raise ValueError("evaluation output must be separate from all inputs")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "plan.lock.json", plan)
    write_new(
        output / "started.json",
        {
            "plan_sha256": args.plan_sha256,
            "started": time.time(),
            "parallel_attempts": args.concurrency,
            "relay_endpoint_sha256": hashlib.sha256(
                os.environ.get("GLM_BASE_URL", "").rstrip("/").encode()
            ).hexdigest(),
        },
    )
    toolchain = OfficialToolchain.resolve(output, args.taskcompendium_source)
    command = toolchain.runtime_command()
    # Freeze inference knobs; credentials and live relay address remain env-only.
    environment = {
        **{k: v for k, v in os.environ.items() if not k.startswith("CAPABILITY_")},
        **plan["runtime_environment"],
    }
    if "daytona_helper" in inputs:
        environment["CAPABILITY_DAYTONA_TOOLS"] = str(inputs["daytona_helper"].parent)

    def attempt(number):
        root = output / "attempts" / f"{number:03d}"
        root.mkdir(parents=True)
        started = time.time()
        write_new(root / "started.json", {"attempt": number, "started": started})
        argv = [*command]
        for key, path in inputs.items():
            if key in {"daytona_helper", "primary_adversarial_gate"}:
                continue
            argv.extend([f"--{key.replace('_', '-')}", str(path)])
        if plan.get("purpose") == "primary_bound_repeatability":
            argv.append("--diagnostic-no-new-adversary")
        argv.extend(["--output", str(root / "runtime-evidence.json")])
        try:
            # No semantic retry or early stopping; all three planned attempts run.
            completed = _run(
                argv, timeout=plan["attempt_timeout_seconds"], env=environment
            )
            (root / "stdout.log").write_text(completed.stdout)
            (root / "stderr.log").write_text(completed.stderr)
            receipt = inspect_attempt(root, inputs, plan)
            receipt["exit_code"] = completed.returncode
            if completed.returncode not in (0, 2):
                receipt.update(
                    state="runtime_error",
                    issues=[*receipt.get("issues", []), "runtime process failed"],
                )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as error:
            receipt = {
                "state": "timeout"
                if isinstance(error, subprocess.TimeoutExpired)
                else "runtime_error",
                "error_type": type(error).__name__,
                "sandbox_ids": [],
            }
            if isinstance(error, subprocess.TimeoutExpired):
                for label, value in (
                    ("stdout", error.stdout),
                    ("stderr", error.stderr),
                ):
                    if value is not None:
                        (root / f"{label}.log").write_text(
                            value.decode(errors="replace")
                            if isinstance(value, bytes)
                            else value
                        )
        receipt.update(attempt=number, elapsed_seconds=time.time() - started)
        try:
            receipt["model_provenance"] = model_provenance(root)
        except (ValueError, TypeError):
            receipt.update(state="invalid_model_provenance", model_provenance=None)
        manifest = artifact_manifest(root)
        write_new(root / "artifacts.manifest.json", manifest)
        receipt["artifact_manifest_sha256"] = sha256(root / "artifacts.manifest.json")
        receipt["artifact_bytes"] = sum(p["bytes"] for p in manifest["files"].values())
        write_new(root / "receipt.json", receipt)
        print(json.dumps({"attempt": number, "state": receipt["state"]}), flush=True)
        return receipt

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        cells = list(pool.map(attempt, range(1, 4)))
    report = aggregate(plan, cells)
    final_inputs = {}
    for key, path in inputs.items():
        try:
            final_inputs[key] = input_hash(path)
        except (OSError, ValueError):
            final_inputs[key] = None
    try:
        validate_plan(plan_path, args.plan_sha256)
    except (OSError, ValueError):
        report.update(state="input_or_controller_changed", integrity_error=True)
        report["gates"] = dict.fromkeys(report["gates"], False)
    write_new(output / "matrix.json", report)
    write_new(
        output / "matrix.attestation.json",
        {
            "schema_version": SCHEMA,
            "plan_sha256": args.plan_sha256,
            "matrix_sha256": sha256(output / "matrix.json"),
            "finished": time.time(),
            "acceptance_files_written": [],
            "input_and_controller_hashes_unchanged_at_finish": not report.get(
                "integrity_error", False
            ),
            "initial_inputs": plan["inputs"],
            "final_input_sha256": final_inputs,
            "initial_controller": plan["controller_files"],
            "final_controller": controller_hashes(),
            "attempt_receipts": {
                f"{i:03d}": sha256(output / "attempts" / f"{i:03d}" / "receipt.json")
                for i in range(1, 4)
            },
        },
    )
    return 0 if report["state"] == "repeated_runtime_passed" else 2


def add_parser(subparsers) -> None:
    plan = subparsers.add_parser(
        "evaluate-plan",
        help="Freeze a three-attempt runtime evaluation (metadata only)",
    )
    for key in ("package", "bundle", "controls", "out"):
        plan.add_argument(f"--{key}", required=True)
    plan.add_argument("--attempt-timeout", type=positive_timeout, default=21600)
    plan.add_argument(
        "--candidate-resources",
        help="Freeze a Docker candidate request JSON: cpu, memory_gb, disk_gb",
    )
    plan.add_argument(
        "--daytona-helper",
        help="Hash-bind the maintained dt.py; required for Daytona candidates or private verifiers",
    )
    plan.add_argument(
        "--shellsim-bridge", default=os.environ.get("TASKCOMPENDIUM_SHELLSIM_BRIDGE")
    )
    plan.set_defaults(func=make_plan)
    run = subparsers.add_parser(
        "evaluate", help="Run the frozen evaluation on a configured cluster worker"
    )
    for key in ("plan", "plan-sha256", "out"):
        run.add_argument(f"--{key}", required=True)
    run.add_argument("--taskcompendium-source")
    run.add_argument("--concurrency", type=int, choices=(1, 2, 3), default=3)
    run.set_defaults(func=run_evaluation)


def positive_timeout(value: str) -> int:
    seconds = int(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("attempt timeout must be positive")
    return seconds

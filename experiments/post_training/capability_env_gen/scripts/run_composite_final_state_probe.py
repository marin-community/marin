"""Probe full-workspace composite FinalState through real Harbor and Daytona."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import shutil
from pathlib import Path

import msgspec
import taskcompendium.harbor.runner as native_runner
import taskcompendium.harbor.verifier as native_verifier
import taskcompendium.lowering as native_lowering
from taskcompendium.execution import (
    DockerEnvironment,
    HarborExecutionConfig,
    HarborLaunchConfig,
    HarborTaskBinding,
    HarnessToolBinding,
)
from taskcompendium.harbor.runner import run_trial
from taskcompendium.lowering import lower_to_harbor, resolve_harbor_execution
from taskcompendium.models import (
    VERIFIER_REVISION,
    AnswerRequirements,
    Capability,
    Embedded,
    FinalState,
    JudgeConfig,
    JudgeModelPolicy,
    JudgeView,
    Rendering,
    Resource,
    ResourceRole,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpec,
    TaskTroveVerifier,
    WorkspaceState,
)
from taskcompendium.serialization import renderings_from_json, to_json
from tasktrove_verify.spec import Mode

from capability_pipeline import composite_grading_diagnostics as fixed_grading
from capability_pipeline.composite_extension import (
    PATCHED_LOWERING_SHA256,
    PATCHED_RUNNER_SHA256,
    PATCHED_VERIFIER_SHA256,
    resolve_runner_specification,
)
from capability_pipeline.composite_policy import (
    SCHEMA_VERSION,
    validate_composite_config,
)
from capability_pipeline.runtime import (
    composite_isolation_record,
    environment_config,
)

REVISION = "dc6b501c8604bcd2e3c20c1e9947679845fdfef8"
IMAGE = "python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
QUESTION = (
    "Assess whether the release note correctly names the owner and canary status."
)
CRITERIA = (
    "The release note says owner Priya and canary passed, as supported by nested/answer.txt.",
    "The trusted machine context shows the full-workspace gate passed with reward 1.",
)
CASES = (
    {
        "id": "valid",
        "marker": "workspace-ready",
        "answer": "Owner Priya; canary passed.",
        "expect": "pass",
    },
    {
        "id": "machine-gate-fail",
        "marker": "workspace-corrupt",
        "answer": "Owner Priya; canary passed.",
        "expect": "machine_zero",
    },
    {
        "id": "judge-view-fail",
        "marker": "workspace-ready",
        "answer": "Owner Rowan; canary failed.",
        "expect": "judge_zero",
    },
)
FIXTURE = "fixture-version-1"


def run_fixed_replay_probe(root: Path, bundle: Path, package: Path, records: list[dict]) -> dict:
    """Probe-specific staging around the production frozen machine controller.

    The three already-run Harbor trials are immutable inputs. This protocol
    probe deliberately does not claim the full three-attempt evaluator source
    contract enforced by run_composite_grading_diagnostics.
    """
    if any(record.get("state") != "passed" for record in records):
        raise ValueError("fixed replay requires three completed original cases")
    attempt = root / "fixed-replay"
    inputs = attempt / "input"
    inputs.mkdir(parents=True, exist_ok=False)
    helper_dir = os.environ.get("CAPABILITY_DAYTONA_TOOLS")
    if not helper_dir or not (Path(helper_dir) / "dt.py").is_file():
        raise ValueError("fixed replay requires pinned Daytona helper")
    (inputs / "tools").mkdir()
    shutil.copy2(Path(helper_dir) / "dt.py", inputs / "tools/dt.py")
    for name in ("composite-specification.json", "renderings.json", "composite-verifier.json"):
        shutil.copy2(package / name, inputs / name)
    shutil.copy2(bundle / "binding.json", inputs / "task-binding.json")
    controls = {"cases": [
        {"id": "valid", "class": "positive"},
        {"id": "machine-gate-fail", "class": "negative"},
        {"id": "judge-view-fail", "class": "negative"},
    ]}
    fixed_grading._write(inputs / "controls.json", controls)
    report = {record["id"]: record for record in records}
    selected = []
    for case in CASES:
        case_id = case["id"]
        source_trial = root / "trials" / case_id
        target = inputs / "runtime-trials" / case_id
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_trial, target)
        selected.append({
            "case_id": case_id,
            "trial_path": f"runtime-trials/{case_id}",
            "grading_path": f"runtime-trials/{case_id}/verifier/taskcompendium-result.json",
            "source": "authored_positive" if case_id == "valid" else "authored_negative",
        })
    fixed_grading._write(inputs / "selected-trials.json", {"trials": selected})
    oracle_grade = inputs / selected[0]["grading_path"]
    oracle_trial = inputs / selected[0]["trial_path"] / "result.json"
    fixed_grading._write(inputs / "authored-oracle.json", {"cases": [{
        "case_id": "valid", "step_index": 0,
        "trial_artifact": "runtime-trials/valid/result.json",
        "trial_sha256": sha256(oracle_trial),
        "grading_artifact": selected[0]["grading_path"],
        "grading_sha256": sha256(oracle_grade),
        "result": report["valid"]["result"],
    }]})
    fixed_grading._write(inputs / "runtime-evidence.json", {"cases": [
        {
            "id": case_id, "step_index": 0,
            "control_type": "authored_adversarial_control",
            "artifact": f"runtime-trials/{case_id}/verifier/taskcompendium-result.json",
            "artifact_sha256": sha256(inputs / f"runtime-trials/{case_id}/verifier/taskcompendium-result.json"),
            "result": report[case_id]["result"],
        }
        for case_id in ("machine-gate-fail", "judge-view-fail")
    ]})
    fixed_grading._write(inputs / "receipt.json", {
        "schema_version": "probe-specific-authored-control-source-v1",
        "original_case_ids": [case["id"] for case in CASES],
    })
    fixed_grading._write(inputs / "artifacts.manifest.json", {
        "schema_version": "probe-specific-authored-control-source-v1",
        "original_trial_hashes": {
            case["id"]: sha256(root / "trials" / case["id"] / "result.json")
            for case in CASES
        },
    })
    source = {
        "first_runtime_evidence_sha256": sha256(inputs / "runtime-evidence.json"),
        "first_authored_oracle_sha256": sha256(inputs / "authored-oracle.json"),
        "first_receipt_sha256": sha256(inputs / "receipt.json"),
        "first_artifact_manifest_sha256": sha256(inputs / "artifacts.manifest.json"),
        "controls_sha256": sha256(inputs / "controls.json"),
        "task_binding_sha256": sha256(inputs / "task-binding.json"),
        "specification_sha256": sha256(inputs / "composite-specification.json"),
        "renderings_sha256": sha256(inputs / "renderings.json"),
        "config_sha256": sha256(inputs / "composite-verifier.json"),
        "daytona_helper_sha256": sha256(inputs / "tools/dt.py"),
    }
    from capability_pipeline.synthesis import SOURCE_LOCK

    binding = {
        "schema_version": fixed_grading.SCHEMA,
        "source": source,
        "controller": fixed_grading._controller(),
        "taskcompendium_source_lock_sha256": sha256(SOURCE_LOCK),
        "timeout_seconds": 3600,
        "repeats": fixed_grading.REPEATS,
        "probe_specific_source": True,
    }
    fixed_grading._write(attempt / "binding.json", binding)
    fixed_grading._write(inputs / "files.json", {"files": fixed_grading._files(inputs, exclude={"files.json"})})
    cells, original_ids = fixed_grading._validate_input(attempt, binding, derive=True)
    if len(cells) != 3 or len(original_ids["candidate_ids"]) != 3 or len(original_ids["verifier_ids"]) != 3:
        raise ValueError("fixed replay original-case denominator or isolation differs")
    binding_sha = sha256(attempt / "binding.json")
    fixed_grading._write(attempt / "run-started.json", {
        "schema_version": fixed_grading.SCHEMA, "binding_sha256": binding_sha,
    })
    os.environ["CAPABILITY_REMOTE_COMPOSITE_GRADE"] = "1"
    fixed_grading._inner(attempt, binding_sha)
    fixed_grading._closure(attempt)
    state, issues, summary = fixed_grading._classify(attempt, binding)
    result = {
        "schema_version": "capability-composite-final-state-fixed-replay-probe-v1",
        "source": "probe_specific_original_harbor_trials",
        "state": state,
        "issues": issues,
        "original_cases": 3,
        "original_machine_checks": len(cells),
        "candidate_sandbox_ids": sorted(original_ids["candidate_ids"]),
        "original_private_sandbox_ids": sorted(original_ids["verifier_ids"]),
        "expected_private_grades": len(cells) * fixed_grading.REPEATS,
        "summary": summary,
        "binding_sha256": binding_sha,
        "artifact_manifest_sha256": sha256(attempt / "artifacts.manifest.json"),
    }
    fixed_grading._write(attempt / "summary.json", result)
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_base(value: str) -> str:
    root = value.rstrip("/").removesuffix("/v1")
    if not root:
        raise ValueError("GLM base URL is empty")
    return root + "/v1"


def build(root: Path, base_url: str, model: str, provider: str) -> tuple[Path, Path]:
    if sha256(Path(native_verifier.__file__)) != PATCHED_VERIFIER_SHA256:
        raise RuntimeError("probe must run from the pinned composite source overlay")
    if sha256(Path(native_runner.__file__)) != PATCHED_RUNNER_SHA256:
        raise RuntimeError("probe must run from the pinned composite runner overlay")
    if sha256(Path(native_lowering.__file__)) != PATCHED_LOWERING_SHA256:
        raise RuntimeError("probe must run from the pinned composite lowering overlay")
    if root.exists() or root.is_symlink():
        raise FileExistsError("probe output already exists")
    bundle, package = root / "task", root / "harbor"
    bundle.mkdir(parents=True, exist_ok=False)
    gate = b"""test "$(cat root.marker)" = workspace-ready && test "$(cat nested/fixture.txt)" = fixture-version-1 && echo 1 || echo 0
"""
    resources = (
        Resource(
            "checks/gate.sh", (ResourceRole.VERIFIER,), Embedded(gate), executable=True
        ),
        Resource(
            "private/context.txt",
            (ResourceRole.VERIFIER,),
            Embedded(b"Expected owner: Priya. Expected canary: passed.\n"),
        ),
    )
    judge = JudgeConfig(
        JudgeModelPolicy(
            model=model,
            size_class="large",
            provider=provider,
            base_url=base_url,
            samples=1,
            temperature=0.0,
        ),
        JudgeView(
            transcript=True,
            files=("nested/answer.txt",),
            reference_context=("private/context.txt",),
        ),
    )
    specification = TaskSpec(
        id="protocol/composite-final-state-live-probe-v1",
        requirements=TaskRequirements(
            (Capability.FILESYSTEM, Capability.SHELL, Capability.PROCESS),
            WorkspaceState(image=IMAGE, workdir="/app"),
        ),
        resources=resources,
        metadata=TaskMetadata(Source("protocol-probe", "1", "composite", REVISION)),
        coverage_tags=("shape:answer", "subject:computing"),
        steps=(
            StepSpecification(
                instructions=(
                    QUESTION
                    + " Write root.marker, nested/fixture.txt, and nested/answer.txt in the candidate workspace."
                ),
                answer_requirements=AnswerRequirements("final_state"),
                verifier=TaskTroveVerifier(
                    Mode.JUDGE,
                    {
                        "rubric": "checklist",
                        "criteria": list(CRITERIA),
                        "question": QUESTION,
                        "exact_gate": False,
                        "request_timeout": 180.0,
                    },
                    judge=judge,
                    implementation_revision=VERIFIER_REVISION,
                ),
            ),
        ),
    )
    rendering = Rendering("workspace", FinalState((".",)))
    binding = HarborTaskBinding(
        DockerEnvironment(IMAGE, workdir="/app"),
        (HarnessToolBinding("terminal", "docker"),),
    )
    specification_path = bundle / "specification.json"
    specification_path.write_bytes(to_json(specification))
    (bundle / "renderings.json").write_bytes(msgspec.json.encode((rendering,)))
    (bundle / "binding.json").write_bytes(msgspec.json.encode(binding))
    adapter = Path(__file__).parents[1] / "capability_pipeline/composite_verifier.py"
    policy = adapter.with_name("composite_policy.py")
    protocol = adapter.with_name("native_judge_protocol.py")
    config = {
        "schema_version": SCHEMA_VERSION,
        "specification_sha256": sha256(specification_path),
        "implementation": {
            "taskcompendium_revision": REVISION,
            "adapter_sha256": sha256(adapter),
            "policy_sha256": sha256(policy),
            "native_judge_protocol_sha256": sha256(protocol),
        },
        "steps": [
            {
                "step_index": 0,
                "machine_checks": [
                    {
                        "id": "workspace-gate",
                        "role": "gate",
                        "script_path": "checks/gate.sh",
                        "args": [],
                        "image": IMAGE,
                        "timeout": 300,
                    }
                ],
                "judge": {
                    "criterion_weights": [1.0, 1.0],
                    "critical_indices": [0],
                    "critical_min": 1.0,
                    "conditional_caps": [],
                },
            }
        ],
    }
    config_path = bundle / "composite-verifier.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    validate_composite_config(
        config,
        specification_sha256=sha256(specification_path),
        adapter_sha256=sha256(adapter),
        policy_sha256=sha256(policy),
        step_count=1,
    )
    lower_to_harbor(
        specification,
        (rendering,),
        binding,
        package,
        composite_specification_path=specification_path,
        composite_config_path=config_path,
    )
    from taskcompendium.serialization import from_json

    try:
        from_json((package / "specification.json").read_bytes())
    except (TypeError, ValueError):
        pass
    else:
        raise RuntimeError("ordinary TaskCompendium decoder accepted guarded export")
    return bundle, package


def assert_case(case: dict, result: dict) -> None:
    reward, detail = result.get("reward"), result.get("detail", {})
    if result.get("status") != "graded":
        raise RuntimeError(f"{case['id']} was not graded: {result}")
    machine = detail.get("machine_results")
    expected_gate = 0.0 if case["expect"] == "machine_zero" else 1.0
    if not (
        isinstance(machine, list)
        and len(machine) == 1
        and machine[0].get("id") == "workspace-gate"
        and machine[0].get("status") == "graded"
        and machine[0].get("reward") == expected_gate
    ):
        raise RuntimeError(f"{case['id']} lacks exact private workspace gate evidence")
    if case["expect"] == "pass" and not (
        reward == 1.0
        and detail.get("judge_path") == "model"
        and detail.get("judge_criterion_scores") == [1.0, 1.0]
        and isinstance(detail.get("judge"), dict)
        and detail["judge"].get("requested_model") == "glm-5.3"
        and len(detail["judge"].get("judgments", ())) == 2
    ):
        raise RuntimeError(f"valid FinalState did not pass gate and judge: {result}")
    if case["expect"] == "machine_zero" and not (
        reward == 0.0
        and detail.get("judge_path") == "skipped_machine_gate"
        and detail.get("judge") is None
    ):
        raise RuntimeError(f"machine gate did not skip native judge: {result}")
    if case["expect"] == "judge_zero" and not (
        reward == 0.0
        and detail.get("judge_path") == "model"
        and 0 in detail.get("failed_judge_critical_indices", [])
    ):
        raise RuntimeError(f"judge did not reject wrong nested answer: {result}")


def assert_fail_closed(package: Path, root: Path) -> list[dict]:
    """Exercise the runner's extension resolver before any Harbor trial exists."""
    records = []
    execution = {
        "verifier": {
            "import_path": "capability_pipeline.composite_verifier:CompositeSemanticVerifier"
        }
    }
    for name in ("missing-marker", "tampered-marker"):
        target = root / name
        shutil.copytree(package, target)
        manifest_path = target / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if name == "missing-marker":
            manifest.pop("required_extensions")
        else:
            manifest["required_extensions"][0]["config_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        try:
            resolve_runner_specification(target, execution)
        except (RuntimeError, ValueError) as error:
            records.append(
                {
                    "id": name,
                    "state": "rejected_before_trial",
                    "error_type": type(error).__name__,
                    "manifest_sha256": sha256(manifest_path),
                }
            )
        else:
            raise RuntimeError(f"{name} extension reached Harbor trial")
    return records


async def run(args) -> int:
    if not os.environ.get(args.api_key_env):
        raise RuntimeError(f"missing judge credential: {args.api_key_env}")
    if not os.environ.get("DAYTONA_API_KEY"):
        raise RuntimeError("missing DAYTONA_API_KEY")
    base = args.base_url or os.environ.get("GLM_BASE_URL")
    if not base:
        raise RuntimeError("GLM_BASE_URL is required")
    root = Path(args.out).resolve()
    bundle, package = build(root, normalize_base(base), args.model, args.provider)
    fail_closed = assert_fail_closed(package, root)
    rendering = renderings_from_json((bundle / "renderings.json").read_bytes())
    binding = HarborTaskBinding(
        DockerEnvironment(IMAGE, workdir="/app"),
        (HarnessToolBinding("terminal", "docker"),),
    )
    environment, _ = environment_config(binding, None)
    trials = root / "trials"
    config_sha256 = sha256(bundle / "composite-verifier.json")
    checks = json.loads((bundle / "composite-verifier.json").read_text())["steps"][0][
        "machine_checks"
    ]

    async def execute(case: dict) -> BaseException | None:
        try:
            execution = resolve_harbor_execution(
                rendering,
                HarborExecutionConfig(binding, HarborLaunchConfig("replay")),
                environment,
                agent_kwargs={
                    "response": "Workspace files written.",
                    "commands": [
                        "mkdir -p nested",
                        f"printf '%s\\n' {shlex.quote(case['marker'])} > root.marker",
                        f"printf '%s\\n' {shlex.quote(FIXTURE)} > nested/fixture.txt",
                        f"printf '%s\\n' {shlex.quote(case['answer'])} > nested/answer.txt",
                    ],
                },
                verifier_kwargs={"judge_api_key_env": args.api_key_env},
            )
            execution["verifier"]["import_path"] = (
                "capability_pipeline.composite_verifier:CompositeSemanticVerifier"
            )
            await run_trial(package, execution, trials, case["id"])
        except Exception as error:  # noqa: BLE001 -- retain other independent case outcomes
            return error
        return None

    # Each case has its own trial directory and candidate/private sandboxes.
    launch_errors = await asyncio.gather(*(execute(case) for case in CASES))
    records, sandbox_ids, candidate_ids = [], set(), set()
    for case, launch_error in zip(CASES, launch_errors, strict=True):
        trial = trials / case["id"]
        paths = sorted(trial.rglob("*")) if trial.is_dir() else []
        record = {
            "id": case["id"],
            "expected": case["expect"],
            "trial_artifacts": {
                str(path.relative_to(root)): sha256(path)
                for path in paths
                if path.is_file() and not path.is_symlink()
            },
        }
        if launch_error is not None:
            record.update(
                state="failed",
                failure="harbor_trial_exception",
                failure_type=type(launch_error).__name__,
            )
        elif any(path.is_symlink() for path in paths):
            record.update(state="failed", failure="trial_artifact_symlink")
        else:
            try:
                artifact = trial / "verifier/taskcompendium-result.json"
                result = json.loads(artifact.read_text())
                record["result"] = result
                record["artifact"] = str(artifact.relative_to(root))
                record["artifact_sha256"] = sha256(artifact)
                assert_case(case, result)
                provider_path = trial / "daytona-environment.json"
                provider = json.loads(provider_path.read_text())
                candidate_id = provider.get("sandbox_id")
                if (
                    provider.get("network_block_all") is not True
                    or not isinstance(candidate_id, str)
                    or not candidate_id
                ):
                    raise RuntimeError("candidate provider isolation is absent")
                if candidate_id in candidate_ids or candidate_id in sandbox_ids:
                    raise RuntimeError("candidate sandbox was reused")
                candidate_ids.add(candidate_id)
                record["candidate_provider_artifact"] = str(
                    provider_path.relative_to(root)
                )
                record["candidate_provider_sha256"] = sha256(provider_path)
                record["candidate_sandbox_id"] = candidate_id
                isolation = composite_isolation_record(
                    result,
                    checks,
                    config_sha256,
                    sandbox_ids,
                    candidate_ids,
                )
                record["private_isolation"] = isolation
                record["state"] = "passed"
            except Exception as error:  # noqa: BLE001 -- retain raw evidence and other cases
                record.update(
                    state="failed",
                    failure="case_evidence_invalid",
                    failure_type=type(error).__name__,
                )
        records.append(record)
    passed = all(record["state"] == "passed" for record in records)
    evidence = {
        "schema_version": "taskcompendium-composite-final-state-probe-v1",
        "state": "passed" if passed else "failed",
        "bundle_specification_sha256": sha256(bundle / "specification.json"),
        "package_manifest_sha256": sha256(package / "manifest.json"),
        "patched_taskcompendium_verifier_sha256": sha256(
            Path(native_verifier.__file__)
        ),
        "patched_taskcompendium_runner_sha256": sha256(Path(native_runner.__file__)),
        "patched_taskcompendium_lowering_sha256": sha256(
            Path(native_lowering.__file__)
        ),
        "probe_script_sha256": sha256(Path(__file__)),
        "preserved_specification_sha256": sha256(
            package / "composite-specification.json"
        ),
        "extension_marker_manifest_sha256": sha256(package / "manifest.json"),
        "ordinary_taskcompendium_decoder_rejected": True,
        "fail_closed": fail_closed,
        "bundle_config_sha256": sha256(bundle / "composite-verifier.json"),
        "bundle_renderings_sha256": sha256(bundle / "renderings.json"),
        "bundle_binding_sha256": sha256(bundle / "binding.json"),
        "cases": records,
        "fresh_private_sandboxes": sorted(sandbox_ids),
        "fresh_candidate_sandboxes": sorted(candidate_ids),
    }
    output = root / "composite-final-state-probe.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    if args.fixed_replay and passed:
        try:
            replay = run_fixed_replay_probe(root, bundle, package, records)
        except Exception as error:  # noqa: BLE001 - keep original and partial replay evidence.
            print(json.dumps({"state": "failed", "fixed_replay_error": type(error).__name__, "message": str(error)}))
            return 2
        fixed_output = root / "fixed-replay-probe.json"
        fixed_output.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")
        if replay["state"] != "ready":
            print(json.dumps({"state": replay["state"], "fixed_replay": str(fixed_output)}))
            return 2
    print(json.dumps({"state": evidence["state"], "evidence": str(output)}))
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--api-key-env", default="GLM_API_TOKEN")
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--provider", default="glm")
    parser.add_argument("--fixed-replay", action="store_true", help="regrade each captured machine check ten times")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

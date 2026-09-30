"""Run the composite machine-check plus native-judge protocol through Harbor."""

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
    Capability,
    Embedded,
    FileSubmission,
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

from capability_pipeline.composite_extension import (
    PATCHED_RUNNER_SHA256,
    PATCHED_VERIFIER_SHA256,
    install_extension_marker,
)
from capability_pipeline.composite_policy import (
    SCHEMA_VERSION,
    validate_composite_config,
)
from capability_pipeline.runtime import environment_config

REVISION = "dc6b501c8604bcd2e3c20c1e9947679845fdfef8"
IMAGE = "python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
QUESTION = """A release is compliant only if it names owner Priya, says to roll back by
reverting release 42, and cites a passed canary as validation. Assess the supplied release note."""
CRITERIA = (
    "The answer says the release is compliant and identifies Priya as the owner.",
    "The answer identifies reverting release 42 and the passed canary, and evidence.txt contains the matching workspace-proof record.",
    "The trusted reference context contains a graded compliance-gate machine result with reward 1.",
)
CASES = (
    {
        "id": "correct",
        "response": "Compliant: Priya owns it; revert release 42; the canary passed.",
        "evidence": "workspace-proof: Priya; release 42; canary passed",
        "expect": "pass",
    },
    {
        "id": "machine-gate-fail",
        "response": "Noncompliant: Priya owns it; revert release 42; the canary passed.",
        "evidence": "workspace-proof: Priya; release 42; canary passed",
        "expect": "machine_zero",
    },
    {
        "id": "critical-judge-fail",
        "response": "Compliant: Rowan owns it; revert release 42; the canary passed.",
        "evidence": "workspace-proof: Rowan; release 42; canary passed",
        "expect": "critical_zero",
    },
    {
        "id": "conditional-section-cap",
        "response": (
            "Compliant: Priya owns it; revert release 42; the canary passed. OVERFLAG"
        ),
        "evidence": "workspace-proof: Priya; release 42; canary passed",
        "expect": "cap",
    },
)


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
    if root.exists():
        shutil.rmtree(root)
    bundle, package = root / "task", root / "harbor"
    bundle.mkdir(parents=True)
    gate = b"""candidate="answer.txt"
if grep -q '^Compliant:' "$candidate"; then echo 1; else echo 0; fi
"""
    penalty = b"""candidate="answer.txt"
if grep -q 'OVERFLAG' "$candidate"; then echo 1; else echo 0; fi
"""
    resources = (
        Resource(
            "checks/gate.sh",
            (ResourceRole.VERIFIER,),
            Embedded(gate),
            executable=True,
        ),
        Resource(
            "checks/penalty.sh",
            (ResourceRole.VERIFIER,),
            Embedded(penalty),
            executable=True,
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
        JudgeView(transcript=True, files=("evidence.txt",)),
    )
    specification = TaskSpec(
        id="protocol/composite-verifier-live-probe-v1",
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
                    + " Write the assessment to answer.txt and its supporting record to evidence.txt."
                ),
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
    rendering = Rendering("files", FileSubmission("answer.txt"))
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
                        "id": "compliance-gate",
                        "role": "gate",
                        "script_path": "checks/gate.sh",
                        "args": [],
                        "image": IMAGE,
                        "timeout": 300,
                    },
                    {
                        "id": "overflag-penalty",
                        "role": "penalty",
                        "weight": 1.0,
                        "script_path": "checks/penalty.sh",
                        "args": [],
                        "image": IMAGE,
                        "timeout": 300,
                    },
                ],
                "judge": {
                    "criterion_weights": [1.0, 2.0, 1.0],
                    "critical_indices": [0],
                    "critical_min": 1.0,
                    "conditional_caps": [
                        {
                            "id": "overflag-section-zero",
                            "trigger_check_id": "overflag-penalty",
                            "trigger_min": 1.0,
                            "target_judge_indices": [1],
                            "max_fraction": 0.0,
                        }
                    ],
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
    lower_to_harbor(specification, (rendering,), binding, package)
    shutil.copy2(config_path, package / config_path.name)
    install_extension_marker(
        package,
        adapter_sha256=sha256(adapter),
        policy_sha256=sha256(policy),
        config_sha256=sha256(config_path),
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
    if case["expect"] == "pass" and reward != 1.0:
        raise RuntimeError(f"correct composite answer did not earn one: {result}")
    if case["expect"] == "machine_zero" and not (
        reward == 0.0
        and detail.get("judge_path") == "skipped_machine_gate"
        and detail.get("judge") is None
    ):
        raise RuntimeError(f"machine gate did not skip the native judge: {result}")
    if case["expect"] == "critical_zero" and not (
        reward == 0.0 and 0 in detail.get("failed_judge_critical_indices", [])
    ):
        raise RuntimeError(f"critical judge criterion did not force zero: {result}")
    if case["expect"] == "cap":
        penalty_results = {
            entry.get("id"): entry.get("reward")
            for entry in detail.get("machine_results", [])
        }
        if not (
            reward == 0.25
            and detail.get("judge_criterion_scores") == [1.0, 1.0, 1.0]
            and detail.get("effective_judge_criterion_scores") == [1.0, 0.0, 1.0]
            and detail.get("positive_weighted_sum") == 2.0
            and detail.get("penalty_weighted_sum") == 1.0
            and detail.get("positive_weight_denominator") == 4.0
            and penalty_results.get("overflag-penalty") == 1.0
            and detail.get("applied_conditional_caps", [{}])[0].get("id")
            == "overflag-section-zero"
        ):
            raise RuntimeError(
                f"conditional section cap did not preserve exact weighted reward: {result}"
            )


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
    rendering = renderings_from_json((bundle / "renderings.json").read_bytes())
    binding = HarborTaskBinding(
        DockerEnvironment(IMAGE, workdir="/app"),
        (HarnessToolBinding("terminal", "docker"),),
    )
    environment, _ = environment_config(binding, None)
    records, sandbox_ids, candidate_ids = [], [], []
    trials = root / "trials"
    for case in CASES:
        execution = resolve_harbor_execution(
            rendering,
            HarborExecutionConfig(binding, HarborLaunchConfig("replay")),
            environment,
            agent_kwargs={
                "response": "Files written.",
                "commands": [
                    f"printf '%s\\n' {shlex.quote(case['response'])} > answer.txt",
                    f"printf '%s\\n' {shlex.quote(case['evidence'])} > evidence.txt",
                ],
            },
            verifier_kwargs={"judge_api_key_env": args.api_key_env},
        )
        execution["verifier"]["import_path"] = (
            "capability_pipeline.composite_verifier:CompositeSemanticVerifier"
        )
        await run_trial(package, execution, trials, case["id"])
        artifact = trials / case["id"] / "verifier/taskcompendium-result.json"
        result = json.loads(artifact.read_text())
        assert_case(case, result)
        provider_path = trials / case["id"] / "daytona-environment.json"
        provider = json.loads(provider_path.read_text())
        if provider.get("network_block_all") is not True:
            raise RuntimeError(f"{case['id']} candidate environment allowed network")
        candidate_ids.append(provider["sandbox_id"])
        machine = result["detail"]["machine_results"]
        case_ids = [entry["detail"]["verifier_sandbox_id"] for entry in machine]
        if any(
            entry["detail"].get("verifier_isolation") != "daytona-network-block-all"
            for entry in machine
        ):
            raise RuntimeError(f"{case['id']} lacks network-blocked verifier evidence")
        sandbox_ids.extend(case_ids)
        records.append(
            {
                "id": case["id"],
                "result": result,
                "artifact": str(artifact.relative_to(root)),
                "artifact_sha256": sha256(artifact),
                "candidate_provider_artifact": str(provider_path.relative_to(root)),
                "candidate_provider_sha256": sha256(provider_path),
            }
        )
    if len(sandbox_ids) != len(set(sandbox_ids)):
        raise RuntimeError("composite probe reused a Daytona verifier sandbox")
    if len(candidate_ids) != len(set(candidate_ids)) or set(candidate_ids) & set(
        sandbox_ids
    ):
        raise RuntimeError("composite probe reused a candidate or verifier sandbox")
    evidence = {
        "schema_version": "taskcompendium-composite-probe-v1",
        "state": "passed",
        "bundle_specification_sha256": sha256(bundle / "specification.json"),
        "package_manifest_sha256": sha256(package / "manifest.json"),
        "patched_taskcompendium_verifier_sha256": sha256(
            Path(native_verifier.__file__)
        ),
        "patched_taskcompendium_runner_sha256": sha256(Path(native_runner.__file__)),
        "ordinary_taskcompendium_decoder_rejected": True,
        "cases": records,
        "fresh_private_sandboxes": sandbox_ids,
        "fresh_candidate_sandboxes": candidate_ids,
    }
    output = root / "composite-probe.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"state": "passed", "evidence": str(output)}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--api-key-env", default="GLM_API_TOKEN")
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--provider", default="glm")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

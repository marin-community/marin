"""Exercise explicit two-then-third composite judge consensus through Harbor."""

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
from capability_pipeline.composite_probe import (
    capture_case_artifacts,
    validate_case_result,
)
from capability_pipeline.runtime import environment_config

REVISION = "dc6b501c8604bcd2e3c20c1e9947679845fdfef8"
IMAGE = "python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
QUESTION = "Assess whether the release note makes a supported rollback recommendation."
CRITERIA = (
    "The answer recommends rolling back release 42.",
    "The answer gives a concrete source-grounded reason for rollback by identifying the canary as failed, and evidence.txt supports that characterization.",
)
BASELINE = {
    "id": "agreement-high",
    "response": "Rollback release 42 because the canary failed.",
    "evidence": "release 42 canary status: failed",
}
AMBIGUOUS = (
    (
        "one-timeout",
        "Consider rolling back release 42 because one canary check timed out.",
        "release 42 canary: 9 checks passed; 1 timed out",
    ),
    (
        "degraded",
        "Rollback release 42 may be prudent because the canary looks degraded.",
        "release 42 canary status: degraded; no terminal result recorded",
    ),
    (
        "mixed-signal",
        "I recommend rolling back release 42 based on the mixed canary signal.",
        "release 42 canary: primary passed; shadow failed",
    ),
    (
        "failed-word-qualified",
        "Rollback release 42: a canary sub-check failed, although the aggregate passed.",
        "release 42 canary aggregate: passed; latency sub-check: failed",
    ),
    (
        "unhealthy",
        "Rollback release 42 because the canary was not fully healthy.",
        "release 42 canary: 99% success; health threshold: 99.5%",
    ),
    (
        "inconclusive",
        "Rollback release 42 because the canary result was inconclusive.",
        "release 42 canary: sample too small to classify",
    ),
)
JUDGE_REQUEST_TIMEOUT = 300.0
VERIFIER_TIMEOUT = 900.0


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
    resources = (
        Resource(
            "checks/nonempty.sh",
            (ResourceRole.VERIFIER,),
            Embedded(b"test -s answer.txt && echo 1 || echo 0\n"),
            executable=True,
        ),
    )
    judge = JudgeConfig(
        JudgeModelPolicy(
            model=model,
            size_class="large",
            provider=provider,
            base_url=base_url,
            samples=2,
            temperature=1.0,
        ),
        JudgeView(files=("evidence.txt",)),
    )
    specification = TaskSpec(
        id="protocol/composite-consensus-live-probe-v1",
        requirements=TaskRequirements(
            (Capability.FILESYSTEM, Capability.SHELL, Capability.PROCESS),
            WorkspaceState(image=IMAGE, workdir="/app"),
        ),
        resources=resources,
        metadata=TaskMetadata(
            Source("protocol-probe", "1", "composite-consensus", REVISION)
        ),
        coverage_tags=("shape:answer", "subject:computing"),
        steps=(
            StepSpecification(
                instructions=(
                    QUESTION
                    + " Write the recommendation to answer.txt and the source record to evidence.txt."
                ),
                verifier=TaskTroveVerifier(
                    Mode.JUDGE,
                    {
                        "rubric": "checklist",
                        "criteria": list(CRITERIA),
                        "question": QUESTION,
                        "exact_gate": False,
                        "request_timeout": JUDGE_REQUEST_TIMEOUT,
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
                        "id": "nonempty",
                        "role": "gate",
                        "script_path": "checks/nonempty.sh",
                        "args": [],
                        "image": IMAGE,
                        "timeout": 300,
                    }
                ],
                "judge": {
                    "criterion_weights": [1.0, 1.0],
                    "critical_indices": [],
                    "critical_min": 1.0,
                    "conditional_caps": [],
                    "consensus": {
                        "mode": "two_then_third",
                        "initial_samples": 2,
                        "disagreement_tolerance": 0.0,
                        "resolution": "median",
                    },
                    "anchor_groups": [
                        {
                            "id": "rollback-support",
                            "criterion_indices": [0, 1],
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
    return bundle, package


async def run_case(case, package, rendering, binding, environment, trials, args):
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
    execution["verifier"]["override_timeout_sec"] = VERIFIER_TIMEOUT
    await run_trial(package, execution, trials, case["id"])
    root = trials / case["id"]
    artifact = root / "verifier/taskcompendium-result.json"
    provider_path = root / "daytona-environment.json"
    result, provider, probe_evidence = capture_case_artifacts(case["id"], root)
    validate_case_result(case["id"], result, provider)
    return {
        "id": case["id"],
        "result": result,
        "artifact": str(artifact.relative_to(trials.parent)),
        "artifact_sha256": sha256(artifact),
        "candidate_provider_artifact": str(provider_path.relative_to(trials.parent)),
        "candidate_provider_sha256": sha256(provider_path),
        "probe_case_evidence": str(probe_evidence.relative_to(trials.parent)),
        "probe_case_evidence_sha256": sha256(probe_evidence),
    }


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
    trials = root / "trials"
    records = []
    baseline = await run_case(
        BASELINE, package, rendering, binding, environment, trials, args
    )
    records.append(baseline)
    baseline_consensus = baseline["result"].get("detail", {}).get("judge_consensus")
    if not (
        baseline["result"].get("status") == "graded"
        and baseline_consensus
        and baseline_consensus.get("disagreed_indices") == []
        and baseline_consensus.get("native_grade_attempt_count") == 1
        and baseline_consensus.get("judge_vector_count") == 2
        and baseline_consensus.get("judge_call_count") == 4
        and baseline_consensus.get("anchor_scores", [{}])[0].get("score") == 2
    ):
        raise RuntimeError(f"agreement path did not retain exact consensus: {baseline}")

    observed = None
    for name, response, evidence in AMBIGUOUS[: args.max_ambiguous]:
        record = await run_case(
            {"id": f"disagreement-{name}", "response": response, "evidence": evidence},
            package,
            rendering,
            binding,
            environment,
            trials,
            args,
        )
        records.append(record)
        consensus = record["result"].get("detail", {}).get("judge_consensus")
        if (
            record["result"].get("status") == "graded"
            and consensus
            and consensus.get("disagreed_indices")
            and consensus.get("native_grade_attempt_count") == 2
            and consensus.get("judge_vector_count") == 3
            and consensus.get("judge_call_count") == 6
            and len(consensus.get("adjudicator_judgments", [])) == 2
        ):
            observed = record["id"]
            break
    if observed is None:
        raise RuntimeError("bounded ambiguous cases produced no real disagreement")

    candidate_ids = []
    verifier_ids = []
    for record in records:
        provider = json.loads(
            (root / record["candidate_provider_artifact"]).read_text()
        )
        candidate_ids.append(provider["sandbox_id"])
        verifier_ids.extend(
            row["detail"]["verifier_sandbox_id"]
            for row in record["result"]["detail"]["machine_results"]
        )
    if len(candidate_ids) != len(set(candidate_ids)) or len(verifier_ids) != len(
        set(verifier_ids)
    ):
        raise RuntimeError("consensus probe reused a candidate or verifier sandbox")
    if set(candidate_ids) & set(verifier_ids):
        raise RuntimeError("candidate and private-verifier sandboxes overlap")
    evidence = {
        "schema_version": "taskcompendium-composite-consensus-probe-v1",
        "state": "passed",
        "observed_disagreement_case": observed,
        "bundle_specification_sha256": sha256(bundle / "specification.json"),
        "package_manifest_sha256": sha256(package / "manifest.json"),
        "patched_taskcompendium_verifier_sha256": sha256(
            Path(native_verifier.__file__)
        ),
        "patched_taskcompendium_runner_sha256": sha256(Path(native_runner.__file__)),
        "composite_adapter_sha256": sha256(
            Path(__file__).parents[1] / "capability_pipeline/composite_verifier.py"
        ),
        "composite_policy_sha256": sha256(
            Path(__file__).parents[1] / "capability_pipeline/composite_policy.py"
        ),
        "cases": records,
        "fresh_candidate_sandboxes": candidate_ids,
        "fresh_private_sandboxes": verifier_ids,
    }
    output = root / "composite-consensus-probe.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"state": "passed", "evidence": str(output)}))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--provider", default="glm")
    parser.add_argument("--api-key-env", default="GLM_API_TOKEN")
    parser.add_argument("--max-ambiguous", type=int, default=len(AMBIGUOUS))
    args = parser.parse_args(argv)
    if not 1 <= args.max_ambiguous <= len(AMBIGUOUS):
        parser.error(f"--max-ambiguous must be between 1 and {len(AMBIGUOUS)}")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

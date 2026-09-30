"""Run calibration candidates through the pinned TaskCompendium judge adapter.

This module is launched inside the exact TaskCompendium environment. Credentials
remain in the process environment and are never written to task or result files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _grade_case(specification, renderings, case, api_key):
    from taskcompendium.grading import grade_attempt
    from taskcompendium.grading_paths import submission_relative
    from taskcompendium.judging import OpenAIJudgeClient
    from taskcompendium.resources import contained_path

    step_index = case.get("step_index", 0)
    with tempfile.TemporaryDirectory(
        prefix="capability-judge-calibration-"
    ) as temporary:
        workspace = Path(temporary)
        state = specification.requirements.state
        for path, content in case.get("workspace_files", {}).items():
            relative = submission_relative(
                path, state.workdir, state.additional_directories
            )
            target = contained_path(workspace, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        result = grade_attempt(
            specification,
            renderings[step_index],
            case["candidate"],
            workspace,
            tuple(case.get("transcript", [])),
            OpenAIJudgeClient(api_key),
            step_index,
        )
    return {
        "status": result.status.value,
        "reward": result.reward,
        "detail": result.detail,
    }


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_-]", "-", value)


async def _grade_composite_case(
    package,
    specification,
    renderings,
    binding,
    step_names,
    baseline_by_step,
    case,
    repeat,
    args,
    trials,
):
    from taskcompendium.execution import (
        HarborExecutionConfig,
        HarborLaunchConfig,
        HarborTaskBinding,
        HarnessToolBinding,
    )
    from taskcompendium.harbor.runner import run_trial
    from taskcompendium.lowering import resolve_harbor_execution

    from capability_pipeline.composite_timeout import composite_verifier_timeout
    from capability_pipeline.runtime import environment_config

    environment, kind = environment_config(binding, args.shellsim_bridge)
    commands = case.get("commands", [])
    replay_binding = binding
    if kind != "none":
        replay_binding = HarborTaskBinding(
            binding.environment, (HarnessToolBinding("terminal", kind),)
        )
    elif commands:
        raise ValueError(
            f"calibration case has commands without an executable environment: {case['id']}"
        )
    step_index = case.get("step_index", 0)
    if len(step_names) == 1:
        agent_kwargs = {"response": case["candidate"], "commands": commands}
    else:
        attempts = []
        for index in range(len(step_names)):
            selected = case if index == step_index else baseline_by_step[index]
            attempts.append(
                {
                    "response": selected["candidate"],
                    "commands": selected.get("commands", []),
                }
            )
        agent_kwargs = {"steps": attempts}
    execution = resolve_harbor_execution(
        renderings,
        HarborExecutionConfig(replay_binding, HarborLaunchConfig("replay")),
        environment,
        agent_kwargs=agent_kwargs,
        verifier_kwargs={"judge_api_key_env": args.api_key_env},
    )
    execution["verifier"]["import_path"] = (
        "capability_pipeline.composite_verifier:CompositeSemanticVerifier"
    )
    composite = json.loads((package / "composite-verifier.json").read_text())
    execution["verifier"]["override_timeout_sec"] = composite_verifier_timeout(
        composite, step_index
    )
    trial_name = _safe_name(f"{case['id']}-repeat-{repeat}")
    trial = await run_trial(package, execution, trials, trial_name)
    trial_root = trials / trial_name
    step_root = (
        trial_root / "steps" / step_names[step_index]
        if len(step_names) > 1
        else trial_root
    )
    artifact = step_root / "verifier" / "taskcompendium-result.json"
    if artifact.is_file():
        return json.loads(artifact.read_text())
    exception = trial.exception_info
    error = (
        exception.exception_type
        if exception is not None
        else "MissingCompositeVerifierResult"
    )
    raise RuntimeError(f"composite Harbor trial produced no grading artifact: {error}")


async def _execute_composite(args, fixture, bundle, specification, renderings):
    import msgspec
    from taskcompendium.execution import HarborTaskBinding

    package = Path(args.package)
    if not package.is_dir() or not (package / "manifest.json").is_file():
        raise ValueError("composite calibration needs the lowered Harbor package")
    binding = msgspec.json.decode(
        (bundle / "binding.json").read_bytes(), type=HarborTaskBinding
    )
    step_names = json.loads((package / "manifest.json").read_text())["step_names"]
    if len(step_names) != len(specification.steps):
        raise ValueError("Harbor package step names do not match the TaskSpec")
    baseline_by_step = {}
    for case in fixture["cases"]:
        lo, _ = case["expected_reward_range"]
        step = case.get("step_index", 0)
        if case["kind"] == "oracle" and lo >= 0.8:
            baseline_by_step.setdefault(step, case)
    if set(baseline_by_step) != set(range(len(step_names))):
        raise ValueError(
            "composite calibration needs an oracle baseline for every step"
        )
    trials = Path(args.output).parent / "taskcompendium-composite-trials"
    trials.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(case, repeat):
        key = f"{case['id']}:{repeat}"
        try:
            async with semaphore:
                result = await _grade_composite_case(
                    package,
                    specification,
                    renderings,
                    binding,
                    step_names,
                    baseline_by_step,
                    case,
                    repeat,
                    args,
                    trials,
                )
            return key, result, None
        except Exception as error:  # noqa: BLE001 -- retain ungraded infrastructure failures
            return (
                key,
                None,
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )

    completed = await asyncio.gather(
        *(
            one(case, repeat)
            for case in fixture["cases"]
            for repeat in range(args.repeats)
        )
    )
    results, failures = {}, {}
    for key, result, failure in completed:
        if failure is None:
            results[key] = result
        else:
            failures[key] = failure
    return results, failures


def execute(args):
    from taskcompendium.models import TaskTroveVerifier
    from taskcompendium.serialization import from_json, renderings_from_json
    from tasktrove_verify.spec import Mode

    bundle = Path(args.bundle)
    fixture = json.loads(Path(args.fixtures).read_text())
    specification = from_json((bundle / "specification.json").read_bytes())
    renderings = renderings_from_json((bundle / "renderings.json").read_bytes())
    if (bundle / "composite-verifier.json").is_file():
        if not os.environ.get(args.api_key_env):
            raise RuntimeError(
                f"judge credential environment variable is absent: {args.api_key_env}"
            )
        results, failures = asyncio.run(
            _execute_composite(args, fixture, bundle, specification, renderings)
        )
        Path(args.output).write_text(
            json.dumps(
                {
                    "mode": "taskcompendium-composite-harbor",
                    "results": results,
                    "failures": failures,
                },
                indent=2,
            )
            + "\n"
        )
        return
    for case in fixture["cases"]:
        step_index = case.get("step_index", 0)
        if not 0 <= step_index < len(specification.steps):
            raise ValueError(f"calibration step index is out of range: {step_index}")
        verifier = specification.steps[step_index].verifier
        if (
            not isinstance(verifier, TaskTroveVerifier)
            or verifier.mode is not Mode.JUDGE
        ):
            raise ValueError(f"calibration case targets a non-judge step: {case['id']}")
        if verifier.judge is None:
            raise ValueError(f"judge step lacks JudgeConfig: {case['id']}")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"judge credential environment variable is absent: {args.api_key_env}"
        )
    results, failures = {}, {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        jobs = {
            executor.submit(
                _grade_case, specification, renderings, case, api_key
            ): f"{case['id']}:{repeat}"
            for case in fixture["cases"]
            for repeat in range(args.repeats)
        }
        for future in as_completed(jobs):
            key = jobs[future]
            try:
                results[key] = future.result()
            except Exception as error:  # noqa: BLE001 -- retain ungraded infrastructure failures
                failures[key] = {
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
    Path(args.output).write_text(
        json.dumps({"results": results, "failures": failures}, indent=2) + "\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--package")
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-key-env", default="GLM_API_TOKEN")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--shellsim-bridge", default=os.environ.get("TASKCOMPENDIUM_SHELLSIM_BRIDGE")
    )
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.repeats < 1:
        parser.error("concurrency and repeats must be positive")
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

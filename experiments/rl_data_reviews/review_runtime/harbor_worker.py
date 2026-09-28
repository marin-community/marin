# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute the same Harbor Trial and result adapter used by MarinSkyRL."""

import argparse
import asyncio
import hashlib
import json
import os
import traceback
from pathlib import Path

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from review_io import append_event, capture_native_sources, native_calls, write_json
from skyrl_train.trajectory_runners.harbor.contracts import verification_from_harbor_result


async def run_task(data: dict, root: Path) -> dict:
    config, model = data["config"], data["model"]
    if model.get("api_key_env") is not None:
        os.environ["OPENAI_API_KEY"] = os.environ[model["api_key_env"]]
    task_dir = data["task"]["task_dir"]
    if data["task"].get("packed_task"):
        # Optional packed-source dependencies live in the selected native runtime.
        from marinskyrl.packed_tasks import PackedTaskMaterializer, PackedTaskReference  # noqa: PLC0415

        reference = PackedTaskReference(**data["task"]["packed_task"])
        materializer = PackedTaskMaterializer(root / "packed-task-cache")
        try:
            task_dir = str(materializer.materialize_batch([reference])[reference])
        finally:
            materializer.close()
    agent = config["harbor_agent"]
    kwargs = {
        **agent.get("kwargs", {}),
        "api_base": model["base_url"],
        "llm_call_kwargs": model["parameters"],
        "max_turns": config["max_turns"],
        "store_all_messages": True,
        "enable_episode_logging": True,
        "trajectory_dump_cadence": "per_turn",
    }
    trial_config = TrialConfig.model_validate(
        {
            "task": {"path": task_dir},
            "trial_name": "review-" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:20],
            "trials_dir": str(root),
            "agent": {
                "name": agent["name"],
                "model_name": "openai/" + model["name"],
                "kwargs": kwargs,
                "override_timeout_sec": config["agent_timeout"],
            },
            "environment": config["harbor_environment"],
            "verifier": config.get("harbor_verifier", {}),
        }
    )
    if trial_config.verifier.disable or trial_config.verifier.import_path:
        raise ValueError("Review must use the task native verifier, enabled")
    append_event(root / "verifier-trace.jsonl", {"event": "trial_start"})
    trial = await Trial.create(trial_config)
    result = await trial.run()
    write_json(root / "harbor-result.json", result.model_dump(mode="json"))
    append_event(
        root / "verifier-trace.jsonl",
        {
            "event": "trial_end",
            "verification": verification_from_harbor_result(result),
            "exception": result.exception_info.model_dump(mode="json") if result.exception_info else None,
        },
    )
    return {
        "verification": verification_from_harbor_result(result),
        "verifier_executed": any(
            timing is not None and timing.started_at is not None
            for timing in [result.verifier, *(step.verifier for step in result.step_results or [])]
        ),
        "done": result.exception_info is None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with native_calls() as called:
        try:
            result = asyncio.run(run_task(json.loads(args.input.read_text()), args.output))
        except Exception as error:
            append_event(
                args.output / "verifier-trace.jsonl",
                {
                    "event": "execution_error",
                    "type": type(error).__name__,
                    "detail": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            result = {
                "verification": {
                    "status": "error",
                    "score": None,
                    "passed": None,
                    "reason": str(error),
                    "diagnostics": {"exception_type": type(error).__name__},
                },
                "verifier_executed": False,
                "done": False,
            }
        finally:
            capture_native_sources(args.output, called)
    write_json(args.output / "attempt.json", result)


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real Pi/Daytona recovery using a deterministic diagnostic model.

The injected context rejection is intentional. This fixture checks the runtime
protocol and original verifier; it is not a benchmark or model-quality result.
"""

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path

from omegaconf import OmegaConf
from rolloutengine.engine import RolloutEngine
from rolloutengine.models import ContextWindowExceeded, ModelTurn, RolloutLimits, TaskRolloutSpec, TraceMode
from skyrl_train.dataset.harbor_import import harbor_trial_task
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.training_record import training_rollout
from taskcompendium.models import Source

from experiments.evaluation.mini_policy_runtime.profile import pi_agent_config


class Decoder:
    def decode(self, token_ids):
        return "diagnostic"


async def main(mode: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        task_dir = root / "task"
        (task_dir / "environment").mkdir(parents=True)
        (task_dir / "tests").mkdir()
        (task_dir / "environment/Dockerfile").write_text("FROM ubuntu:24.04\nRUN mkdir -p /app\nWORKDIR /app\n")
        (task_dir / "instruction.md").write_text("Use bash to inspect /app, then write exactly 12 to /app/answer.")
        (task_dir / "task.toml").write_text(
            "[agent]\ntimeout_sec=1800\n[verifier]\ntimeout_sec=300\n[environment]\ncpus=1\nmemory_mb=1024\nstorage_mb=2048\n"
        )
        (task_dir / "tests/test.sh").write_text(
            "#!/bin/bash\nmkdir -p /logs/verifier\n"
            'if [ "$(cat /app/answer 2>/dev/null)" = 12 ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt\n'
        )
        task = harbor_trial_task(
            task_dir, source=Source(dataset="runtime-control", revision="1", row="context", importer_revision="1")
        )
        limits = RolloutLimits(
            max_turns=8,
            max_total_tokens=None,
            model_turn_timeout=None,
            tool_turn_timeout=None,
            total_turn_timeout=10 if mode == "timeout" else 1800,
            attempt_timeout=2460,
            verifier_timeout=300,
            cleanup_timeout=60,
        )
        spec = TaskRolloutSpec(
            task=task,
            harness="harbor_trial",
            task_machine=None,
            verifier_machine=None,
            limits=limits,
            trace=TraceMode.MESSAGES,
        )
        backend = AgentInferenceBackend("diagnostic", 65536, tokenize, Decoder())
        runtime = InstalledAgentRuntime(
            InstalledAgentSettings(
                agent=pi_agent_config(),
                output_path=str(Path(os.environ["IRIS_OUTPUT_DIR"]) / "native-recovery-evidence"),
                setup_timeout=360,
                run_timeout=1800,
                cleanup_timeout=60,
                service_startup_timeout=60,
                service_shutdown_timeout=60,
            ),
            backend,
            16384,
        )
        settings = HarborTaskSettings.from_config(
            OmegaConf.create(
                {
                    "harbor": {
                        "environment_type": "daytona",
                        "daytona_api_key_env": "DAYTONA_EVAL_API_KEY",
                        "eval_daytona_api_key_env": "DAYTONA_EVAL_API_KEY",
                    }
                }
            )
        )
        await runtime.startup()
        try:
            harness = runtime.original_harness(spec, "diagnostic", settings, "eval")
            outer = spec.model_copy(update={"limits": limits.model_copy(update={"total_turn_timeout": None})})
            calls = []

            async def model(request):
                calls.append(request)
                index = len(calls)
                if index == 2 and mode == "timeout":
                    await asyncio.sleep(3600)
                if index == 3:
                    raise ContextWindowExceeded(
                        "This request exceeds the maximum context length of 65536 tokens. Compact and retry."
                    )
                if index in {1, 2, 5}:
                    command = "ls /app" if index < 5 else "printf 12 > /app/answer"
                    message = {
                        "role": "assistant",
                        "content": "History detail. " * 4000 if index < 3 and mode == "context" else "",
                        "tool_calls": [
                            {
                                "id": f"call-{index}",
                                "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                            }
                        ],
                    }
                else:
                    message = {
                        "role": "assistant",
                        "content": (
                            "Task: write 12 to /app/answer using bash. No work completed." if index == 4 else "Done."
                        ),
                    }
                return ModelTurn(message, (10, 11), (20,), None, "stop")

            result = training_rollout(await RolloutEngine({}, {"harbor_trial": harness}).run_task(outer, model), outer)
            if mode == "timeout":
                assert len(calls) >= 2, "The controlled hang must follow a completed model turn"
                assert result.grade.reward == 0, result.grade
                assert result.grade.diagnostics["harbor_exception_type"] == "AgentTimeoutError"
                print(
                    json.dumps(
                        {
                            "scope": "native-agent-timeout-control",
                            "agent_timeout": 10,
                            "original_verifier_reward": 0,
                            "verified_after_agent_timeout": True,
                        }
                    )
                )
                return
            assert result.grade.reward == 1, result.grade
            assert len(calls) == 6, len(calls)
            assert "summar" in str(calls[3].messages).lower(), "Pi did not request compaction after the rejection"
            print(
                json.dumps(
                    {
                        "scope": "injected-context-runtime-control",
                        "real_pi": True,
                        "daytona": True,
                        "requests": len(calls),
                        "recovered_after_rejection": True,
                        "original_verifier_reward": 1,
                    }
                )
            )
        finally:
            await runtime.shutdown()


async def tokenize(payload):
    return {"tokens": [10, 11]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("context", "timeout"), default="context")
    asyncio.run(main(parser.parse_args().mode))

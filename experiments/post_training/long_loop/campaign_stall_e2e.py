# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the campaign's authored stall limit and final-state verifier locally."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import yaml
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.models import ContextRecovery, ModelTurn, Prompt, RolloutLimits
from rolloutengine.test_support import arithmetic_task, engine, rollout_spec

REFERENCE = Path(__file__).with_name("campaign") / "mochi131.yaml"
SESSION_DEFAULTS = (
    Path(__file__).parents[3] / "vendor/skyrl/skyrl-train/skyrl_train/config/task_session_config/default.yaml"
)


def campaign_limits() -> RolloutLimits:
    defaults = yaml.safe_load(SESSION_DEFAULTS.read_text())["task_spec"]["session"]
    recipe = yaml.safe_load(REFERENCE.read_text())["rl"]["recipe"]
    session = {**defaults, **recipe["environment"]["task_sessions"]["task_spec"]["session"]}
    session.pop("command_timeout")
    session.pop("shell_tool")
    return RolloutLimits.model_validate(session)


class LoopingHarness:
    context_recovery = ContextRecovery.NEVER

    def check(self, task, limits) -> None:
        pass

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        await session.start(Prompt(({"role": "user", "content": "What is six plus six?"},), {}))
        for _ in range(8):
            await session.reply(({"role": "user", "content": "unchanged observation"},))


class ConstantModel:
    def __init__(self, answer: str):
        self.answer = answer

    async def complete(self, request) -> ModelTurn:
        served = request.continuation.served_token_ids if request.continuation is not None else (10, 11)
        return ModelTurn({"role": "assistant", "content": self.answer}, served, (20,), (-0.5,), "stop")


async def run_proof() -> dict:
    """Verify that stopping preserves both a positive and a zero verifier grade."""
    limits = campaign_limits()
    assert limits.repeat_stall_batches == 3
    cases = []
    for answer, expected in (("12", 1.0), ("11", 0.0)):
        spec = rollout_spec(arithmetic_task(), harness="stall-proof", **limits.model_dump())
        result = await engine(ConstantModel(answer), harnesses={"stall-proof": LoopingHarness()}).run_task(spec)
        turns = sum(len(conversation.steps) for conversation in result.rollout.conversations)
        assert result.rollout.stop_reason == "repeat_stall"
        assert turns == 3
        assert result.grade.reward == expected
        cases.append(
            {
                "answer": answer,
                "stop_reason": result.rollout.stop_reason,
                "turns": turns,
                "verifier_reward": result.grade.reward,
            }
        )
    return {
        "status": "PASS",
        "campaign_config_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        "limits": limits.model_dump(),
        "cases": cases,
        "live_model_inference": False,
        "fired_fraction": sum(case["stop_reason"] == "repeat_stall" for case in cases) / len(cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    receipt = asyncio.run(run_proof())
    output = StoragePath(args.output)
    output.parent.mkdirs()
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Harbor lowerings through MarinSkyRL's trajectory consumer interface.

Run in the pinned MarinSkyRL environment with TaskCompendium's Harbor extra.
This adapter does not construct a trainer, Ray worker, or inference engine.
"""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.trajectory_processing import (
    get_response_ids_and_loss_mask_from_messages,
    normalize_token_ids,
    prepare_trajectory_request,
)
from skyrl_train.trajectory_runners.types import TrajectoryBatch, TrajectoryRequestBatch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from taskcompendium.harbor.generation import GenerationRequest, generate_attempts, write_attempts
from taskcompendium.models import Outcome

ENV_CLASS = "taskcompendium_harbor"


class UngradedBatchError(RuntimeError):
    """Semantic attempts were retained but cannot form a numeric SkyRL batch."""


class TaskCompendiumTrajectoryRunner(TrajectoryRunner):
    """Resolve per-task Harbor launches and return graded, reconstructed tokens."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, output_dir: Path, *, concurrency: int):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.last_attempts = []

    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        extras = input_batch.get("env_extras")
        identities = input_batch.get("trajectory_ids")
        size = len(input_batch["prompts"])
        if (
            extras is None
            or identities is None
            or not (len(extras) == len(identities) == len(input_batch["env_classes"]) == size)
        ):
            raise ValueError("SkyRL requests require aligned launches, environment classes, and trajectory IDs")
        if any(name != ENV_CLASS for name in input_batch["env_classes"]):
            raise ValueError(f"This runner requires env_class={ENV_CLASS}")
        if input_batch.get("sampling_params"):
            raise ValueError("Set generation budgets in each explicit Harbor launch, not batch sampling parameters")
        requests = []
        for prompt, launch, identity in zip(input_batch["prompts"], extras, identities, strict=True):
            task_dir = Path(launch["task_dir"])
            if prompt != [{"role": "user", "content": str(task_dir)}]:
                raise ValueError("Harbor request prompts must identify the corresponding lowering package")
            requests.append(
                GenerationRequest(task_dir, launch["execution"], identity.instance_id, identity.repetition_id)
            )
        attempts = await generate_attempts(requests, self.output_dir / "trials", concurrency=self.concurrency)
        self.last_attempts = attempts
        archive = self.output_dir / f"attempts-{uuid4().hex}.jsonl"
        write_attempts(attempts, archive)
        if any(attempt.status != Outcome.GRADED or attempt.reward is None for attempt in attempts):
            raise UngradedBatchError(f"Ungraded attempts retained in {archive}; no numeric batch was returned")
        prompt_ids, responses, masks = [], [], []
        for attempt in attempts:
            messages = list(attempt.messages)
            template_kwargs = {
                **attempt.chat_template_kwargs,
                **({"tools": list(attempt.tools)} if attempt.tools else {}),
            }
            first_assistant = next(
                (index for index, message in enumerate(messages) if message["role"] == "assistant"), None
            )
            if first_assistant is None:
                raise ValueError(f"Graded trial has no assistant trace: {attempt.trial_dir}")
            prompt_ids.append(
                normalize_token_ids(
                    self.tokenizer.apply_chat_template(
                        messages[:first_assistant], tokenize=True, add_generation_prompt=False, **template_kwargs
                    )
                )
            )
            ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
                messages[first_assistant:],
                self.tokenizer,
                rollout_logprobs_required=False,
                tito_full=False,
                chat_template_kwargs=template_kwargs,
            )
            responses.append(ids)
            masks.append(loss_mask)
        rewards = [attempt.reward for attempt in attempts]
        return {
            "prompt_token_ids": prompt_ids,
            "response_ids": responses,
            "rewards": rewards,
            "unshaped_rewards": list(rewards),
            "loss_masks": masks,
            "stop_reasons": None,
            "exception_types": [None] * size,
            "error_treatments": [None] * size,
            "trajectory_ids": list(identities),
            "rollout_metrics": {"taskcompendium/reconstructed_trajectories": float(size)},
            "rollout_logprobs": None,
            "rollout_routed_experts": None,
            "reward_shaping_components": None,
            "reward_shaping_loop_spans": None,
            "loop_advantages": None,
            "reward_shaping_versions": None,
            "verifier_tests": None,
            "teacher_evidence": None,
            "distillation": None,
            "token_level_shaping": None,
            "response_span_tags": None,
            "teacher_route_keys": None,
            "is_last_step": [True] * size,
            "exclude_from_baseline": [False] * size,
            "actual_global_step": None,
        }


def request_batch(rows: list[dict], *, repetitions: int) -> TrajectoryRequestBatch:
    """Prepare the same request envelope used by the pinned SkyRL consumer."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    dataset_rows = [
        {
            "uid": row["uid"],
            "prompt": [{"role": "user", "content": row["task_dir"]}],
            "env_class": ENV_CLASS,
            "env_extras": {"task_dir": row["task_dir"], "execution": row["execution"]},
        }
        for row in rows
    ]
    batch, _ = prepare_trajectory_request(dataset_rows, repetitions, {}, ENV_CLASS, "eval", 0)
    return batch


async def generate(args) -> None:
    rows = json.loads(args.requests.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    runner = TaskCompendiumTrajectoryRunner(tokenizer, args.output, concurrency=args.concurrency)
    await runner.startup()
    try:
        batch = await runner.run(request_batch(rows, repetitions=args.repetitions))
    finally:
        await runner.shutdown()
    # Dataclass trajectory identities need an explicit JSON wire projection.
    record = {**batch, "trajectory_ids": [vars(identity) for identity in batch["trajectory_ids"]]}
    with (args.output / "batch.json").open("x") as output:
        json.dump(record, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests", type=Path, required=True, help="JSON array of uid, task_dir, and resolved execution"
    )
    parser.add_argument(
        "--tokenizer", required=True, help="Tokenizer repository or local path; separate from served model ID"
    )
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    asyncio.run(generate(parser.parse_args()))


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt mixed TaskCompendium Harbor trials to SkyRL trajectory batches."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.trajectory_processing import (
    get_response_ids_and_loss_mask_from_messages,
    normalize_token_ids,
    prepare_trajectory_request,
)
from skyrl_train.trajectory_runners.types import TrajectoryBatch, TrajectoryRequestBatch
from transformers import PreTrainedTokenizerBase

from taskcompendium.harbor.generation import GeneratedAttempt, GenerationRequest, generate_attempts, write_attempts
from taskcompendium.models import Outcome

ENV_CLASS = "taskcompendium_harbor"


class UngradedBatchError(RuntimeError):
    """A verifier or task failure prevented construction of a numeric batch."""


def _trainer_reward(attempt: GeneratedAttempt) -> tuple[float, str | None, str | None]:
    """Apply the explicit training-admission policy without changing semantic results."""
    if attempt.status is Outcome.GRADED and attempt.reward is not None:
        return attempt.reward, None, None
    if attempt.status is Outcome.EXTRACTION_ERROR and attempt.messages:
        return 0.0, Outcome.EXTRACTION_ERROR.value, "zero"
    raise UngradedBatchError(
        f"{attempt.instance_id}/{attempt.repetition_id} has semantic status {attempt.status.value}; "
        "the retained attempt cannot enter a numeric training batch"
    )


class TaskCompendiumTrajectoryRunner(TrajectoryRunner):
    """Run each lowering with its own Harbor launch and reconstruct trainable tokens."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, output_dir: Path, *, concurrency: int):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.last_attempts: list[GeneratedAttempt] = []

    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        del disable_tqdm
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
        try:
            admission = [_trainer_reward(attempt) for attempt in attempts]
        except UngradedBatchError as error:
            raise UngradedBatchError(f"{error}; all attempts remain archived at {archive}") from error

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
                raise UngradedBatchError(f"Retained trial has no assistant trace: {attempt.trial_dir}")
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

        rewards, exception_types, error_treatments = zip(*admission, strict=True)
        return {
            "prompt_token_ids": prompt_ids,
            "response_ids": responses,
            "rewards": list(rewards),
            "unshaped_rewards": list(rewards),
            "loss_masks": masks,
            "stop_reasons": None,
            "exception_types": list(exception_types),
            "error_treatments": list(error_treatments),
            "trajectory_ids": list(identities),
            "rollout_metrics": {
                "taskcompendium/reconstructed_trajectories": float(size),
                "taskcompendium/extraction_errors_zeroed": float(
                    sum(attempt.status is Outcome.EXTRACTION_ERROR for attempt in attempts)
                ),
            },
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
    """Prepare TaskCompendium rows with SkyRL's production request envelope."""
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

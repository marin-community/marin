# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared RL training observability helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from collections.abc import Callable

from iris.marin_fs import url_to_fs
from levanter import callbacks
from levanter.trainer import Trainer
from levanter.utils.fsspec_utils import exists
from transformers import PreTrainedTokenizer

from marin.rl.replay_buffer import RolloutWithCount

logger = logging.getLogger(__name__)

SAMPLE_TABLE_COLUMNS = ["step", "prompt_id", "prompt", "response", "reward"]


@dataclass(frozen=True)
class TrainingSamplePreview:
    """Minimal decoded-sample metadata for W&B logging."""

    prompt_id: str
    prompt_tokens: list[int]
    response_tokens: list[int]
    reward: float


def training_sample_preview_path(batch_path: str) -> str:
    """Return the sidecar preview path for one materialized training batch."""
    if batch_path.endswith(".pkl"):
        return f"{batch_path[:-4]}_samples.json"
    return f"{batch_path}.samples.json"


def write_training_sample_previews(path: str, previews: list[TrainingSamplePreview]) -> None:
    """Write sample previews as JSON."""
    fs, fs_path = url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0] if "/" in fs_path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    payload = [asdict(preview) for preview in previews]
    with fs.open(fs_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_training_sample_previews(path: str) -> list[TrainingSamplePreview]:
    """Read sample previews from JSON."""
    fs, fs_path = url_to_fs(path)
    with fs.open(fs_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [TrainingSamplePreview(**item) for item in payload]


def maybe_read_training_sample_previews(path: str) -> list[TrainingSamplePreview] | None:
    """Return sample previews when the sidecar exists."""
    if not exists(path):
        return None
    return read_training_sample_previews(path)


def training_sample_previews_from_rollouts(rollouts: list[RolloutWithCount]) -> list[TrainingSamplePreview]:
    """Match async RL sample-table selection for the first five prompt ids."""
    prompts: dict[str, list[RolloutWithCount]] = {}
    for rollout_with_count in rollouts:
        prompt_id = rollout_with_count.rollout.env_example_id
        prompts.setdefault(prompt_id, []).append(rollout_with_count)

    previews: list[TrainingSamplePreview] = []
    for prompt_id in list(prompts.keys())[:5]:
        for rollout_with_count in prompts[prompt_id]:
            rollout = rollout_with_count.rollout
            previews.append(
                TrainingSamplePreview(
                    prompt_id=prompt_id,
                    prompt_tokens=rollout.prompt_tokens.tolist(),
                    response_tokens=rollout.response_tokens.tolist(),
                    reward=float(rollout.episode_reward),
                )
            )
    return previews


def log_training_sample_table(
    trainer: Trainer,
    *,
    tokenizer: PreTrainedTokenizer,
    step: int,
    previews: list[TrainingSamplePreview],
) -> None:
    """Log the async-style `train/samples` W&B table."""
    if not previews:
        return

    import wandb

    rows = []
    for preview in previews:
        rows.append(
            [
                step,
                preview.prompt_id,
                tokenizer.decode(preview.prompt_tokens, skip_special_tokens=False),
                tokenizer.decode(preview.response_tokens, skip_special_tokens=False),
                preview.reward,
            ]
        )

    trainer.tracker.log(
        {"train/samples": wandb.Table(columns=SAMPLE_TABLE_COLUMNS, data=rows)},
        step=step,
    )


def configure_rl_training_metric_hooks(
    trainer: Trainer,
    *,
    tokenizer: PreTrainedTokenizer,
    tokens_per_example: int,
    flops_per_example: float | None,
    batch_schedule: int,
    batch_prep_time: Callable[[], float],
    sample_previews: Callable[[], list[TrainingSamplePreview] | None],
) -> None:
    """Install the async RL metric hooks on a trainer."""

    def _log_step_timing(info):
        prep_duration = batch_prep_time()
        forward_backward_duration = max(0.0, info.step_duration - prep_duration)
        metrics = {
            "throughput/step_duration_seconds": info.step_duration,
            "throughput/batch_prep_duration_seconds": prep_duration,
            "throughput/forward_backward_duration_seconds": forward_backward_duration,
            "train/loss": float(info.loss),
        }
        trainer.tracker.log(metrics, step=info.step)
        logger.info(
            "Training step %d completed: duration=%.2fs (batch_prep=%.2fs, fwd_bwd=%.2fs), loss=%.4f",
            info.step,
            info.step_duration,
            prep_duration,
            forward_backward_duration,
            info.loss,
        )

    trainer.add_hook(_log_step_timing, every=1)

    def _log_samples_hook(info):
        previews = sample_previews()
        if previews is None:
            return
        log_training_sample_table(
            trainer,
            tokenizer=tokenizer,
            step=info.step,
            previews=previews,
        )

    trainer.add_hook(_log_samples_hook, every=1)
    trainer.add_hook(
        callbacks.log_performance_stats(
            tokens_per_example=tokens_per_example,
            batch_schedule=batch_schedule,
            flops_per_example=flops_per_example,
            prefix="throughput",
        ),
        every=1,
    )

    # Explicitly commit each step so one-step alternating phases still flush a history row.
    def _commit_step_metrics(info):
        trainer.tracker.log({}, step=info.step, commit=True)

    trainer.add_hook(_commit_step_metrics, every=1)

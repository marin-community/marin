# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Levanter-based LM Evaluation Harness — in-process TPU shim."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections.abc import Sequence

import jmp
import levanter
import levanter.eval_harness as eval_harness
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from rigging.filesystem import filesystem as marin_filesystem

from marin.evaluation.evaluation_config import EvalTaskConfig, convert_to_levanter_task_config

logger = logging.getLogger(__name__)


def run_levanter_lm_eval(
    *,
    name: str,
    path: str,
    evals: Sequence[EvalTaskConfig],
    output_path: str,
    max_eval_instances: int | None = None,
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
) -> None:
    """Run Levanter's in-process lm-eval harness on an HF checkpoint.

    Args:
        name: Human-readable model name (used for wandb run naming).
        path: HF checkpoint path readable by Levanter (GCS or local).
        evals: Tasks to run.
        output_path: Where to write `results.json`.
        max_eval_instances: Optional cap on per-task instance count.
        apply_chat_template: Whether the model was trained with a chat template.
        wandb_tags: W&B run tags.
    """
    wandb_name = f"{name}_lmeval_{'-'.join(task.name for task in evals)}"
    logger.info(f"WandB Run Name: {wandb_name}")
    logger.info(f"Running eval harness on model: {path}")

    # NOTE: per-device batch size 1 fits 8B on v4-8; size this up if needed.
    trainer_config = TrainerConfig(
        tracker=WandbConfig(project="marin", tags=wandb_tags, name=wandb_name),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        per_device_eval_parallelism=1,
    )
    model_config = HFCheckpointConverter.from_hf(path).LevConfigClass()
    tasks = convert_to_levanter_task_config(evals)
    logger.info(f"Tasks: {tasks}")

    eval_config = eval_harness.EvalHarnessMainConfig(
        eval_harness=eval_harness.LmEvalHarnessConfig(
            task_spec=tasks,
            max_examples=max_eval_instances,
            log_samples=False,
            max_length=4096,
            apply_chat_template=apply_chat_template,
            confirm_run_unsafe_code=True,
            sample_logging=eval_harness.SampleLoggingConfig(max_samples_per_benchmark=20),
        ),
        tokenizer=path,  # Levanter picks up the tokenizer from the checkpoint path
        checkpoint_path=path,
        checkpoint_is_hf=True,
        trainer=trainer_config,
        model=model_config,
    )

    try:
        results = eval_harness.run_eval_harness_main(eval_config)
        results_path = os.path.join(output_path, "results.json")
        logger.info(f"Uploading results to GCS: {results_path}")
        fs = marin_filesystem("gcs")
        with fs.open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)
        levanter.tracker.current_tracker().finish()
        logger.info("Upload completed successfully.")
    except Exception as e:
        logger.error(f"Error running eval harness: {e}")
        raise


def _json_default(value):
    """Best-effort JSON serialization for objects returned by the eval harness."""
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, set):
        return list(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict()
        except Exception:
            pass
    return repr(value)

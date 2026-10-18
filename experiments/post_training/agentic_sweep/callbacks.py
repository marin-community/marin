"""Exact epoch accounting and exports for the fixed-batch SFT sweep."""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from levanter.adaptor import AdaptorConfig, NoAdaptorConfig
from levanter.checkpoint import CheckpointRetention
from levanter.data.text.datasets import LmDataConfig
from levanter.tracker import log
from levanter.tracker.tracker import NoopTracker, TrackerConfig
from rigging.filesystem.storage_path import StoragePath, prefix_join

logger = logging.getLogger(__name__)

PLATEAU_TOTALS = ("plateau_kept_tokens", "plateau_kept_loss_sum", "plateau_dropped_tokens", "plateau_dropped_loss_sum")


class EpochTracker(NoopTracker):
    name = "epoch_accounting"

    def __init__(self, output_path: str, steps_per_epoch: int, tokens_per_epoch: int, start_step: int = 0):
        if start_step % steps_per_epoch:
            raise ValueError("Epoch accounting can resume only at an epoch boundary")
        self.output_path = output_path
        self.steps_per_epoch = steps_per_epoch
        self.tokens_per_epoch = tokens_per_epoch
        self.next_token_step = start_step
        self.loss_sum = 0.0
        self.token_sum = 0.0
        self.plateau_sums = dict.fromkeys(PLATEAU_TOTALS, 0.0)
        self.epoch_rows: dict[int, dict[str, Any]] = {}
        self.published_epochs: set[int] = set()

    def log(self, metrics: Mapping[str, Any], *, step: int | None, commit: bool | None = None):
        if step is None:
            return
        epoch = (step + 1) // self.steps_per_epoch
        boundary = (step + 1) % self.steps_per_epoch == 0
        if "train/supervised_tokens" in metrics:
            if step != self.next_token_step:
                raise ValueError(f"Missing or duplicated training statistics: {step} != {self.next_token_step}")
            self.next_token_step += 1
            self.loss_sum += float(metrics["train/loss_sum"])
            self.token_sum += float(metrics["train/supervised_tokens"])
            for name in PLATEAU_TOTALS:
                self.plateau_sums[name] += float(metrics.get(f"train/{name}", 0.0))
            if boundary:
                difference = self.token_sum - self.tokens_per_epoch
                if abs(difference) > 0.001 * self.tokens_per_epoch:
                    logger.warning(
                        "Window supervised tokens differ from manifest: actual=%s expected=%s",
                        self.token_sum,
                        self.tokens_per_epoch,
                    )
                self.epoch_rows.setdefault(epoch, {}).update(
                    train_loss=self.loss_sum / self.token_sum,
                    supervised_tokens=int(self.token_sum),
                    manifest_supervised_tokens=self.tokens_per_epoch,
                    supervised_token_difference=int(difference),
                )
                if "train/plateau_kept_tokens" in metrics:
                    self.epoch_rows[epoch].update(plateau_epoch_row(self.plateau_sums, self.token_sum))
                self.loss_sum = 0.0
                self.token_sum = 0.0
                self.plateau_sums = dict.fromkeys(PLATEAU_TOTALS, 0.0)
        row = self.epoch_rows.get(epoch, {})
        if boundary and epoch not in self.published_epochs and "train_loss" in row:
            row.update(epoch=epoch, completed_updates=step + 1, checkpoint_step=step)
            if jax.process_index() == 0:
                text = "".join(
                    json.dumps(self.epoch_rows[key], sort_keys=True) + "\n" for key in sorted(self.epoch_rows)
                )
                StoragePath(prefix_join(self.output_path, "epoch-metrics.jsonl")).write_text(text)
            self.published_epochs.add(epoch)
            log({f"epoch/{key}": value for key, value in row.items()}, step=step, commit=commit)


def plateau_epoch_row(sums: Mapping[str, float], supervised_tokens: float) -> dict[str, float]:
    """Share of supervised tokens dropped in the epoch, and the mean loss on kept and dropped tokens."""
    kept, dropped = sums["plateau_kept_tokens"], sums["plateau_dropped_tokens"]
    return {
        "plateau_kept_tokens": int(kept),
        "plateau_dropped_tokens": int(dropped),
        "plateau_dropped_fraction": dropped / supervised_tokens,
        "plateau_kept_loss": sums["plateau_kept_loss_sum"] / kept if kept else float("nan"),
        "plateau_dropped_loss": sums["plateau_dropped_loss_sum"] / dropped if dropped else float("nan"),
    }


@TrackerConfig.register_subclass("agentic_sweep_epochs")
@dataclass
class EpochTrackerConfig(TrackerConfig):
    output_path: str
    steps_per_epoch: int
    tokens_per_epoch: int
    start_step: int = 0

    def init(self, run_id: str | None) -> EpochTracker:
        return EpochTracker(
            self.output_path,
            self.steps_per_epoch,
            self.tokens_per_epoch,
            self.start_step,
        )


@AdaptorConfig.register_subclass("agentic_sweep_full")
@dataclass(frozen=True)
class SweepExportsConfig(NoAdaptorConfig):
    validation_data: LmDataConfig
    steps_per_epoch: int
    sequence_length: int

    def install_export_hooks(self, *, trainer, converter, tokenizer, export) -> None:
        if converter is None or export.hf_save_path is None:
            raise ValueError("The sweep requires an HF converter and export path")
        if trainer.config.microbatch_size != trainer.config.train_batch_size:
            raise ValueError("The sweep forbids separately normalized microbatch accumulation")
        if self.steps_per_epoch < 4:
            raise ValueError("Sweep epochs must have at least four batches")
        total_steps = trainer.config.num_train_steps
        last_exported = -1

        def epoch_hook(info, force=False):
            nonlocal last_exported
            del force
            completed = info.next_step
            if completed + 1 <= total_steps and (completed + 1) % self.steps_per_epoch == 0:
                trainer.request_checkpoint(CheckpointRetention.PERMANENT)
            if completed != total_steps or completed == last_exported:
                return
            converter.save_pretrained(
                info.eval_model,
                prefix_join(export.hf_save_path, f"step-{info.step}"),
                dtype=jnp.bfloat16,
                generation_config=export.generation_config,
                save_reference_code=False,
            )
            last_exported = completed

        trainer.add_hook(epoch_hook, every=1)

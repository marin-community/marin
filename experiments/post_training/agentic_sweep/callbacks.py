# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact epoch accounting and exports for the fixed-batch SFT sweep."""

import bisect
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from levanter.adaptor import AdaptorConfig, NoAdaptorConfig
from levanter.checkpoint import CheckpointRetention
from levanter.data.text.datasets import LmDataConfig
from levanter.tracker import log
from levanter.tracker.tracker import NoopTracker, TrackerConfig
from rigging.filesystem.storage_path import StoragePath, prefix_join

PLATEAU_TOTALS = ("plateau_kept_tokens", "plateau_kept_loss_sum", "plateau_dropped_tokens", "plateau_dropped_loss_sum")
TOTALS = ("loss_sum", "supervised_tokens", "context_tokens", *PLATEAU_TOTALS)


class EpochTracker(NoopTracker):
    """Writes one row per epoch to epoch-metrics.jsonl and fails if an epoch's tokens differ from the manifest.

    ``epoch_updates[e]`` is the number of updates in epoch e + 1. ``supervised_tokens[e]`` and, when given,
    ``context_tokens[e]`` are that epoch's manifest totals; the trained totals must match them within 0.1%.
    """

    name = "epoch_accounting"

    def __init__(
        self,
        output_path: str,
        epoch_updates: Sequence[int],
        supervised_tokens: Sequence[int],
        context_tokens: Sequence[int] | None = None,
        start_step: int = 0,
    ):
        if len(supervised_tokens) != len(epoch_updates) or (
            context_tokens is not None and len(context_tokens) != len(epoch_updates)
        ):
            raise ValueError("Every epoch needs its update count and token totals")
        self.output_path = output_path
        self.epoch_ends = list(itertools.accumulate(epoch_updates))
        if start_step and start_step not in self.epoch_ends:
            raise ValueError("Epoch accounting can resume only at an epoch boundary")
        self.supervised_tokens = list(supervised_tokens)
        self.context_tokens = None if context_tokens is None else list(context_tokens)
        self.next_token_step = start_step
        self.sums = dict.fromkeys(TOTALS, 0.0)
        self.epoch_rows: dict[int, dict[str, Any]] = {}
        self.published_epochs: set[int] = set()

    def log(self, metrics: Mapping[str, Any], *, step: int | None, commit: bool | None = None):
        if step is None:
            return
        boundary = step + 1 in self.epoch_ends
        epoch = bisect.bisect_right(self.epoch_ends, step + 1)
        if "train/supervised_tokens" in metrics:
            if step != self.next_token_step:
                raise ValueError(f"Missing or duplicated training statistics: {step} != {self.next_token_step}")
            self.next_token_step += 1
            for name in TOTALS:
                self.sums[name] += float(metrics.get(f"train/{name}", 0.0))
            if boundary:
                row = self.epoch_rows.setdefault(epoch, {})
                row.update(
                    train_loss=self.sums["loss_sum"] / self.sums["supervised_tokens"],
                    **token_check(
                        "supervised_tokens", self.sums["supervised_tokens"], self.supervised_tokens[epoch - 1]
                    ),
                )
                if self.context_tokens is not None:
                    if "train/context_tokens" not in metrics:
                        raise ValueError("Context-token accounting needs the context_tokens training metric")
                    row.update(
                        token_check("context_tokens", self.sums["context_tokens"], self.context_tokens[epoch - 1])
                    )
                if "train/plateau_kept_tokens" in metrics:
                    row.update(plateau_epoch_row(self.sums, self.sums["supervised_tokens"]))
                self.sums = dict.fromkeys(TOTALS, 0.0)
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


def token_check(name: str, actual: float, expected: int) -> dict[str, int]:
    """The epoch's trained token total next to the manifest's; more than 0.1% apart is an error."""
    difference = actual - expected
    if abs(difference) > 0.001 * expected:
        raise ValueError(f"Epoch {name} differ from the manifest: trained {actual}, manifest {expected}")
    return {
        name: int(actual),
        f"manifest_{name}": expected,
        f"{name.removesuffix('_tokens')}_token_difference": int(difference),
    }


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
    epoch_updates: list[int]
    supervised_tokens: list[int]
    context_tokens: list[int] | None = None
    start_step: int = 0

    def init(self, run_id: str | None) -> EpochTracker:
        return EpochTracker(
            self.output_path, self.epoch_updates, self.supervised_tokens, self.context_tokens, self.start_step
        )


@AdaptorConfig.register_subclass("agentic_sweep_full")
@dataclass(frozen=True)
class SweepExportsConfig(NoAdaptorConfig):
    """A permanent checkpoint at every epoch end; HF exports after the epochs in ``export_epochs`` and at the end.

    ``validation_data`` is read by the offline validation job, not during training.
    """

    validation_data: LmDataConfig | None
    epoch_updates: list[int]
    sequence_length: int
    export_epochs: list[int] = field(default_factory=list)

    def install_export_hooks(self, *, trainer, converter, tokenizer, export) -> None:
        if converter is None or export.hf_save_path is None:
            raise ValueError("The sweep requires an HF converter and export path")
        if trainer.config.microbatch_size != trainer.config.train_batch_size:
            raise ValueError("The sweep forbids separately normalized microbatch accumulation")
        if min(self.epoch_updates) < 4:
            raise ValueError("Sweep epochs must have at least four batches")
        total_steps = trainer.config.num_train_steps
        epoch_ends = list(itertools.accumulate(self.epoch_updates))
        if epoch_ends[-1] != total_steps:
            raise ValueError(f"Epochs hold {epoch_ends[-1]} updates but the run trains {total_steps}")
        exports = {epoch_ends[epoch - 1] for epoch in self.export_epochs} | {total_steps}
        last_exported = -1

        def epoch_hook(info, force=False):
            nonlocal last_exported
            del force
            completed = info.next_step
            if completed + 1 in epoch_ends:
                trainer.request_checkpoint(CheckpointRetention.PERMANENT)
            if completed not in exports or completed == last_exported:
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

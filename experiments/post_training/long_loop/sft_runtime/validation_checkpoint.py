# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain the native parameter checkpoint with the lowest held-out loss."""

import json
import math
import struct
from collections.abc import Callable
from dataclasses import asdict, dataclass

import jax
import levanter.callbacks as callbacks
import levanter.tracker
from jax._src import config as jax_config
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from jaxtyping import PyTree
from levanter.checkpoint import discover_checkpoint_candidates, save_checkpoint
from levanter.eval import TaggedEvaluator, eval_model
from levanter.tensorstore_serialization import TensorStoreWriteConfig
from rigging.filesystem.storage_path import StoragePath

SELECTION_KIND = "snowball_sft_inline_loss_v1"
CHECKPOINT_TIMEOUT = 30 * 60


@dataclass(frozen=True)
class ValidationSelection:
    path: str
    step: int
    score: float
    metric: str


class BestValidationCheckpoint:
    def __init__(self, path: str, metric: str, write_config: TensorStoreWriteConfig):
        self.path = path
        self.metric = metric
        self.write_config = write_config
        # Match the training checkpointer's timeout; the implicit manager's
        # five-minute barrier can expire while one rank is still staging weights.
        self.manager = GlobalAsyncCheckpointManager(timeout_secs=CHECKPOINT_TIMEOUT)
        candidates = discover_checkpoint_candidates(path)
        selections = []
        for candidate in candidates:
            metadata = candidate.metadata
            if metadata.get("selection_kind") != SELECTION_KIND or metadata.get("validation_metric") != metric:
                raise ValueError(f"Unrecognized selection checkpoint: {candidate.path}")
            selections.append(
                ValidationSelection(candidate.path, candidate.step, float(metadata["validation_score"]), metric)
            )
        self.best = min(selections, key=lambda item: (item.score, item.step, item.path)) if selections else None
        self._owned_paths = {selection.path for selection in selections}
        if self.best is not None:
            self._write_selection()

    def _write_selection(self) -> None:
        assert self.best is not None
        if jax.process_index() == 0:
            StoragePath(self.path + "/best.json").write_text(json.dumps(asdict(self.best), indent=2) + "\n")

    def consider(self, model: PyTree, step: int, score: float) -> ValidationSelection | None:
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite checkpoint selection metric {self.metric}: {score}")
        if self.best is not None and score >= self.best.score:
            return None
        destination = f"{self.path}/step-{step}-score-{struct.pack('!d', score).hex()}"
        save_checkpoint(
            {"params": model},
            step,
            destination,
            is_temporary=False,
            manager=self.manager,
            write_config=self.write_config,
            metadata={
                "selection_kind": SELECTION_KIND,
                "validation_metric": self.metric,
                "validation_score": score,
            },
        )
        self.manager.wait_until_finished()
        self.best = ValidationSelection(destination, step, score, self.metric)
        self._write_selection()
        # Only checkpoints created by this selector, under this run's dedicated
        # prefix, are eligible. Commit the replacement before removing any old one.
        if jax.process_index() == 0:
            for previous in sorted(self._owned_paths - {destination}):
                StoragePath(previous).rmtree()
        self._owned_paths = {destination}
        return self.best


def best_validation_callback(
    evaluator: TaggedEvaluator, selection: BestValidationCheckpoint, *, prefix: str
) -> Callable:
    last_eval_step = None

    def evaluate(step: callbacks.StepInfo, force: bool = False) -> None:
        del force
        nonlocal last_eval_step
        if step.step < 0 or step.step == last_eval_step:
            return
        with callbacks.progress_event_scope(
            step.emit_event, callbacks.ProgressEvent.EVALUATION_STARTED, callbacks.ProgressEvent.EVALUATION_FINISHED
        ):
            with jax_config.enable_pgle(False):
                metrics = eval_model(evaluator, step.model, prefix=prefix)
            levanter.tracker.log(metrics, step=step.step)
            with callbacks.progress_event_scope(
                step.emit_event, callbacks.ProgressEvent.CHECKPOINT_STARTED, callbacks.ProgressEvent.CHECKPOINT_FINISHED
            ):
                selected = selection.consider(step.model, step.next_step, float(metrics[selection.metric]))
            if selected is not None:
                levanter.tracker.log_summary({"best_validation": asdict(selected)})
        last_eval_step = step.step

    return evaluate

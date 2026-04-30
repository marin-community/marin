# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V4: grug-moe with tanh(pos_in_segment / 10) loss reweighting.

Same data path as the V1 baseline (concat-and-split, segment_ids derived from
EOS markers). Per-position loss_weight is multiplied by
``tanh(position_within_segment / 10)`` inside the train step, which downweights
loss on the first few tokens of each document.
"""

import dataclasses

from marin.execution.executor import ExecutorStep, executor_main, this_output_path

from experiments.grug.moe.launch import (
    _resolve_run_id,
    baseline_moe,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugTrainerConfig

POSITIONAL_TANH_DIVISOR: float = 10.0


_baseline_grug_trainer = baseline_moe.config.grug_trainer
if isinstance(_baseline_grug_trainer, GrugTrainerConfig):
    _grug_trainer = _baseline_grug_trainer
else:
    _grug_trainer = _baseline_grug_trainer.value  # versioned wrapper


positional_weight_moe = ExecutorStep(
    name="grug/4_10_positional_weight_moe",
    fn=run_grug_moe_trial,
    config=dataclasses.replace(
        baseline_moe.config,
        output_path=this_output_path(),
        run_id=_resolve_run_id("4_10_positional_weight_moe"),
        grug_trainer=dataclasses.replace(
            _grug_trainer,
            positional_loss_tanh_divisor=POSITIONAL_TANH_DIVISOR,
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[positional_weight_moe],
        description="V4 grug MoE: tanh(pos/10) positional loss reweighting.",
    )

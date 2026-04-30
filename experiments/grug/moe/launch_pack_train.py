# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""V2: grug-moe with pack=True on training data (whole-document packing)."""

import dataclasses

from marin.execution.executor import ExecutorStep, executor_main, this_output_path

from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    _pack_training_components,
    _resolve_run_id,
    baseline_moe,
    run_grug_moe_trial,
)

NEMOTRON_MIX_PACK_TRAIN = _pack_training_components(NEMOTRON_MIX_WITH_DEFAULT_VALIDATION)


pack_train_moe = ExecutorStep(
    name="grug/4_10_pack_train_moe",
    fn=run_grug_moe_trial,
    config=dataclasses.replace(
        baseline_moe.config,
        data=NEMOTRON_MIX_PACK_TRAIN,
        output_path=this_output_path(),
        run_id=_resolve_run_id("4_10_pack_train_moe"),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[pack_train_moe],
        description="V2 grug MoE: training data pack=True (whole-doc packing).",
    )

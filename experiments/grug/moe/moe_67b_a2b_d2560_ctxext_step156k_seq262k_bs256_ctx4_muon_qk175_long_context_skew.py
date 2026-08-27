# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-document-skew twin of the qk=1.75 262K context-extension run."""

import dataclasses

from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path

from experiments.grug.moe.launch_datakit_moe_mix import _phase_weights
from experiments.grug.moe.long_context_datakit_moe_mix import (
    LONG_CONTEXT_SKEW,
    long_context_datakit_components,
    long_context_phase_weights,
)
from experiments.grug.moe.moe_67b_a2b_d2560_ctxext_step156k_seq262k_bs256_ctx4_muon import step as base_step

_RUN_ID = f"moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew{LONG_CONTEXT_SKEW:g}"


def _treatment_step() -> ExecutorStep:
    base_config = base_step.config
    original_train_weights = _phase_weights(1)
    validation_components = {
        name: component for name, component in base_config.data.components.items() if name not in original_train_weights
    }
    validation_weights = {name: 0.0 for name in validation_components}
    data = dataclasses.replace(
        base_config.data,
        components={**long_context_datakit_components(LONG_CONTEXT_SKEW), **validation_components},
        train_weights=[(0, {**long_context_phase_weights(), **validation_weights})],
    )
    tracker = dataclasses.replace(
        base_config.tracker,
        tags=[*base_config.tracker.tags, "long_context_store", f"long_context_skew_{LONG_CONTEXT_SKEW:g}"],
    )
    config = dataclasses.replace(
        base_config,
        data=data,
        output_path=this_output_path(),
        run_id=_RUN_ID,
        tracker=tracker,
    )
    return dataclasses.replace(
        base_step,
        name=f"grug/{_RUN_ID}",
        config=config,
        description=(
            "Treatment twin of the qk=1.75 262K context-extension run. "
            f"Uses a {LONG_CONTEXT_SKEW:g}x long-document skew within each phase-1 domain-by-quality bucket."
        ),
    )


step = _treatment_step()


if __name__ == "__main__":
    executor_main(steps=[step], description=step.description)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-document-skew twin of the qk=1.57 262K context-extension run."""

from marin.execution.executor import executor_main

from experiments.grug.moe.long_context_datakit_moe_mix import (
    LONG_CONTEXT_SKEW,
    long_context_treatment_step,
)
from experiments.grug.moe.moe_67b_a2b_d2560_ctxext_step156k_seq262k_bs256_ctx4_muon_qk157 import (
    step as base_step,
)

_RUN_ID = f"moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk157_longctx_skew{LONG_CONTEXT_SKEW:g}"


step = long_context_treatment_step(base_step, run_id=_RUN_ID)


if __name__ == "__main__":
    executor_main(steps=[step], description=step.description)

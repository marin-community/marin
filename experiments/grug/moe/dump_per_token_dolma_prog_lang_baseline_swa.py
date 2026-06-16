# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-token loss dump for paloma's dolma_100_programing_languages slice -- baseline + SWA.

Loads the d=1280 EP=1 seq=4k baseline at step 14325 and applies SWA-only extension
at seq=8k (long_sliding_window=4096) -- matches the model config used in the
`..._swa_d1280_ep1_paloma_bs128_sweep` paloma eval. Dumps per-position
(loss, tokens, weight, segment_ids) for the first 8 batches at bs=128 = same
1024 sequences x 8192 tokens that the paloma TaggedEvaluator scored at bs=128.

Companion dump for the seq8k_train final ckpt is in
``dump_per_token_dolma_prog_lang_seq8k_train``. Both runs use the same
``validation_sets`` lookup (deterministic), so their .npz outputs are alignable
batch-by-batch and position-by-position. Offline analysis can compute
per-token delta = loss_seq8k_train - loss_baseline_swa to see exactly where
the ~0.012 bpb gap on this slice lives.

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.dump_per_token_dolma_prog_lang_baseline_swa
"""

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.dump_per_token_paloma_slice import PerTokenDumpConfig, run_per_token_dump
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION

_DIM: int = 1280
_CKPT: str = "gs://marin-us-central2/grug/moe_may_compute_opt_d1280_ep1-b9a7ad/checkpoints/step-14325"
_SLICE: str = "paloma/dolma_100_programing_languages"
_SEQ: int = 8192
_BS: int = 128
_MAX_BATCHES: int = 8  # 8 x 128 x 8192 = 8.39M tokens, matches paloma_bs128 sweep
_EP: int = 1
_LONG_SWA: int = 4096

_run_id = f"moe_may_dump_per_token_dolma_prog_lang_d{_DIM}_baseline_swa"
dump_step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_per_token_dump,
    config=PerTokenDumpConfig(
        dim=versioned(_DIM),
        run_id=_run_id,
        output_path=this_output_path(),
        resources=versioned(ResourceConfig.with_tpu("v4-32")),
        checkpoint_path=versioned(_CKPT),
        slice_name=versioned(_SLICE),
        seq_len=versioned(_SEQ),
        batch_size=versioned(_BS),
        max_batches=versioned(_MAX_BATCHES),
        expert_parallel=versioned(_EP),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        long_sliding_window=versioned(_LONG_SWA),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[dump_step],
        description=(
            f"Dump per-token loss for {_SLICE} at seq={_SEQ} bs={_BS} max={_MAX_BATCHES}. "
            f"d={_DIM} EP=1 baseline ckpt (step 14325) + SWA long_sliding_window={_LONG_SWA}. "
            f"v4-32 us-central2."
        ),
    )

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment R2a: v5p-8 explicit CE block-size probe.

Follow-up to Exp Q and Exp R. Exp Q established that on v5p-8 the same
good LoRA recipe stays stuck near ln(2) at both pd=8 and pd=4. Exp R
lowered `train_batch_size` to 32 to try to match the good v5p-16 pd=2
CE kernel shape, but (a) the resulting DEBUGCE line was
`x.shape=(16384, 4096), b_block_size=1024, num_b_blocks=16` — not a
match to the good run's `(32768, 4096), b_block_size=32768,
num_b_blocks=1` — and (b) the run still stayed stuck.

Exp R's "reduce bs" lever changes three things at once on v5p-8:
  1. per-chip token count B
  2. grad_accum
  3. the heuristic's choice of b_block_size / num_b_blocks

R2a isolates the third variable. We hold bs=64, pd=4, grad_accum, and
microbatch exactly identical to the Exp Q bad baseline, and change ONLY
the fused CE kernel's block tiling by forcing explicit block sizes via
the MARIN_DEBUG_CE_{B,V}_BLOCK_SIZE environment variables.

Case A (this script): b_block_size=65536, v_block_size=8192, so
num_b_blocks=1, num_v_blocks=16. This eliminates the 64-way bf16
inter-block accumulation of `gw_block` / `gx_block` that the heuristic
path currently performs on v5p-8 pd=4 bs=64. It does NOT match the good
v5p-16 pd=2 run's per-tile shape (tile is 2x larger in the batch dim),
but it does match the good run on "no inter-batch-block accumulation"
and on the vocab-block structure.

Interpretation of possible outcomes:
  - If the run escapes ln(2) like the good v5p-16 pd=2 run, then CE
    backward bf16 accumulation across batch blocks on v5p-8 is the
    load-bearing cause. That is a focused, actionable kernel-level bug.
  - If the run still stays near ln(2) with `explicit_block_sizes=True`
    confirmed in the DEBUGCE line, then CE tiling is definitively
    ruled out as the load-bearing cause, and the investigation pivots
    to sub-CE suspects (FSDP sharding / attention kv-head mapping /
    reference-network graph), which is what Exp T is already probing.

HBM analysis:
  Concurrent CE backward temporaries scale with b_block_size. At the
  current heuristic pick (b_block=1024) peak CE temps per chip are
  ~100 MiB. At b_block=65536, peak temps rise to ~2.1 GiB (x_block +
  delta + gx_inner tiles). v5p chips have 95 GiB HBM and current
  training uses roughly 20-30 GiB/chip, so this ~2 GiB bump is safe.
  A 1-step compile-only probe is still worth doing first to confirm.

Required kernel-side change:
  `xla.py:linear_softmax_cross_entropy_loss_xla` reads
  `MARIN_DEBUG_CE_B_BLOCK_SIZE` and `MARIN_DEBUG_CE_V_BLOCK_SIZE` at
  kernel-call time and constructs a BlockSizes override when both are
  set. Verify the DEBUGCE line shows `explicit_block_sizes=True`
  before trusting any downstream result.

Override knobs via env:
  - EXPERIMENT_R2A_BS              (default 64 — matches Exp Q baseline)
  - EXPERIMENT_R2A_PD              (default 4  — matches Exp Q baseline)
  - EXPERIMENT_R2A_CE_B_BLOCK_SIZE (default 65536 — covers full B=65536 in 1 batch block)
  - EXPERIMENT_R2A_CE_V_BLOCK_SIZE (default 8192  — matches heuristic/good run)
  - REGIONS_OVERRIDE
  - MARIN_DEBUG_RUN_TAG

Follow-ups (if R2a stays stuck):
  - R2b: b_block_size=32768, num_b_blocks=2 — matches per-tile compute
    shape to the good run, accepts 2-way inter-block accumulation.
  - Pivot to attention kv-head / FSDP / reference-graph probes.

Note
----

New runs in this campaign should use the same-region GCS copy of
``Llama-3.1-8B-Instruct`` as the base model (see
``experiments.posttrain.per_stmt_dpo.base_model.LLAMA_3_1_8B_INSTRUCT_GCS_PATH``).
Loading is ~3-5x faster than the HuggingFace Hub CDN and Bug-1 has been
verified model-agnostic. Do NOT retroactively edit the ``model_name_or_path``
here - its historical value is tied to the run artifacts. This note is
for future forks of this experiment.
"""

import os

from levanter.adaptation import LoraAdaptationConfig
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.main.train_dpo import AdapterBaseReferenceConfig

from experiments.defaults import default_dpo
from experiments.llama import LLAMA3_CHAT_STOP_TOKEN_IDS, llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.per_stmt_dpo.common import (
    tokenized_full_val,
    tokenized_preferences,
    tokenized_stmt_val,
    tokenized_train,
)
from experiments.simple_dpo_config import SimpleDPOConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

TPU_TYPE = "v5p-8"
DEFAULT_REGIONS = ["us-central1", "us-east5"]
DEFAULT_BS = 64
DEFAULT_PD = 4
DEFAULT_CE_B_BLOCK_SIZE = 65536
DEFAULT_CE_V_BLOCK_SIZE = 8192

_region_override = os.environ.get("REGIONS_OVERRIDE", "")
REGIONS_FOR_TPU = [r.strip() for r in _region_override.split(",") if r.strip()] or DEFAULT_REGIONS

TRAIN_BATCH_SIZE = int(os.environ.get("EXPERIMENT_R2A_BS", str(DEFAULT_BS)))
PER_DEVICE = int(os.environ.get("EXPERIMENT_R2A_PD", str(DEFAULT_PD)))
CE_B_BLOCK_SIZE = int(os.environ.get("EXPERIMENT_R2A_CE_B_BLOCK_SIZE", str(DEFAULT_CE_B_BLOCK_SIZE)))
CE_V_BLOCK_SIZE = int(os.environ.get("EXPERIMENT_R2A_CE_V_BLOCK_SIZE", str(DEFAULT_CE_V_BLOCK_SIZE)))

_DBG_RUN_TAG = os.environ.get(
    "MARIN_DEBUG_RUN_TAG",
    f"bs{TRAIN_BATCH_SIZE}pd{PER_DEVICE}bb{CE_B_BLOCK_SIZE}vb{CE_V_BLOCK_SIZE}",
)

config = SimpleDPOConfig(
    resources=ResourceConfig.with_tpu(
        TPU_TYPE,
        ram="250g" if PER_DEVICE <= 4 else "400g",
        regions=REGIONS_FOR_TPU,
    ),
    per_device_parallelism=PER_DEVICE,
    per_device_eval_parallelism=PER_DEVICE,
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=10,
    steps_per_eval=10,
    learning_rate=1e-6,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",
    adapter=LoraAdaptationConfig(r=64, alpha=64, dropout=0.0, zero_init_b=True, target_modules=None),
    reference=AdapterBaseReferenceConfig(),
    train_seq_len=4096,
    max_seq_len=4096,
    beta=0.1,
    validation_split_fraction=None,
    reference_eval_cache=ReferenceEvalCacheConfig(mode="disabled"),
    steps_per_checkpoint=9999,
    steps_per_hf_export=9999,
    hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
    seed=0,
    max_eval_batches=1,
    env_vars={
        "MARIN_DEBUG_LOG_BATCH_INDICES": "1",
        "MARIN_DEBUG_LOG_STEP_TRACE": "1",
        "MARIN_DEBUG_CE_B_BLOCK_SIZE": str(CE_B_BLOCK_SIZE),
        "MARIN_DEBUG_CE_V_BLOCK_SIZE": str(CE_V_BLOCK_SIZE),
    },
)

training_step = default_dpo(
    name=(
        f"dpo/stmt_dpo/debug/experiment_r2a_r64_v5p8_bs{TRAIN_BATCH_SIZE}"
        f"_pd{PER_DEVICE}_b{CE_B_BLOCK_SIZE}_v{CE_V_BLOCK_SIZE}_s10_{_DBG_RUN_TAG}"
    ),
    tokenized=tokenized_preferences,
    model_config=llama_8b,
    dpo_config=config,
    tags=[
        "dpo",
        "lora-dpo",
        "bloom",
        "per-stmt",
        "support-mental-health",
        "debug-accum",
        "experiment-r2a",
        "explicit-ce-block-sizes",
        "v5p-8",
        f"bs{TRAIN_BATCH_SIZE}",
        f"pd{PER_DEVICE}",
        f"bblock{CE_B_BLOCK_SIZE}",
        f"vblock{CE_V_BLOCK_SIZE}",
        "r64-alpha64",
    ],
    include_lm_validation=False,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_stmt_val, tokenized_full_val, training_step])

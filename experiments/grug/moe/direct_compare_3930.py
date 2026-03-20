# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct comparison: es3r2 vs Qwen A3B-like baselines.

Encodes the three comparison targets from issue #3930:
- es3r2 incumbent (promoted from #3528)
- q30a3b-proxy (earlier Qwen-style Grug proxy from #3357 bring-up)
- q35a3b-fa (new Qwen3.5-35B-A3B-inspired full-attention variant)

All configs share vocab_size=128_256 (Llama3-ish tokenizer) and max_seq_len=4096.
"""

import os

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe,
)
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

VOCAB_SIZE = 128_256
MAX_SEQ_LEN = 4096

# --- Model configs ---

# es3r2: promoted primary target from #3528.
# h4096/l27/e64/ep4/topk4/ix1024/sx1024/cf1.25, GQA 32 heads / 4 kv heads.
ES3R2_MODEL = GrugModelConfig(
    vocab_size=VOCAB_SIZE,
    hidden_dim=4096,
    intermediate_dim=1024,
    shared_expert_intermediate_dim=1024,
    num_experts=64,
    num_experts_per_token=4,
    num_layers=27,
    num_heads=32,
    num_kv_heads=4,
    head_dim=128,
    max_seq_len=MAX_SEQ_LEN,
)

# q30a3b-proxy: earlier Qwen-style Grug proxy from #3357 32B-A4B bring-up.
# h2048/l48/e128/topk8/ix768/sx2048, GQA 32 heads / 4 kv heads.
Q30A3B_PROXY_MODEL = GrugModelConfig(
    vocab_size=VOCAB_SIZE,
    hidden_dim=2048,
    intermediate_dim=768,
    shared_expert_intermediate_dim=2048,
    num_experts=128,
    num_experts_per_token=8,
    num_layers=48,
    num_heads=32,
    num_kv_heads=4,
    head_dim=128,
    max_seq_len=MAX_SEQ_LEN,
)

# q35a3b-fa: Qwen3.5-35B-A3B-inspired full-attention variant.
# h2048/l40/e256/topk8/ix512/sx512, GQA 16 heads / 2 kv heads, head_dim=256.
# Forced full attention (no linear attention), Llama3 tokenizer, same harness.
Q35A3B_FA_MODEL = GrugModelConfig(
    vocab_size=VOCAB_SIZE,
    hidden_dim=2048,
    intermediate_dim=512,
    shared_expert_intermediate_dim=512,
    num_experts=256,
    num_experts_per_token=8,
    num_layers=40,
    num_heads=16,
    num_kv_heads=2,
    head_dim=256,
    max_seq_len=MAX_SEQ_LEN,
)

# --- Shared launch defaults ---

WANDB_GROUP = "grug-moe-direct-compare-3930"
DEFAULT_RESOURCES = ResourceConfig.with_tpu("v5p-64")
DEFAULT_STEPS = 2_000
DEFAULT_BATCH_SIZE = 320
DEFAULT_SEED = 0
DEFAULT_MP = "params=float32,compute=bfloat16,output=bfloat16"

DEFAULT_OPTIMIZER = AdamConfig(
    learning_rate=3e-3,
    weight_decay=0.1,
    lr_schedule="cosine",
    decay=0.2,
    min_lr_ratio=0.1,
    warmup=1000,
)

DEFAULT_GRUG_TRAINER = GrugTrainerConfig(
    z_loss_weight=1e-4,
    ema_beta=None,
    log_every=1,
)

DEFAULT_EVAL = GrugEvalConfig(
    eval_batch_size=320,
    steps_per_eval=500,
    max_eval_batches=8,
    eval_current=True,
    eval_ema=False,
)


def _resolve_run_id(suffix: str) -> str:
    base = os.environ.get("GRUG_RUN_ID", f"direct-compare-{suffix}")
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        base = f"{base}-{ferry_date}"
    return base


def _make_step(
    name_suffix: str,
    model: GrugModelConfig,
    run_id: str,
    tags: list[str],
) -> ExecutorStep:
    return ExecutorStep(
        name=f"grug/direct-compare-{name_suffix}",
        fn=run_grug_moe,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(DEFAULT_RESOURCES),
            steps=versioned(DEFAULT_STEPS),
            batch_size=versioned(DEFAULT_BATCH_SIZE),
            seed=versioned(DEFAULT_SEED),
            mp=versioned(DEFAULT_MP),
            tracker=WandbConfig(
                project="marin",
                tags=["grug", "moe", "direct-compare", *tags],
                group=WANDB_GROUP,
                name=None,
            ),
            optimizer=versioned(DEFAULT_OPTIMIZER),
            grug_trainer=versioned(DEFAULT_GRUG_TRAINER),
            eval=versioned(DEFAULT_EVAL),
            profiler=ProfilerConfig(enabled=True, start_step=50, num_steps=5),
        ),
    )


es3r2_step = _make_step(
    "es3r2",
    ES3R2_MODEL,
    _resolve_run_id("es3r2"),
    ["es3r2", "incumbent"],
)

q30a3b_proxy_step = _make_step(
    "q30a3b-proxy",
    Q30A3B_PROXY_MODEL,
    _resolve_run_id("q30a3b-proxy"),
    ["q30a3b-proxy", "qwen-baseline"],
)

q35a3b_fa_step = _make_step(
    "q35a3b-fa",
    Q35A3B_FA_MODEL,
    _resolve_run_id("q35a3b-fa"),
    ["q35a3b-fa", "qwen35-inspired"],
)

ALL_STEPS = [es3r2_step, q30a3b_proxy_step, q35a3b_fa_step]

if __name__ == "__main__":
    executor_main(
        steps=ALL_STEPS,
        description="Direct comparison: es3r2 vs Qwen A3B-like baselines (#3930).",
    )

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fine-tune the 262K Snowball skew8 base on one 8xH100 node.

The Hugging Face model is the BF16 export of ``BASE_CHECKPOINT`` at
``HF_REVISION``. Training reads the native checkpoint so each process restores
only its parameter shards rather than materializing the 134 GB HF export on one
host.
"""

import argparse
import json
import logging
import math
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.data.text.datasets import LmDataConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.training.training import temporary_checkpoint_base_path
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.train import (
    GrugRunConfig,
    GrugTrainerConfig,
    WeightInitialization,
    grug_trainer_mesh_config,
    run_grug,
)
from experiments.grug_sft.special_token_lr import (
    ANCHORS,
    EXPECTED_ANCHOR_IDS,
    TOKENIZER,
    mixture_data_config,
)
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig

logger = logging.getLogger(__name__)

HF_MODEL = "open-athena/snowball-67b-a2b-base-262k-qk175-skew8"
HF_REVISION = "058ecaf27b9e4f37219df221a51e7d490d58ec3d"
BASE_CHECKPOINT = (
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew8-c06695/"
    "checkpoints/step-157000"
)
DEFAULT_STORES_MANIFEST = "gs://marin-us-central2/users/held/grug_sft/grug-67b-sft-20260919-48h-data/stores.json"
DEFAULT_RUN_ID = "grug-67b-sft-262k-h100x8"
WANDB_PROJECT = "marin_moe_sft"

CONTEXT_LENGTH = 262_144
CONTEXT_SHARDS = 8
BATCH_SIZE = 1
DEFAULT_STEPS = 1_000
BASE_STEP = 157_000
TENSORSTORE_CACHE_BYTES = 2 * 1024**3


def model_config() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=2_560,
        intermediate_dim=1_280,
        shared_expert_intermediate_dim=2_560,
        num_shared_experts=1,
        num_experts=256,
        num_experts_per_token=4,
        num_layers=26,
        num_heads=20,
        num_kv_heads=5,
        head_dim=128,
        max_seq_len=CONTEXT_LENGTH,
        sliding_window=2_048,
        global_every=4,
        layer_norm_eps=1e-5,
        initializer_std=0.009882117688026186,
        qk_mult=1.75,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="sonic",
        remat_mode="recompute_all",
    )


def optimizer_config(steps: int) -> GrugMoeMuonHConfig:
    return GrugMoeMuonHConfig(
        learning_rate=5e-5,
        adam_lr=5e-5,
        beta1=0.9062,
        beta2=0.95,
        epsilon=3.8339433005718795e-15,
        weight_decay=0.0,
        max_grad_norm=None,
        min_lr_ratio=0.1,
        warmup=0,
        decay=max(1, steps // 10),
        lr_schedule="linear",
        rmsnorm_to_adam=True,
    )


def run_config(
    run_id: str,
    steps: int,
    data: LmDataConfig,
    token_ids: tuple[int, ...],
    anchor_ids: tuple[tuple[int, ...], ...],
) -> GrugRunConfig:
    if not run_id.strip():
        raise ValueError("Run ID must not be empty")
    if steps <= 0:
        raise ValueError("Steps must be positive")

    output = f"gs://marin-us-central2/users/held/grug_sft/{run_id}"
    permanent_checkpoints = str(StoragePath(output) / "checkpoints")
    temporary_checkpoints = temporary_checkpoint_base_path(output)
    trainer = TrainerConfig(
        id=run_id,
        seed=0,
        train_batch_size=BATCH_SIZE,
        num_train_steps=steps,
        profiler=ProfilerConfig(enabled=False),
        mp=jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            entity="marin-community",
            project=WANDB_PROJECT,
            name=run_id,
            id=run_id,
            resume="allow",
            tags=["sft", "snowball", "262k", "h100x8", f"hf-{HF_REVISION[:12]}"],
        ),
        watch=WatchConfig(interval=0),
        progress_watchdog=ProgressWatchdogConfig(
            startup_timeout=timedelta(hours=2),
            step_timeout=timedelta(hours=1),
            process_timeout=timedelta(hours=2),
        ),
        use_explicit_mesh_axes=True,
        mesh=grug_trainer_mesh_config(CONTEXT_SHARDS),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        initialize_from=BASE_CHECKPOINT,
        load_checkpoint_path=[permanent_checkpoints, temporary_checkpoints],
        checkpointer=CheckpointerConfig(
            base_path=permanent_checkpoints,
            temporary_base_path=temporary_checkpoints,
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=30),
            keep=None,
            delete_old_temp_checkpoints=True,
            keep_last_temporary_checkpoints=1,
        ),
    )
    return GrugRunConfig(
        model=model_config(),
        data=data,
        resources=ResourceConfig.with_gpu(
            "H100",
            count=8,
            cpu=64,
            ram="768g",
            disk="256g",
            preemptible=False,
        ),
        tensorstore_cache_bytes=TENSORSTORE_CACHE_BYTES,
        optimizer=optimizer_config(steps),
        trainer=GrugTrainerConfig(
            trainer=trainer,
            log_every=1,
            ema_beta=None,
            z_loss_weight=1e-4,
            offload_opt_state=True,
            save_checkpoints=True,
            expert_axis_size=1,
            replica_axis_size=1,
            context_axis_size=CONTEXT_SHARDS,
            weight_initialization=WeightInitialization.LEGACY_SINGLE_SHARED_EXPERT,
            reinitialize_token_ids=token_ids,
            reinitialize_token_anchors=anchor_ids,
            special_token_lr_ids=token_ids,
            special_token_lr_multiplier=math.sqrt(32),
        ),
        eval=None,
        processes_per_task=8,
        max_retries_failure=3,
        max_task_failures=3,
    )


def train(run_id: str, steps: int, stores_manifest: str) -> None:
    metadata = json.loads((StoragePath(BASE_CHECKPOINT) / "metadata.json").read_text())
    if metadata["step"] != BASE_STEP:
        raise ValueError(f"Base checkpoint is step {metadata['step']}, expected {BASE_STEP}")

    tokenizer = load_tokenizer(TOKENIZER)
    token_ids = tuple(token_id for token_id, _ in ANCHORS)
    anchor_ids = tuple(tuple(tokenizer.encode(text, add_special_tokens=False)) for _, text in ANCHORS)
    if anchor_ids != EXPECTED_ANCHOR_IDS:
        raise ValueError(f"Semantic token anchors changed: {anchor_ids}")

    run_grug(run_config(run_id, steps, mixture_data_config(stores_manifest), token_ids, anchor_ids))


if __name__ == "__main__":
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--stores-manifest", default=DEFAULT_STORES_MANIFEST)
    args = parser.parse_args()
    train(args.run_id, args.steps, args.stores_manifest)

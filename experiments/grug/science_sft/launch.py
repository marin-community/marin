# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train one packed epoch on a converted science-forward Harmony token store."""

import argparse
import dataclasses
import json
import math
import re
from datetime import timedelta

import jmp
from fray.types import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.mixture import StopStrategy
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.datakit.sft import SftTokenStore, sft_data_config
from marin.training.training import temporary_checkpoint_base_path
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe.optimizer import GrugMoeAdamHConfig
from experiments.grug.science_sft.prepare import CONTEXT, MODEL_REVISION, SOURCE_NAME, TOKENIZER
from experiments.grug.science_sft.train import GrugRunConfig, GrugTrainerConfig, RouterBiasUpdate, RouterFreeze, run_grug
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig

MODEL_PATH = "s3://marin-us-east-02a/models/open-athena--Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38"
OUTPUT_ROOT = "s3://marin-us-east-02a/marin/users/benfeuer/grug-science-converted-sft-runs"
BATCH = 64
NODES = 8
EXPERT_PARALLEL = 8
MIXTURE_BLOCK_SIZE = BATCH
CONTROL_IDS = (128006, 128007, 128002, 128003, 128005, 128011, 128009)
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclasses.dataclass(frozen=True)
class ScienceSftRecipe:
    """Pinned model identity and run settings for a converted science SFT arm."""

    tokenizer: str
    model_path: str
    output_root: str
    identity_prefix: str
    epochs: int
    qk_mult: float
    tags: tuple[str, ...]


STEP38_RECIPE = ScienceSftRecipe(
    tokenizer=TOKENIZER,
    model_path=MODEL_PATH,
    output_root=OUTPUT_ROOT,
    identity_prefix="science-forward-converted-step38",
    epochs=1,
    qk_mult=1.5703274004183787,
    tags=("science-forward", "minimax-converted", "step38", "router-bias:per-step-qb"),
)


def build_run_config(store: SftTokenStore, version: str, recipe: ScienceSftRecipe = STEP38_RECIPE) -> GrugRunConfig:
    """Build a packed SFT run with an explicit pass count and per-step QB router updates."""
    if not _VERSION_RE.fullmatch(version):
        raise ValueError("Version must contain only letters, digits, '.', '_', and '-'")
    if recipe.epochs < 1:
        raise ValueError("SFT run must specify at least one data epoch")
    if store.max_length != CONTEXT or store.tokenizer != recipe.tokenizer:
        raise ValueError("SFT store context or tokenizer does not match the pinned Snowball model")
    if set(store.sources) != {SOURCE_NAME}:
        raise ValueError("SFT store does not contain exactly the audited converted science source")
    if any(count.overlength_conversations for count in store.sources.values()):
        raise ValueError("SFT store excluded overlength conversations")
    steps = recipe.epochs * store.packed_sequences // BATCH
    if steps < 1:
        raise ValueError("SFT store has no full training batch")
    identity = f"{recipe.identity_prefix}-{version}"
    output = prefix_join(recipe.output_root, identity)
    data = dataclasses.replace(
        sft_data_config({"converted": store}, minimum_weight=1.0),
        stop_strategy=(StopStrategy.FIRST_STOP_STRATEGY if recipe.epochs == 1 else StopStrategy.RESTART_STRATEGY),
        mixture_block_size=MIXTURE_BLOCK_SIZE,
    )
    model = GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=2560,
        intermediate_dim=1280,
        shared_expert_intermediate_dim=2560,
        num_experts=256,
        num_experts_per_token=4,
        num_layers=26,
        num_heads=20,
        num_kv_heads=5,
        initializer_std=0.5 / math.sqrt(2560),
        disable_pko=True,
        disable_long_rope=True,
        sliding_window=2048,
        use_array_stacked_blocks=True,
        qk_mult=recipe.qk_mult,
        max_seq_len=CONTEXT,
        attention_implementation="gpu_fa4_cute",
        ce_implementation="batched_xla",
    )
    trainer = TrainerConfig(
        id=identity,
        seed=0,
        train_batch_size=BATCH,
        per_device_parallelism=1,
        num_train_steps=steps,
        mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            entity="nyu-dice-lab",
            project="snowball-sft",
            name=identity,
            id=identity,
            mode="online",
            resume="allow",
            save_code=False,
            tags=list(recipe.tags),
        ),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": EXPERT_PARALLEL}, compute_mapping={"batch": ["data", "expert"]}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        initialize_from=None,
        load_checkpoint=None,
        checkpointer=CheckpointerConfig(
            base_path=prefix_join(output, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(output),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=30),
            keep=None,
            keep_last_temporary_checkpoints=1,
        ),
    )
    optimizer = GrugMoeAdamHConfig(
        learning_rate=5e-6,
        adam_lr=5e-6,
        beta1=0.9062,
        beta2=0.95,
        epsilon=3.8339433005718795e-15,
        weight_decay=0.0,
        max_grad_norm=None,
        min_lr_ratio=0.1,
        warmup=0.05,
        decay=max(1, steps // 10),
        lr_schedule="linear",
    )
    return GrugRunConfig(
        model=model,
        data=data,
        optimizer=optimizer,
        resources=ResourceConfig.with_gpu(
            "H100", count=8, cpu=32, ram="512g", disk="256g", replicas=NODES, preemptible=False
        ),
        trainer=GrugTrainerConfig(
            trainer=trainer,
            initialize_from_hf=recipe.model_path,
            max_data_epochs=recipe.epochs,
            special_token_lr_ids=CONTROL_IDS,
            special_token_lr_multiplier=math.sqrt(32),
            router_freeze=RouterFreeze.BIAS,
            router_bias_update=RouterBiasUpdate.PER_STEP,
            z_loss_weight=1e-4,
            ema_beta=None,
            log_every=1,
            replica_axis_size=1,
            expert_axis_size=EXPERT_PARALLEL,
        ),
        eval=None,
    )


def launch(store_path: str, version: str) -> None:
    configure_coreweave_s3()
    model_manifest = json.loads(StoragePath(prefix_join(MODEL_PATH, "source-revision.json")).read_text())
    if model_manifest["revision"] != MODEL_REVISION or model_manifest["weight_files"] != 39:
        raise ValueError("Step38 model mirror revision or weight count changed")
    store = SftTokenStore.raw_load(store_path)
    run_grug(build_run_config(store, version))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-path", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    launch(args.store_path, args.version)


if __name__ == "__main__":
    main()

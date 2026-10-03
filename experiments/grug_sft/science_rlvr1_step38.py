# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train three science mixes with GLM 5.3 RLVR1 replay from the Step38 Snowball export."""

import argparse
import json
import math
from datetime import timedelta

import jax
import jmp
import numpy as np
from fray.types import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import AsyncDataset
from levanter.data.mixture import StopStrategy
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.store.cache import CacheLedger, TreeCache
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.training.training import temporary_checkpoint_base_path
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug_sft.head_only_train import (
    GrugRunConfig,
    GrugTrainerConfig,
    RouterBiasUpdate,
    RouterFreeze,
    run_grug,
)
from experiments.grug_sft.science_mix import ScienceMix
from experiments.grug_sft.science_step38_model import CONTEXT, MODEL_PATH, MODEL_REVISION, science_model_config
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeAdamHConfig

BASE_MIX_ROOT = "s3://marin-us-east-02a/marin/users/benfeuer/grug-science-100b-mixes/2026.09.21-v1"
RLVR_CACHE = "s3://marin-us-east-02a/marin/users/benfeuer/targeted-sft/glm53-rlvr1-32k-2026.09.25-v1"
COMBINED_ROOT = "s3://marin-us-east-02a/marin/users/benfeuer/grug-science-rlvr1-mixes/2026.09.25-v1"
OUTPUT_ROOT = "s3://marin-us-east-02a/marin/users/benfeuer/grug-science-rlvr1-runs"
BASE_CONTEXT = 262_144
BATCH = 64
EXPERT_PARALLEL = 8
NODES = 8
BLOCK_SIZE = 32_768
CONTROL_IDS = (128006, 128007, 128002, 128003, 128005, 128011, 128009)


class WindowedPrebuiltDataset(AsyncDataset[GrugLmExample]):
    """Read fixed packed rows and expose smaller consecutive training windows."""

    def __init__(self, path: str, source_context: int):
        if source_context % CONTEXT:
            raise ValueError(f"Source context {source_context} is not divisible by {CONTEXT}")
        self.path = path
        self.source_context = source_context
        self.windows_per_row = source_context // CONTEXT
        self._cache = None
        self._rows = 0

    def _open(self) -> TreeCache:
        if self._cache is None:
            configure_coreweave_s3()
            ledger = CacheLedger.load(self.path)
            if not ledger.is_finished:
                raise ValueError(f"Unfinished cache: {self.path}")
            exemplar = {
                "input_ids": np.zeros(0, dtype=np.int32),
                "loss_weight": np.zeros(0, dtype=np.float32),
                "segment_ids": np.zeros(0, dtype=np.int32),
            }
            self._cache = TreeCache.load(self.path, exemplar=exemplar, options=ledger.metadata)
            self._rows = ledger.total_num_rows
        return self._cache

    async def async_len(self) -> int:
        self._open()
        return self._rows * self.windows_per_row

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices):
        cache = self._open()
        parents = list(dict.fromkeys(index // self.windows_per_row for index in indices))
        rows = await cache.get_batch(parents)
        by_parent = dict(zip(parents, rows, strict=True))
        examples = []
        for index in indices:
            row = by_parent[index // self.windows_per_row]
            offset = (index % self.windows_per_row) * CONTEXT
            stop = offset + CONTEXT
            if any(len(row[field]) != self.source_context for field in ("input_ids", "loss_weight", "segment_ids")):
                raise ValueError(f"Invalid packed row length in {self.path}")
            examples.append(
                GrugLmExample.causal(
                    jax.numpy.asarray(row["input_ids"][offset:stop]),
                    loss_weight=jax.numpy.asarray(row["loss_weight"][offset:stop]),
                    segment_ids=jax.numpy.asarray(row["segment_ids"][offset:stop]),
                    block_cross_document_attention=True,
                )
            )
        return examples


def validated_sources(mix: ScienceMix) -> tuple[int, int]:
    model_manifest = json.loads(StoragePath(prefix_join(MODEL_PATH, "source-revision.json")).read_text())
    if model_manifest["revision"] != MODEL_REVISION or model_manifest["weight_files"] != 39:
        raise ValueError("Step38 model mirror revision or weight count changed")
    manifest = json.loads(StoragePath(prefix_join(prefix_join(COMBINED_ROOT, mix.value), "manifest.json")).read_text())
    base = CacheLedger.load(prefix_join(BASE_MIX_ROOT, mix.value))
    rlvr = CacheLedger.load(RLVR_CACHE)
    if manifest["base_cache"] != prefix_join(BASE_MIX_ROOT, mix.value) or manifest["rlvr_cache"] != RLVR_CACHE:
        raise ValueError(f"Combined mix paths changed: {mix}")
    if not base.is_finished or base.total_num_rows != manifest["base_sequences_262k"]:
        raise ValueError(f"Base science cache is incomplete: {mix}")
    if not rlvr.is_finished or rlvr.total_num_rows != manifest["rlvr_sequences_32k"]:
        raise ValueError("RLVR1 cache is incomplete")
    return base.total_num_rows * (BASE_CONTEXT // CONTEXT), rlvr.total_num_rows


def run(
    mix: ScienceMix,
    version: str,
    *,
    learning_rate_multiplier: float = 1.0,
    train_router_bias_residual: bool = False,
    router_bias_update: RouterBiasUpdate = RouterBiasUpdate.FIXED,
) -> None:
    if learning_rate_multiplier <= 0:
        raise ValueError("learning_rate_multiplier must be positive")
    if train_router_bias_residual and router_bias_update == RouterBiasUpdate.PER_STEP:
        raise ValueError("Per-step QB updates cannot be combined with a learned bias residual")
    configure_coreweave_s3()
    base_rows, rlvr_rows = validated_sources(mix)
    total_rows = base_rows + rlvr_rows
    steps = total_rows // BATCH
    if steps < 1:
        raise ValueError("Combined science mix has no full batch")
    identity = f"science-rlvr1-step38-{mix}-{version}"
    output = prefix_join(OUTPUT_ROOT, identity)
    data = LmDataConfig(
        tokenizer=MODEL_PATH,
        components={
            "science": DirectDatasetComponent(
                datasets={"train": WindowedPrebuiltDataset(prefix_join(BASE_MIX_ROOT, mix.value), BASE_CONTEXT)}
            ),
            "rlvr1": DirectDatasetComponent(datasets={"train": WindowedPrebuiltDataset(RLVR_CACHE, CONTEXT)}),
        },
        train_weights={"science": base_rows / total_rows, "rlvr1": rlvr_rows / total_rows},
        auto_build_caches=False,
        shuffle=True,
        block_cross_document_attention=True,
        mixture_block_size=BLOCK_SIZE,
        stop_strategy=StopStrategy.RESTART_STRATEGY,
    )
    model = science_model_config(trainable_router_bias=train_router_bias_residual)
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
            tags=[
                "science",
                "rlvr1",
                f"mix:{mix}",
                "step38",
                f"lr-multiplier:{learning_rate_multiplier:g}",
                f"train-router-bias-residual:{train_router_bias_residual}",
                f"router-bias-update:{router_bias_update}",
            ],
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
        learning_rate=5e-6 * learning_rate_multiplier,
        adam_lr=5e-6 * learning_rate_multiplier,
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
    run_grug(
        GrugRunConfig(
            model=model,
            data=data,
            optimizer=optimizer,
            resources=ResourceConfig.with_gpu(
                "H100", count=8, cpu=32, ram="512g", disk="256g", replicas=NODES, preemptible=False
            ),
            trainer=GrugTrainerConfig(
                trainer=trainer,
                initialize_from_hf=MODEL_PATH,
                data_start_step=0,
                special_token_lr_ids=CONTROL_IDS,
                special_token_lr_multiplier=math.sqrt(32),
                reinitialize_token_ids=(),
                reinitialize_token_anchors=(),
                router_freeze=RouterFreeze.NONE if train_router_bias_residual else RouterFreeze.BIAS,
                router_bias_update=router_bias_update,
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
                replica_axis_size=1,
                expert_axis_size=EXPERT_PARALLEL,
            ),
            eval=None,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", choices=list(ScienceMix), type=ScienceMix, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--learning-rate-multiplier", type=float, default=1.0)
    parser.add_argument("--train-router-bias-residual", action="store_true")
    parser.add_argument(
        "--router-bias-update", choices=list(RouterBiasUpdate), type=RouterBiasUpdate, default=RouterBiasUpdate.FIXED
    )
    args = parser.parse_args()
    run(
        args.mix,
        args.version,
        learning_rate_multiplier=args.learning_rate_multiplier,
        train_router_bias_residual=args.train_router_bias_residual,
        router_bias_update=args.router_bias_update,
    )

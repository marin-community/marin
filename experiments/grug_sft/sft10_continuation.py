# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue the 10.05 Grug SFT for a few steps, checkpointing every step.

This is the 10.05 recipe (``special_token_lr.py`` as run by ``special_token_full_ab_grouped.py``) restored from
its final state, step 161750, with full MuonH optimizer state. Changes from 10.05:

* hardware: a smaller TPU v4 slice in us-central2 (next to the 10.05 checkpoint) or RNO2A H100 nodes with FA4
  attention and blocked-vocab cross-entropy; smaller geometries train on fewer sequences per step;
* data: a fixed sample of the 10.05 mixture (``sample_continuation_data.py``) instead of the one-epoch schedule;
* checkpoints: a permanent checkpoint after every step, for per-step evaluation.

The learning-rate schedule keeps the 10.05 shape (linear decay over the last 475 steps to 0.1 of 5e-5). Extending
``num_train_steps`` by a few steps continues that decay, so the rate stays at about 5e-6.
"""

import argparse
import dataclasses
import json
import logging
import math

import jmp
from fray.types import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.data.mixture import StopStrategy
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from rigging.filesystem.buckets import filesystem_for
from rigging.log_setup import configure_logging

from experiments.grug_sft.sample_continuation_data import MIX_PATH
from experiments.grug_sft.special_token_lr import ANCHORS, CONTEXT, EXPECTED_ANCHOR_IDS, MIXTURE_BLOCK_SIZE
from experiments.grug_sft.special_token_lr import TOKENIZER as TOKENIZER_10_05
from experiments.grug_sft.special_token_train import GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig

logger = logging.getLogger(__name__)

START_STEP = 161_750
SFT10_STEPS = 4_750  # the 10.05 run length, which sets the shape of its LR schedule
GPUS_PER_NODE = 8
SFT10_CHECKPOINT = (
    "gs://marin-us-central2/users/held/grug_sft/"
    "grug-67b-sft-20260929-special-token-lr-full-ab-4750-grouped/checkpoints/step-161750"
)


@dataclasses.dataclass(frozen=True)
class Storage:
    start: str
    tokenizer: str
    data: str
    checkpoints: str


RENO_ROOT = "s3://marin-rno2a/tmp/ttl=30d/dml/sft10_10400"
RENO = Storage(
    start=f"{RENO_ROOT}/start/step-161750",
    tokenizer=f"{RENO_ROOT}/tokenizer/2026.09.18",
    data=f"{RENO_ROOT}/data",
    checkpoints="s3://marin-rno2a/tmp/ttl=1d/dml/sft10_10400/runs",
)
CENTRAL2 = Storage(
    start=SFT10_CHECKPOINT,
    tokenizer=TOKENIZER_10_05,
    data="gs://marin-us-central2/tmp/ttl=30d/dml/sft10_10400/data",
    checkpoints="gs://marin-us-central2/tmp/ttl=1d/dml/sft10_10400/runs",
)


@dataclasses.dataclass(frozen=True)
class Geometry:
    """One hardware layout: each device holds ``CONTEXT / context_axis_size`` tokens of one sequence."""

    devices: int
    batch: int
    context_axis_size: int
    resources: ResourceConfig
    storage: Storage
    gpu: bool

    def __post_init__(self):
        if self.batch * self.context_axis_size != self.devices:
            raise ValueError(f"batch {self.batch} x context {self.context_axis_size} must equal {self.devices} devices")


def _reno(nodes: int) -> ResourceConfig:
    """Training runs as a child of the launcher, so submit the launcher itself with ``--target-cluster cw-rno2a``."""
    return ResourceConfig.with_gpu(
        "H100",
        count=GPUS_PER_NODE,
        cpu=32,
        ram="512g",
        disk="256g",
        replicas=nodes,
        preemptible=False,
    )


GEOMETRIES = {
    "reno-full": Geometry(512, 256, 2, _reno(64), RENO, gpu=True),
    "reno-half": Geometry(512, 128, 4, _reno(64), RENO, gpu=True),
    "reno-quarter": Geometry(256, 64, 4, _reno(32), RENO, gpu=True),
    # TPU v4 chips match 10.05's per-chip load (64k tokens) at a smaller batch.
    "v4-512": Geometry(256, 64, 4, ResourceConfig.with_tpu("v4-512", zone="us-central2-b"), CENTRAL2, gpu=False),
    "v4-1024": Geometry(512, 128, 4, ResourceConfig.with_tpu("v4-1024", zone="us-central2-b"), CENTRAL2, gpu=False),
}


def data_config(storage: Storage) -> LmDataConfig:
    sft_fraction = float(json.loads(MIX_PATH.read_text())["sft_fraction"])
    return LmDataConfig(
        tokenizer=storage.tokenizer,
        cache_dir=None,
        components={
            "sft": DatasetComponent(
                source=None, cache_dir=f"{storage.data}/sft", format=TextLmDatasetFormat(), pack=CONTEXT
            ),
            "pretrain": DatasetComponent(
                source=None, cache_dir=f"{storage.data}/replay", format=TextLmDatasetFormat(), flat_cache=True
            ),
        },
        train_weights={"sft": sft_fraction, "pretrain": 1.0 - sft_fraction},
        auto_build_caches=False,
        shuffle=True,
        block_cross_document_attention=True,
        mixture_block_size=MIXTURE_BLOCK_SIZE,
        stop_strategy=StopStrategy.RESTART_STRATEGY,
    )


def train(run_id: str, steps: int, geometry: Geometry) -> None:
    storage = geometry.storage
    # Read through the bucket-routed filesystem, which carries the CoreWeave endpoint.
    fs, metadata_path = filesystem_for(f"{storage.start}/metadata.json")
    with fs.open(metadata_path) as handle:
        metadata = json.load(handle)
    assert metadata["step"] == START_STEP, metadata
    tokenizer = load_tokenizer(storage.tokenizer)
    token_ids = tuple(token_id for token_id, _ in ANCHORS)
    anchor_ids = tuple(tuple(tokenizer.encode(text, add_special_tokens=False)) for _, text in ANCHORS)
    assert anchor_ids == EXPECTED_ANCHOR_IDS, anchor_ids
    output = f"{storage.checkpoints}/{run_id}"
    kernels = dict(attention_implementation="gpu_fa4_cute", ce_implementation="batched_xla") if geometry.gpu else {}
    model = dataclasses.replace(
        MoeMuonHHeuristic(min_lr_ratio=0.05).build_model_config(2560, seq_len=CONTEXT),
        disable_pko=True,
        disable_long_rope=True,
        sliding_window=2048,
        use_array_stacked_blocks=True,
        qk_mult=1.75,
        max_seq_len=CONTEXT,
        **kernels,
    )
    trainer = TrainerConfig(
        id=run_id,
        seed=0,
        train_batch_size=geometry.batch,
        per_device_parallelism=1,
        num_train_steps=START_STEP + steps,
        mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(entity="marin-community", project="marin_moe_sft", name=run_id, id=run_id, resume="allow"),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(
            axes={"expert": 1, "context": geometry.context_axis_size}, compute_mapping={"batch": ["data", "expert"]}
        ),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        initialize_from=storage.start,
        load_checkpoint=None,
        checkpointer=CheckpointerConfig(
            base_path=f"{output}/checkpoints",
            append_run_id_to_base_path=False,
            save_interval=None,
            keep=[dict(every=1)],
        ),
    )
    optimizer = GrugMoeMuonHConfig(
        learning_rate=5e-5,
        adam_lr=5e-5,
        beta1=0.9062,
        beta2=0.95,
        epsilon=3.8339433005718795e-15,
        weight_decay=0.0,
        max_grad_norm=None,
        min_lr_ratio=0.1,
        warmup=0,
        decay=SFT10_STEPS // 10,
        lr_schedule="linear",
        rmsnorm_to_adam=True,
    )
    run_grug(
        GrugRunConfig(
            model=model,
            data=data_config(storage),
            optimizer=optimizer,
            resources=geometry.resources,
            trainer=GrugTrainerConfig(
                trainer=trainer,
                sft_weights_only_init=False,
                data_start_step=START_STEP,
                max_data_epochs=None,
                special_token_lr_ids=token_ids,
                special_token_lr_multiplier=math.sqrt(32),
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
                replica_axis_size=1,
                context_axis_size=geometry.context_axis_size,
            ),
            eval=None,
        )
    )


if __name__ == "__main__":
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--geometry", choices=sorted(GEOMETRIES), required=True)
    args = parser.parse_args()
    train(args.run_id, args.steps, GEOMETRIES[args.geometry])

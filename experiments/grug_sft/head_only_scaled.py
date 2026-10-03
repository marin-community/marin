# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run versioned source ablations of the September 17 Grug 67B SFT recipe.

The baseline was recovered from Iris execution bundle
``85b3d97a5fed66084b5633e32e910aec59219657761e179fca9f968e436b147f``.
With no ``--exclude-group`` arguments, this module preserves its model, data,
optimizer, and hardware settings. Every invocation requires a version so a
reproduction or ablation cannot overwrite the released run.

An ablation removes complete allocation groups, renormalizes the surviving SFT
groups to 80% of the mixture, and leaves pretraining replay at 20%. It keeps the
1,000-update token budget. Surviving SFT stores may restart after one pass.
"""

import argparse
import dataclasses
import json
import logging
import math
from datetime import timedelta
from pathlib import Path

import jmp
from fray.types import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.data.mixture import StopStrategy
from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.training.training import temporary_checkpoint_base_path
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug_sft.data_ablation import ablated_group_weights, canonical_exclusions, versioned_run_id
from experiments.grug_sft.head_only_train import GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.grug_sft.prepared_store import SftTokenStore, sft_data_config
from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig

logger = logging.getLogger(__name__)

BASE_RUN_ID = "grug-67b-sft-20260917-head-only-sqrt32-frozen-router-1000"
OUTPUT_ROOT = "gs://marin-us-central2/users/held/grug_sft"
BASE = (
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew8-c06695/"
    "checkpoints/step-157000"
)
DATA_OUTPUT = "gs://marin-us-central2/users/held/grug_sft/grug-67b-sft-20260914-semantic-1000"
TOKENIZER = "gs://marin-us-central2/grug_sft/tokenizer/2026.09.12"
ANCHORS = {
    128006: " role",
    128007: " message",
    128002: " think",
    128003: " answer",
    128009: "<|end_of_text|>",
}
EXPECTED_ANCHOR_IDS = ((3560,), (1984,), (1781,), (4320,), (128001,))

CONTEXT = 262_144
BATCH = 256
START_STEP = 157_000
SFT_SHARE = 0.8
REPLAY_SHARE = 0.2


def source_group(name: str) -> str:
    """Return the allocation group for a prepared SFT store."""
    return "/".join(name.split("/")[:2]) if name.startswith("penfever-traces/") else name


def data_config(steps: int, excluded_groups: tuple[str, ...]) -> tuple[LmDataConfig, int]:
    """Build the fixed-budget baseline or leave-groups-out mixture."""
    raw = json.loads(StoragePath(DATA_OUTPUT + "/stores.json").read_text())
    stores = {name: SftTokenStore.model_validate(record) for name, record in raw.items()}
    plan = json.loads(StoragePath(DATA_OUTPUT + "/allocation.json").read_text())
    if steps != plan["steps"]:
        raise ValueError("Step count must match the recorded 1,000-step allocation")
    original_group_weights = plan["group_weights"]
    if not math.isclose(math.fsum(original_group_weights.values()), SFT_SHARE):
        raise ValueError("Recorded SFT group weights must sum to 0.8")
    group_weights = ablated_group_weights(original_group_weights, excluded_groups)
    block_size = steps * BATCH
    components = {}
    weights = {}
    for group, share in group_weights.items():
        selected = {name: store for name, store in stores.items() if source_group(name) == group}
        if not selected:
            raise ValueError(f"Missing stores for {group}")
        group_data = sft_data_config(selected, minimum_weight=1.000001 / (share * block_size))
        for name, component in group_data.components.items():
            key = group + "/" + name
            components[key] = component
            weights[key] = share * group_data.train_weights[name]

    replay = json.loads(Path(__file__).with_name("replay_skew8.json").read_text())
    replay_tail = {}
    replay_tail_weight = 0.0
    for name, record in replay.items():
        children = {}
        for path, copies in record["copies"].items():
            for copy in range(copies):
                children[f"{path}/{copy}"] = DatasetComponent(
                    source=None, cache_dir=path, format=TextLmDatasetFormat(), flat_cache=True
                )
        share = REPLAY_SHARE * record["weight"]
        if share * block_size < 1.000001:
            replay_tail.update({name + "/" + key: child for key, child in children.items()})
            replay_tail_weight += share
        else:
            components["pretrain/" + name] = ConcatDatasetComponent(children=children)
            weights["pretrain/" + name] = share
    if replay_tail and replay_tail_weight * block_size < 1.000001:
        smallest = min((key for key in weights if key.startswith("pretrain/")), key=weights.__getitem__)
        replay_tail.update(components.pop(smallest).children)
        replay_tail_weight += weights.pop(smallest)
    if replay_tail:
        components["pretrain/pooled"] = ConcatDatasetComponent(children=replay_tail)
        weights["pretrain/pooled"] = replay_tail_weight
    if not math.isclose(math.fsum(weights.values()), 1.0):
        raise ValueError("Mixture weights must sum to one")
    if not math.isclose(math.fsum(v for k, v in weights.items() if k.startswith("pretrain/")), REPLAY_SHARE):
        raise ValueError("Pretraining replay weights must sum to 0.2")
    if any(int(weight * block_size) == 0 for weight in weights.values()):
        raise ValueError("Mixture contains a component that rounds to zero")
    logger.info("SFT group shares after exclusions %s: %s", excluded_groups, group_weights)
    return (
        LmDataConfig(
            tokenizer=TOKENIZER,
            cache_dir=None,
            components=components,
            train_weights=weights,
            auto_build_caches=False,
            shuffle=True,
            block_cross_document_attention=True,
            mixture_block_size=block_size,
            stop_strategy=StopStrategy.RESTART_STRATEGY,
        ),
        steps,
    )


def train(version: str, excluded_groups: tuple[str, ...], steps: int) -> None:
    """Launch one versioned baseline or source-group ablation."""
    excluded_groups = canonical_exclusions(excluded_groups)
    run_id = versioned_run_id(BASE_RUN_ID, version, excluded_groups)
    output = f"{OUTPUT_ROOT}/{run_id}"
    data, steps = data_config(steps, excluded_groups)
    metadata = json.loads(StoragePath(BASE + "/metadata.json").read_text())
    if metadata["step"] != START_STEP:
        raise ValueError(f"Expected base step {START_STEP}, found {metadata['step']}")
    tokenizer = load_tokenizer(TOKENIZER)
    token_ids = tuple(ANCHORS)
    anchor_ids = tuple(tuple(tokenizer.encode(text, add_special_tokens=False)) for text in ANCHORS.values())
    if anchor_ids != EXPECTED_ANCHOR_IDS:
        raise ValueError(f"Tokenizer anchor IDs changed: {anchor_ids}")
    logger.info("Semantic token anchors: %s", dict(zip(token_ids, anchor_ids, strict=True)))
    model = dataclasses.replace(
        MoeMuonHHeuristic(min_lr_ratio=0.05).build_model_config(2560, seq_len=CONTEXT),
        disable_pko=True,
        disable_long_rope=True,
        sliding_window=2048,
        use_array_stacked_blocks=True,
        qk_mult=1.75,
        max_seq_len=CONTEXT,
    )
    trainer = TrainerConfig(
        id=run_id,
        seed=0,
        train_batch_size=BATCH,
        per_device_parallelism=1,
        num_train_steps=START_STEP + steps,
        mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            entity="marin-community",
            project="marin_moe",
            name=run_id,
            id=run_id,
            resume="allow",
            tags=["data-ablation", *(f"exclude:{group}" for group in excluded_groups)],
        ),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1, "context": 4}, compute_mapping={"batch": ["data", "expert"]}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        initialize_from=BASE,
        load_checkpoint=None,
        checkpointer=CheckpointerConfig(
            base_path=output + "/checkpoints",
            temporary_base_path=temporary_checkpoint_base_path(output),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=30),
            keep=None,
            keep_last_temporary_checkpoints=1,
        ),
    )
    optimizer = GrugMoeMuonHConfig(
        learning_rate=5e-5 * math.sqrt(32),
        adam_lr=5e-5 * math.sqrt(32),
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
    run_grug(
        GrugRunConfig(
            model=model,
            data=data,
            optimizer=optimizer,
            resources=ResourceConfig.with_tpu("v4-2048", zone="us-central2-b", preemptible=False),
            trainer=GrugTrainerConfig(
                trainer=trainer,
                sft_weights_only_init=False,
                data_start_step=START_STEP,
                max_data_epochs=1 if not excluded_groups else None,
                reinitialize_token_ids=token_ids,
                reinitialize_token_anchors=anchor_ids,
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
                replica_axis_size=1,
                context_axis_size=4,
            ),
            eval=None,
        )
    )


if __name__ == "__main__":
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Immutable experiment version appended to run identity")
    parser.add_argument(
        "--exclude-group",
        action="append",
        default=[],
        help="Allocation group to remove; repeat for a multi-group ablation",
    )
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    train(args.version, tuple(args.exclude_group), args.steps)

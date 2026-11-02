# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train SFT with LM-head-only initialization, higher special-token LR, and frozen router biases."""

import argparse
import dataclasses
import json
import logging
import math
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import jmp
from fray.types import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.data.mixture import StopStrategy
from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent, DatasetComponentBase, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.training.training import temporary_checkpoint_base_path
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug_sft.special_token_train import GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig

logger = logging.getLogger(__name__)
RUN_ID = "grug-67b-sft-20260928-special-token-lr-marin-identity-1000"
BASE = (
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew8-c06695/"
    "checkpoints/step-157000"
)
TOKENIZER = "gs://marin-us-central2/grug_sft/tokenizer/2026.09.18"
MIX_PATH = Path(__file__).with_name("special_token_1000_marin_identity_mix.json")
MIXTURE_BLOCK_SIZE = 64_000
MIN_SAMPLES_PER_BLOCK = 1.000001
SFT_COMPONENT_PREFIX = "sft/source/"
SFT_POOLED_COMPONENT = "sft/pooled"
PRETRAIN_COMPONENT_PREFIX = "pretrain/"
ANCHORS = (
    (128006, " role"),
    (128007, " message"),
    (128002, " think"),
    (128003, " answer"),
    (128005, " tool"),
    (128011, " end"),
    (128009, "<|end_of_text|>"),
)
EXPECTED_ANCHOR_IDS = ((3560,), (1984,), (1781,), (4320,), (5507,), (842,), (128001,))


CONTEXT = 262144
CONTEXT_AXIS_SIZE = 4
BATCH = 256
START_STEP = 157000
DEFAULT_STEPS = 1000


class StoreInfo(NamedTuple):
    cache_path: str
    tokenizer: str
    max_length: int
    packed_sequences: int


def _store_info(name: str, record: object) -> StoreInfo:
    if not isinstance(record, dict):
        raise ValueError(f"Store record for {name} must be an object")
    sources = record.get("sources")
    if not isinstance(sources, dict):
        raise ValueError(f"Store record for {name} has no source counts")
    try:
        return StoreInfo(
            cache_path=str(record["cache_path"]),
            tokenizer=str(record["tokenizer"]),
            max_length=int(record["max_length"]),
            packed_sequences=int(record["packed_sequences"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed store record for {name}") from error


def _store_component(store: StoreInfo) -> DatasetComponent:
    return DatasetComponent(
        source=None,
        cache_dir=store.cache_path,
        format=TextLmDatasetFormat(),
        pack=store.max_length,
    )


def _sft_components(
    stores: Mapping[str, StoreInfo],
    allocations: Mapping[str, int],
    *,
    sft_fraction: float,
    block_size: int = MIXTURE_BLOCK_SIZE,
    pooled_sources: set[str] | None = None,
) -> tuple[dict[str, DatasetComponentBase], dict[str, float]]:
    missing = sorted(set(allocations) - set(stores))
    if missing:
        raise ValueError(f"Missing token stores for selected SFT sources: {missing}")

    token_budget = sum(allocations.values())
    if token_budget <= 0:
        raise ValueError("SFT token allocation must be positive")

    components: dict[str, DatasetComponentBase] = {}
    weights: dict[str, float] = {}
    pooled: dict[str, DatasetComponent] = {}
    pooled_weight = 0.0
    minimum_weight = MIN_SAMPLES_PER_BLOCK / block_size
    for name, allocated_tokens in sorted(allocations.items()):
        store = stores[name]
        if store.tokenizer != TOKENIZER:
            raise ValueError(f"{name} uses tokenizer {store.tokenizer}, expected {TOKENIZER}")
        if store.max_length != CONTEXT:
            raise ValueError(f"{name} uses context length {store.max_length}, expected {CONTEXT}")
        if store.packed_sequences <= 0:
            raise ValueError(f"{name} has no packed training sequences")
        packed_capacity = store.packed_sequences * CONTEXT
        if allocated_tokens > packed_capacity:
            raise ValueError(f"{name} allocates {allocated_tokens} positions from a {packed_capacity}-position store")

        weight = sft_fraction * allocated_tokens / token_budget
        component = _store_component(store)
        should_pool = name in pooled_sources if pooled_sources is not None else weight < minimum_weight
        if should_pool:
            pooled[name] = component
            pooled_weight += weight
        else:
            key = SFT_COMPONENT_PREFIX + name
            components[key] = component
            weights[key] = weight

    if pooled and pooled_weight < minimum_weight:
        smallest = min(weights, key=weights.__getitem__)
        smallest_component = components.pop(smallest)
        assert isinstance(smallest_component, DatasetComponent)
        pooled[smallest.removeprefix(SFT_COMPONENT_PREFIX)] = smallest_component
        pooled_weight += weights.pop(smallest)
    if pooled:
        components[SFT_POOLED_COMPONENT] = ConcatDatasetComponent(children=pooled)
        weights[SFT_POOLED_COMPONENT] = pooled_weight

    zero_count = [name for name, weight in weights.items() if int(weight * block_size) == 0]
    if zero_count:
        raise ValueError(f"SFT components round to zero examples per mixture block: {zero_count}")
    return components, weights


def _one_epoch_weights(
    components: Mapping[str, DatasetComponentBase],
    allocations: Mapping[str, int],
    replay_weights: Mapping[str, float],
    *,
    block_size: int,
    run_sequences: int,
) -> list[tuple[int, dict[str, float]]]:
    """Spread each SFT source's packed examples evenly across complete mixture blocks."""
    if run_sequences % block_size:
        raise ValueError("One-epoch source mixing requires complete mixture blocks")
    if block_size % BATCH:
        raise ValueError("Mixture block boundaries must fall on training steps")
    num_blocks = run_sequences // block_size
    sft_counts: dict[str, int] = {}
    for name, component in components.items():
        if name == SFT_POOLED_COMPONENT:
            assert isinstance(component, ConcatDatasetComponent)
            sft_counts[name] = sum(allocations[child] // CONTEXT for child in component.children)
        elif name.startswith(SFT_COMPONENT_PREFIX):
            sft_counts[name] = allocations[name.removeprefix(SFT_COMPONENT_PREFIX)] // CONTEXT
    if any(count < num_blocks for count in sft_counts.values()):
        raise ValueError("Each SFT component needs at least one example per mixture block")
    if any(tokens % CONTEXT for tokens in allocations.values()):
        raise ValueError("One-epoch allocations must contain whole packed examples")

    replay_total_weight = sum(replay_weights.values())
    schedule = []
    for block in range(num_blocks):
        counts = {
            name: ((block + 1) * length // num_blocks) - (block * length // num_blocks)
            for name, length in sft_counts.items()
        }
        replay_draws = block_size - sum(counts.values())
        replay_targets = {name: replay_draws * weight / replay_total_weight for name, weight in replay_weights.items()}
        replay_counts = {name: max(1, int(target)) for name, target in replay_targets.items()}
        remainder = replay_draws - sum(replay_counts.values())
        if remainder < 0:
            raise ValueError("Replay components do not fit in the mixture block")
        largest_remainders = sorted(replay_targets, key=lambda name: replay_targets[name] % 1, reverse=True)
        for name in largest_remainders[:remainder]:
            replay_counts[name] += 1
        counts.update(replay_counts)
        assert sum(counts.values()) == block_size
        schedule.append((block * block_size // BATCH, {name: count / block_size for name, count in counts.items()}))
    return schedule


def data_config(steps: int, stores_manifest: str, *, mix_path: Path = MIX_PATH) -> tuple[LmDataConfig, int]:
    raw = json.loads(StoragePath(stores_manifest).read_text())
    if not isinstance(raw, dict):
        raise ValueError("SFT stores manifest must be an object")
    stores = {name: _store_info(name, record) for name, record in raw.items()}
    plan = json.loads(mix_path.read_text())
    if steps != plan["steps"]:
        raise ValueError(f"Step count must match the fixed {plan['steps']}-step allocation")
    if plan["batch_size"] != BATCH or plan["context_length"] != CONTEXT:
        raise ValueError("SFT allocation was prepared for a different batch size or context length")
    block_size = int(plan.get("mixture_block_size", MIXTURE_BLOCK_SIZE))
    allocations = {name: int(tokens) for name, tokens in plan["allocations_tokens"].items()}
    if plan.get("block_schedule") == "one_epoch":
        if steps * BATCH % block_size:
            raise ValueError("One-epoch source mixing requires complete mixture blocks")
        if missing := sorted(allocations.keys() - stores.keys()):
            raise ValueError(f"Missing token stores for selected SFT sources: {missing}")
        num_blocks = steps * BATCH // block_size
        pooled_sources = {name for name in allocations if stores[name].packed_sequences < num_blocks}
    else:
        pooled_sources = set(plan["pooled_sources"]) if "pooled_sources" in plan else None
    if pooled_sources is not None and not pooled_sources <= allocations.keys():
        raise ValueError("Pooled SFT sources must be included in the allocation")
    if any(tokens <= 0 for tokens in allocations.values()):
        raise ValueError("Every selected SFT source must have a positive token allocation")
    if sum(allocations.values()) != plan["sft_token_budget"]:
        raise ValueError("SFT allocation token counts do not match its declared budget")
    sft_fraction = float(plan["sft_fraction"])
    if not 0.0 < sft_fraction < 1.0:
        raise ValueError("SFT fraction must be between zero and one")
    expected_sft_tokens = int(steps * BATCH * CONTEXT * sft_fraction)
    if plan["sft_token_budget"] != expected_sft_tokens:
        raise ValueError(f"SFT allocation has {plan['sft_token_budget']} tokens, expected {expected_sft_tokens}")
    components, weights = _sft_components(
        stores, allocations, sft_fraction=sft_fraction, block_size=block_size, pooled_sources=pooled_sources
    )

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
        share = (1.0 - sft_fraction) * record["weight"]
        if share * block_size < MIN_SAMPLES_PER_BLOCK:
            replay_tail.update({name + "/" + key: child for key, child in children.items()})
            replay_tail_weight += share
        else:
            components[PRETRAIN_COMPONENT_PREFIX + name] = ConcatDatasetComponent(children=children)
            weights[PRETRAIN_COMPONENT_PREFIX + name] = share
    if replay_tail and replay_tail_weight * block_size < MIN_SAMPLES_PER_BLOCK:
        smallest = min((key for key in weights if key.startswith(PRETRAIN_COMPONENT_PREFIX)), key=weights.__getitem__)
        smallest_component = components.pop(smallest)
        assert isinstance(smallest_component, ConcatDatasetComponent)
        replay_tail.update(smallest_component.children)
        replay_tail_weight += weights.pop(smallest)
    if replay_tail:
        components[PRETRAIN_COMPONENT_PREFIX + "pooled"] = ConcatDatasetComponent(children=replay_tail)
        weights[PRETRAIN_COMPONENT_PREFIX + "pooled"] = replay_tail_weight
    if plan.get("block_schedule") == "one_epoch":
        if any(allocations[name] != stores[name].packed_sequences * CONTEXT for name in allocations):
            raise ValueError("One-epoch source mixing requires full packed-store allocations")
        replay_weights = {name: weight for name, weight in weights.items() if name.startswith(PRETRAIN_COMPONENT_PREFIX)}
        weights = _one_epoch_weights(
            components, allocations, replay_weights, block_size=block_size, run_sequences=steps * BATCH
        )
    elif "block_counts" in plan:
        block_counts = {name: int(count) for name, count in plan["block_counts"].items()}
        if set(block_counts) != set(components) or any(count <= 0 for count in block_counts.values()):
            raise ValueError("Block counts must cover every positive-weight mixture component")
        if sum(block_counts.values()) != block_size:
            raise ValueError("Block counts must sum to the mixture block size")
        weights = {name: block_counts[name] / block_size for name in components}
    weight_stages = weights if isinstance(weights, list) else [(0, weights)]
    for _, stage_weights in weight_stages:
        assert math.isclose(sum(stage_weights.values()), 1.0)
        if any(round(weight * block_size) == 0 for weight in stage_weights.values()):
            raise ValueError("Mixture contains a component that rounds to zero")
    if not isinstance(weights, list):
        replay_weight = sum(v for k, v in weights.items() if k.startswith(PRETRAIN_COMPONENT_PREFIX))
        if "block_counts" in plan:
            assert abs(replay_weight - (1.0 - sft_fraction)) <= 1 / block_size
        else:
            assert math.isclose(replay_weight, 1.0 - sft_fraction)
    logger.info(
        "%d-step mixture: %d SFT sources, %d top-level SFT components, %.3fB allocated SFT tokens",
        steps,
        len(allocations),
        sum(name.startswith("sft/") for name in components),
        sum(allocations.values()) / 1e9,
    )
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


def train(steps: int, stores_manifest: str, *, mix_path: Path = MIX_PATH, run_id: str = RUN_ID) -> None:
    data, steps = data_config(steps, stores_manifest, mix_path=mix_path)
    output = f"gs://marin-us-central2/users/held/grug_sft/{run_id}"
    metadata = json.loads((StoragePath(BASE) / "metadata.json").read_text())
    assert metadata["step"] == START_STEP
    tokenizer = load_tokenizer(TOKENIZER)
    token_ids = tuple(token_id for token_id, _ in ANCHORS)
    anchor_ids = tuple(tuple(tokenizer.encode(text, add_special_tokens=False)) for _, text in ANCHORS)
    assert anchor_ids == EXPECTED_ANCHOR_IDS, anchor_ids
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
        tracker=WandbConfig(entity="marin-community", project="marin_moe_sft", name=run_id, id=run_id, resume="allow"),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1, "context": CONTEXT_AXIS_SIZE}, compute_mapping={"batch": ["data", "expert"]}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        initialize_from=BASE,
        load_checkpoint=None,
        checkpointer=CheckpointerConfig(
            base_path=str(StoragePath(output) / "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(output),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=30),
            keep=None,
            keep_last_temporary_checkpoints=1,
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
                max_data_epochs=1,
                special_token_lr_ids=token_ids,
                special_token_lr_multiplier=math.sqrt(32),
                reinitialize_token_ids=token_ids,
                reinitialize_token_anchors=anchor_ids,
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
                replica_axis_size=1,
                context_axis_size=CONTEXT_AXIS_SIZE,
            ),
            eval=None,
        )
    )


if __name__ == "__main__":
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--stores-manifest", required=True)
    args = parser.parse_args()
    train(args.steps, args.stores_manifest)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run 100B-token proof and science curricula on supported preemptible TPU slices."""

import argparse
import dataclasses
import json
import logging
import math
from datetime import timedelta

import jmp
from fray.types import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.data.mixture import StopStrategy
from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat, TextLmDatasetFormat
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.datakit.sft_text import SftTokenStore
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import temporary_checkpoint_base_path
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging

from experiments.datasets.science_curricula import science_curriculum_datasets
from experiments.grug_sft.head_only_train import GrugRunConfig, GrugTrainerConfig, RouterFreeze, run_grug
from experiments.grug_sft.prepare_science_sft import SCIENCE_SFT_SOURCES
from experiments.grug_sft.regional_pool import (
    SNOWBALL_VERIFIED_WEIGHTS_REVISION,
    snowball_model_path,
    snowball_release_manifest_path,
)
from experiments.grug_sft.science_mix import (
    BASE_STEP,
    BATCH,
    CONTEXT,
    FINAL_STEP,
    MIX_BUDGETS,
    MIXTURE_BLOCK_SIZE,
    START_STEP,
    STEPS,
    ScienceMix,
    run_id,
)
from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig

logger = logging.getLogger(__name__)

MIN_EXPECTED_SEQUENCES_PER_BLOCK = 1.000001
SFT_VERSION = "2026.09.20"
SUPPORTED_TPU_ZONES = {
    ("v4-32", "us-central2-b"),
    ("v4-64", "us-central2-b"),
    ("v4-128", "us-central2-b"),
    ("v4-1024", "us-central2-b"),
    ("v4-2048", "us-central2-b"),
    ("v5p-64", "us-east5-a"),
    ("v5p-256", "us-east5-a"),
    ("v5p-1024", "us-east5-a"),
    ("v5p-2048", "us-central1-a"),
    ("v5p-2048", "us-east5-a"),
    ("v6e-128", "us-east5-b"),
    ("v6e-256", "us-east5-b"),
}

ANCHORS = {
    128006: " role",
    128007: " message",
    128002: " think",
    128003: " answer",
    128009: "<|end_of_text|>",
}
EXPECTED_ANCHOR_IDS = ((3560,), (1984,), (1781,), (4320,), (128001,))

PROOF_SOURCES = (
    "nemotron_sft_v3/math_proofs_v1/lean",
    "nemotron_sft_v3/math_proofs_v2/train",
)
SCIENCE_SOURCES = (
    "nemotron_sft_v3/science_v2/rqa",
    "nemotron_sft_v3/science_v2/so",
    "nemotron_sft_v3/science_v2/syn_mcq",
    "nemotron_sft_v3/science_v2/vendor",
)
TEXTBOOK_REASONING = "megascience/textbook-reasoning"
BIOCOLLECTION_SOURCES = (
    "biocollection/free_text_stream",
    "biocollection/instruction_stream",
)
SNOWBALL_REPLAY_SOURCES = ("nemotron_specialized/math_textbooks",)
SWALLOW_SOURCES = (
    "swallow-math-v2/qa",
    "swallow-math-v2/textbook",
)
ULTRADATA_SOURCES = ("ultradata-math/l2",)
CURRICULUM_TEXT_SOURCES = (
    "arxiv/physics-compatible-novel-abstracts",
    "hal/mathematics-licensed-abstracts",
    "openstax/physics",
    "openstax/chemistry",
    "openstax/biology",
    "openstax/college-physics",
    "openstax/university-physics",
    "openstax/organic-chemistry",
    "mit-ocw/physics/classical-mechanics",
    "mit-ocw/biology/fundamentals",
    "mit-ocw/chemistry/solid-state",
)


def _share(tokens_b: float) -> float:
    return tokens_b / 100.0


def _sft_store_path(name: str) -> str:
    if name not in SCIENCE_SFT_SOURCES:
        raise ValueError(f"Unknown science SFT source: {name}")
    return prefix_join(marin_prefix(), f"grug_sft/science_sft/{name}/{SFT_VERSION}")


def _add_weighted_group(
    *,
    group: str,
    share: float,
    entries: dict[str, tuple[DatasetComponent, float]],
    components: dict[str, DatasetComponent | ConcatDatasetComponent],
    weights: dict[str, float],
) -> None:
    if not entries or share <= 0:
        raise ValueError(f"{group} must have positive capacity and share")
    capacity = math.fsum(value for _, value in entries.values())
    if capacity <= 0:
        raise ValueError(f"{group} has no training sequences")
    pooled: dict[str, DatasetComponent] = {}
    pooled_weight = 0.0
    weighted_entries: list[tuple[str, DatasetComponent, float]] = []
    for name, (component, sequences) in entries.items():
        weight = share * sequences / capacity
        if weight * MIXTURE_BLOCK_SIZE < MIN_EXPECTED_SEQUENCES_PER_BLOCK:
            pooled[name] = component
            pooled_weight += weight
        else:
            weighted_entries.append((name, component, weight))

    if pooled and pooled_weight * MIXTURE_BLOCK_SIZE < MIN_EXPECTED_SEQUENCES_PER_BLOCK:
        if not weighted_entries:
            raise ValueError(f"{group} is too small to receive one sequence per mixture block")
        smallest = min(weighted_entries, key=lambda entry: entry[2])
        weighted_entries.remove(smallest)
        name, component, weight = smallest
        pooled[name] = component
        pooled_weight += weight

    for name, component, weight in weighted_entries:
        key = f"{group}/{name}"
        components[key] = component
        weights[key] = weight
    if pooled:
        components[f"{group}/pooled"] = ConcatDatasetComponent(children=pooled)
        weights[f"{group}/pooled"] = pooled_weight


def _text_entries(names: tuple[str, ...]) -> dict[str, tuple[DatasetComponent, float]]:
    handles = science_curriculum_datasets()
    entries = {}
    for name in names:
        cache = TokenizedCache.raw_load(handles[name].path(marin_prefix()))
        entries[name] = (cache.as_component(), cache.num_train_tokens / CONTEXT)
    return entries


def _sft_entries(names: tuple[str, ...]) -> dict[str, tuple[DatasetComponent, float]]:
    entries = {}
    for name in names:
        store = SftTokenStore.raw_load(_sft_store_path(name))
        entries[name] = (
            DatasetComponent(
                source=UrlDatasetSourceConfig(train_urls=[], validation_urls=[]),
                cache_dir=_sft_store_path(name),
                format=TextLmDatasetFormat(),
                pack=store.max_length,
            ),
            float(store.packed_sequences),
        )
    return entries


def _add_replay(
    share: float,
    components: dict[str, DatasetComponent | ConcatDatasetComponent],
    weights: dict[str, float],
) -> None:
    _add_weighted_group(
        group="replay",
        share=share,
        entries=_text_entries(SNOWBALL_REPLAY_SOURCES),
        components=components,
        weights=weights,
    )


def data_config(mix: ScienceMix, prebaked_root: str | None = None) -> LmDataConfig:
    """Build a one-pass mixture from materialized source measurements."""
    if prebaked_root is not None:
        return LmDataConfig(
            tokenizer=snowball_model_path(),
            cache_dir=None,
            components={
                "prebaked": DatasetComponent(
                    cache_dir=prefix_join(prebaked_root, mix.value),
                    format=PrebuiltLmDatasetFormat(
                        loss_weights_key="loss_weight",
                        segment_ids_key="segment_ids",
                    ),
                    flat_cache=True,
                )
            },
            train_weights={"prebaked": 1.0},
            auto_build_caches=False,
            shuffle=False,
            block_cross_document_attention=True,
            mixture_block_size=MIXTURE_BLOCK_SIZE,
            stop_strategy=StopStrategy.RESTART_STRATEGY,
        )
    budget = MIX_BUDGETS[mix]
    if not math.isclose(math.fsum(budget.as_dict().values()), 100.0):
        raise ValueError(f"{mix} budget must total 100B tokens")
    components: dict[str, DatasetComponent | ConcatDatasetComponent] = {}
    weights: dict[str, float] = {}
    _add_weighted_group(
        group="proof",
        share=_share(budget.proof),
        entries=_sft_entries(PROOF_SOURCES),
        components=components,
        weights=weights,
    )
    _add_weighted_group(
        group="swallow",
        share=_share(budget.swallow),
        entries=_text_entries(SWALLOW_SOURCES),
        components=components,
        weights=weights,
    )
    _add_weighted_group(
        group="ultradata",
        share=_share(budget.ultradata),
        entries=_text_entries(ULTRADATA_SOURCES),
        components=components,
        weights=weights,
    )
    _add_weighted_group(
        group="biocollection",
        share=_share(budget.biocollection),
        entries=_text_entries(BIOCOLLECTION_SOURCES),
        components=components,
        weights=weights,
    )
    _add_weighted_group(
        group="science",
        share=_share(budget.science),
        entries=_sft_entries(SCIENCE_SOURCES),
        components=components,
        weights=weights,
    )
    curriculum = _text_entries(CURRICULUM_TEXT_SOURCES)
    curriculum.update(_sft_entries((TEXTBOOK_REASONING,)))
    _add_weighted_group(
        group="curriculum",
        share=_share(budget.curriculum),
        entries=curriculum,
        components=components,
        weights=weights,
    )
    _add_replay(_share(budget.replay), components, weights)
    if not math.isclose(math.fsum(weights.values()), 1.0):
        raise ValueError(f"{mix} component weights sum to {math.fsum(weights.values())}")
    if any(weight * MIXTURE_BLOCK_SIZE < 1 for weight in weights.values()):
        raise ValueError(f"{mix} contains a component that rounds to zero sequences")
    logger.info("%s token budget: %s", mix, budget.as_dict())
    return LmDataConfig(
        tokenizer=snowball_model_path(),
        cache_dir=None,
        components=components,
        train_weights=weights,
        auto_build_caches=False,
        shuffle=True,
        block_cross_document_attention=True,
        mixture_block_size=MIXTURE_BLOCK_SIZE,
        stop_strategy=StopStrategy.RESTART_STRATEGY,
    )


def train(
    mix: ScienceMix,
    version: str,
    tpu: str,
    zone: str,
    wandb_mode: str | None = None,
    prebaked_root: str | None = None,
) -> None:
    """Dispatch one preemptible curriculum run on a supported TPU slice."""
    if (tpu, zone) not in SUPPORTED_TPU_ZONES:
        raise ValueError(f"Unsupported TPU and zone combination: {tpu} in {zone}")
    identity = run_id(mix, version)
    output = prefix_join(marin_prefix(), f"runs/{identity}")
    data = data_config(mix, prebaked_root)
    export_manifest = json.loads(
        StoragePath(prefix_join(snowball_release_manifest_path(), "export-manifest.json")).read_text()
    )
    if export_manifest["step"] != BASE_STEP:
        raise ValueError(f"Expected base step {BASE_STEP}, found {export_manifest['step']}")
    if export_manifest["verified_weights_revision"] != SNOWBALL_VERIFIED_WEIGHTS_REVISION:
        raise ValueError("Snowball release manifest does not verify the staged weights revision")
    tokenizer = load_tokenizer(snowball_model_path())
    token_ids = tuple(ANCHORS)
    anchor_ids = tuple(tuple(tokenizer.encode(text, add_special_tokens=False)) for text in ANCHORS.values())
    if anchor_ids != EXPECTED_ANCHOR_IDS:
        raise ValueError(f"Tokenizer anchor IDs changed: {anchor_ids}")
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
        id=identity,
        seed=0,
        train_batch_size=BATCH,
        per_device_parallelism=1,
        num_train_steps=FINAL_STEP,
        mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            entity="marin-community",
            project="marin_moe",
            name=identity,
            id=identity,
            mode=wandb_mode,
            resume="allow",
            tags=["science-curriculum", "100b", f"mix:{mix}"],
        ),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1, "context": 4}, compute_mapping={"batch": ["data", "expert"]}),
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
        decay=max(1, STEPS // 10),
        lr_schedule="linear",
        rmsnorm_to_adam=True,
    )
    run_grug(
        GrugRunConfig(
            model=model,
            data=data,
            optimizer=optimizer,
            resources=ResourceConfig.with_tpu(tpu, zone=zone, preemptible=True),
            trainer=GrugTrainerConfig(
                trainer=trainer,
                initialize_from_hf=snowball_model_path(),
                data_start_step=START_STEP,
                max_data_epochs=1,
                special_token_lr_ids=token_ids,
                special_token_lr_multiplier=math.sqrt(32),
                reinitialize_token_ids=token_ids,
                reinitialize_token_anchors=anchor_ids,
                router_freeze=RouterFreeze.BIAS,
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
    parser.add_argument("--mix", type=ScienceMix, choices=ScienceMix, nargs="+", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--tpu",
        choices=(
            "v4-32",
            "v4-64",
            "v4-128",
            "v4-1024",
            "v4-2048",
            "v5p-64",
            "v5p-256",
            "v5p-1024",
            "v5p-2048",
            "v6e-128",
            "v6e-256",
        ),
        required=True,
    )
    parser.add_argument("--zone", choices=("us-central2-b", "us-central1-a", "us-east5-a", "us-east5-b"), required=True)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--prebaked-root")
    args = parser.parse_args()
    for mix in args.mix:
        train(mix, args.version, args.tpu, args.zone, args.wandb_mode, args.prebaked_root)

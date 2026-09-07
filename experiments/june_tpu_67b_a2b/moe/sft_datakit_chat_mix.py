# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue the long-context checkpoint on Datakit chat data with a new WSD schedule."""

import dataclasses
import logging
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.data.text.datasets import (
    ConcatDatasetComponent,
    DatasetComponent,
    LmDataConfig,
)
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tracker.wandb import WandbConfig
from marin.datakit.chat import render_chat_source
from marin.datakit.sft_sources import all_sft_sources
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize
from rigging.filesystem.cluster_config import marin_prefix, region_from_prefix
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.launch_datakit_moe_mix import _TAIL_BUCKETS, _phase_weights
from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeMuonHConfig
from experiments.june_tpu_67b_a2b.moe.sft_launch import GrugMoeSFTConfig, run_grug_moe_sft_trial
from experiments.june_tpu_67b_a2b.moe.train import GrugTrainerConfig
from experiments.marin_tokenizer import MARIN_CHAT_TEMPLATE, marin_tokenizer

logger = logging.getLogger(__name__)

_BASE_CHECKPOINT = (
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew4-102e3c/"
    "checkpoints/step-157000/"
)
_PRETRAIN_STORE = "gs://marin-us-central2/datakit/store/june-67b-a2b-length64k/2026.08.24"
_RUN_NAME = "grug/moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k_2026.09.07"
_TOKENIZER = marin_tokenizer
_SEQ_LEN = 262_144
_BATCH_SIZE = 256
_TRAIN_STEPS = 2_000
_RESUME_STEP = 157_000
_WARMUP_STEPS = 60
_DECAY_STEPS = 200
_MIXTURE_BLOCK_SIZE = 49_152
_SFT_FRACTION = 0.8
_PRETRAIN_FRACTION = 0.2
_LONG_CONTEXT_SKEW = 4
_TOKENIZE_MAX_WORKERS = 64
_TAIL_BUCKETS_WITHOUT_LONG = frozenset({"c07q3", "c38q1", "c38q3", "c38q4", "c39q0", "c39q1", "c39q2", "c39q3", "c39q4"})
_SFT_LR = 5e-5
_MIN_SAMPLES_PER_MIXTURE_BLOCK = 1.000001
_TRAIN_REGION = "us-central2"
_TRAIN_ZONE = "us-central2-b"

# AdamH is used by the H100 SFT recipe because MuonH's Newton-Schulz workspace is
# expensive on 80 GB GPUs. On this v4-2048 geometry, however, AdamH's two expert
# moments make jit__init_state require 253 GiB per 32 GiB chip. The known-good
# 262K TPU context-extension run uses MuonH, whose single expert momentum fits.
_OPTIMIZER = GrugMoeMuonHConfig(
    learning_rate=_SFT_LR,
    adam_lr=_SFT_LR,
    beta1=0.9,
    beta2=0.95,
    epsilon=1e-8,
    max_grad_norm=None,
    weight_decay=0.0,
    min_lr_ratio=0.1,
    warmup=_WARMUP_STEPS,
    decay=_DECAY_STEPS,
    lr_schedule="linear",
    schedule_start_step=_RESUME_STEP,
    rmsnorm_to_adam=True,
)


@dataclass(frozen=True)
class _SftMixture:
    components: dict[str, DatasetComponent | ConcatDatasetComponent]
    weights: dict[str, float]
    deps: list[StepSpec]


def _floor_component_weights(weights: dict[str, float], total_weight: float) -> dict[str, float]:
    """Floor mixture weights without changing the aggregate mixture share."""
    minimum_weight = _MIN_SAMPLES_PER_MIXTURE_BLOCK / _MIXTURE_BLOCK_SIZE
    minimum_total = len(weights) * minimum_weight
    if minimum_total >= total_weight:
        raise ValueError("Mixture block is too small to sample every component")
    surplus = sum(max(weight - minimum_weight, 0.0) for weight in weights.values())
    target_surplus = total_weight - minimum_total
    return {
        name: minimum_weight + max(weight - minimum_weight, 0.0) * target_surplus / surplus
        for name, weight in weights.items()
    }


def _tokenize_rendered_source(name: str, rendered: StepSpec) -> StepSpec:
    cache_path = prefix_join(rendered.output_path, "levanter-cache")

    def build_cache(output_path: str) -> None:
        tokenize(
            TokenizeConfig(
                train_paths=[prefix_join(rendered.output_path, "outputs/main/*.parquet")],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=_TOKENIZER,
                max_workers=_TOKENIZE_MAX_WORKERS,
                allow_test_in_train=True,
                tags=["sft", name],
            )
        )

    return StepSpec(
        name=f"tokenized-rendered-chat/{name}",
        deps=[rendered],
        fn=build_cache,
        hash_attrs={"tokenizer": _TOKENIZER},
        override_output_path=cache_path,
    )


def _sft_mixture() -> _SftMixture:
    sources = all_sft_sources()
    total_tokens = sum(source.rough_token_count_b for source in sources.values())
    components: dict[str, DatasetComponent | ConcatDatasetComponent] = {}
    weights: dict[str, float] = {}
    deps: list[StepSpec] = []
    for name, source in sources.items():
        source = dataclasses.replace(
            source,
            format=dataclasses.replace(source.format, chat_template=MARIN_CHAT_TEMPLATE),
        )
        rendered = render_chat_source(source, tokenizer=_TOKENIZER)
        terminal = rendered.normalized
        tokenized = _tokenize_rendered_source(name, terminal)
        component = DatasetComponent(
            source=None,
            cache_dir=tokenized.output_path,
            format=TextLmDatasetFormat(),
            tags=["sft", name],
            pack=True,
        )
        components[f"sft/{name}"] = component
        weights[f"sft/{name}"] = _SFT_FRACTION * source.rough_token_count_b / total_tokens
        deps.append(tokenized)
    weights = _floor_component_weights(weights, _SFT_FRACTION)
    return _SftMixture(components=components, weights=weights, deps=deps)


def _pretrain_child(bucket: str, length: str) -> DatasetComponent:
    cluster = int(bucket[1:3])
    quality = int(bucket[-1])
    cache_dir = f"{_PRETRAIN_STORE}/cluster={cluster}/quality={quality}/length={length}"
    return DatasetComponent(
        source=None,
        cache_dir=cache_dir,
        format=TextLmDatasetFormat(),
        tags=[bucket, length],
        flat_cache=True,
        pack=True,
    )


def _pretrain_component(bucket: str) -> ConcatDatasetComponent:
    buckets = _TAIL_BUCKETS if bucket == "tail" else (bucket,)
    children: dict[str, DatasetComponent] = {}
    for child_bucket in buckets:
        children[f"{child_bucket}/lte_64k"] = _pretrain_child(child_bucket, "lte_64k")
        if child_bucket in _TAIL_BUCKETS_WITHOUT_LONG:
            continue
        gt = _pretrain_child(child_bucket, "gt_64k")
        for copy in range(_LONG_CONTEXT_SKEW):
            children[f"{child_bucket}/gt_64k/{copy}"] = gt
    return ConcatDatasetComponent(children=children, tags=["pretrain", bucket])


def _pretrain_components() -> tuple[dict[str, ConcatDatasetComponent], dict[str, float]]:
    phase_weights = _phase_weights(1)
    tail_weight = phase_weights["tail"] / len(_TAIL_BUCKETS)
    bucket_weights = {
        **{bucket: weight for bucket, weight in phase_weights.items() if bucket != "tail"},
        **dict.fromkeys(_TAIL_BUCKETS, tail_weight),
    }
    components = {f"pretrain/{bucket}": _pretrain_component(bucket) for bucket in bucket_weights}
    weights = {f"pretrain/{bucket}": _PRETRAIN_FRACTION * weight for bucket, weight in bucket_weights.items()}
    return components, _floor_component_weights(weights, _PRETRAIN_FRACTION)


def _model_config():
    model = MoeMuonHHeuristic(min_lr_ratio=0.05).build_model_config(2560, seq_len=_SEQ_LEN)
    return dataclasses.replace(
        model,
        disable_pko=True,
        disable_long_rope=True,
        sliding_window=2048,
        use_array_stacked_blocks=True,
        qk_mult=1.75,
        max_seq_len=_SEQ_LEN,
    )


def build() -> StepSpec:
    storage_paths = {
        "MARIN_PREFIX": marin_prefix(),
        "base checkpoint": _BASE_CHECKPOINT,
        "pretraining store": _PRETRAIN_STORE,
    }
    misplaced = {name: path for name, path in storage_paths.items() if region_from_prefix(path) != _TRAIN_REGION}
    if misplaced:
        raise ValueError(f"SFT storage must be in {_TRAIN_REGION}: {misplaced}")

    sft = _sft_mixture()
    pretrain_components, pretrain_weights = _pretrain_components()
    weights = {**sft.weights, **pretrain_weights}
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    zero_count_components = [name for name, weight in weights.items() if int(weight * _MIXTURE_BLOCK_SIZE) == 0]
    if zero_count_components:
        raise ValueError(f"Mixture weights round to zero samples per block: {zero_count_components}")

    data = LmDataConfig(
        tokenizer=_TOKENIZER,
        cache_dir=None,
        components={**sft.components, **pretrain_components},
        train_weights=weights,
        auto_build_caches=False,
        mixture_block_size=_MIXTURE_BLOCK_SIZE,
        block_cross_document_attention=True,
    )

    def train(output_path: str) -> None:
        run_grug_moe_sft_trial(
            GrugMoeSFTConfig(
                model=_model_config(),
                data=data,
                output_path=output_path,
                run_id=_RUN_NAME.removeprefix("grug/"),
                resources=ResourceConfig.with_tpu("v4-2048", zone=_TRAIN_ZONE, preemptible=False),
                steps=_RESUME_STEP + _TRAIN_STEPS,
                batch_size=_BATCH_SIZE,
                seed=0,
                mp="params=float32,compute=bfloat16,output=bfloat16",
                tracker=WandbConfig(
                    project="marin_moe_sft",
                    name=_RUN_NAME.removeprefix("grug/"),
                    tags=["sft", "datakit", "v4-2048", "ctx262k", "sft80-pretrain20"],
                ),
                optimizer=_OPTIMIZER,
                init_from_path=_BASE_CHECKPOINT,
                expert_parallel=1,
                context_parallel=4,
                per_device_parallelism=1,
                save_interval_minutes=30,
                grug_trainer=GrugTrainerConfig(
                    sft_weights_only_init=False,
                    z_loss_weight=1e-4,
                    ema_beta=None,
                    log_every=1,
                    replica_axis_size=1,
                    context_axis_size=4,
                ),
                eval=None,
            )
        )

    return StepSpec(
        name=_RUN_NAME,
        deps=sft.deps,
        fn=train,
        hash_attrs={
            "base_checkpoint": _BASE_CHECKPOINT,
            "tokenizer": _TOKENIZER,
            "seq_len": _SEQ_LEN,
            "batch_size": _BATCH_SIZE,
            "steps": _TRAIN_STEPS,
            "resume_step": _RESUME_STEP,
            "initialization": "full_state",
            "lr_schedule": "linear",
            "warmup_steps": _WARMUP_STEPS,
            "decay_steps": _DECAY_STEPS,
            "sft_fraction": _SFT_FRACTION,
            "pretrain_fraction": _PRETRAIN_FRACTION,
            "long_context_skew": _LONG_CONTEXT_SKEW,
        },
    )


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run([build()], max_concurrent=12)

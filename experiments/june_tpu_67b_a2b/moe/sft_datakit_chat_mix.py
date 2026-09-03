# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cold-start SFT over the structured Datakit chat registry on v4-2048."""

import dataclasses
import logging

from fray.cluster import ResourceConfig
from levanter.data.text.datasets import (
    ConcatDatasetComponent,
    DatasetComponent,
    LmDataConfig,
    UrlDatasetSourceConfig,
)
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tracker.wandb import WandbConfig
from marin.datakit.chat import render_chat_source
from marin.datakit.sft_sources import all_sft_sources
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.log_setup import configure_logging

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.launch_datakit_moe_mix import _TAIL_BUCKETS, _phase_weights
from experiments.june_tpu_67b_a2b.moe.sft_67b_a2b_2stage import _optimizer
from experiments.june_tpu_67b_a2b.moe.sft_launch import GrugMoeSFTConfig, run_grug_moe_sft_trial
from experiments.june_tpu_67b_a2b.moe.train import GrugTrainerConfig
from experiments.marin_tokenizer import marin_tokenizer

logger = logging.getLogger(__name__)

_BASE_CHECKPOINT = (
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew4-102e3c/"
    "checkpoints/step-157000/"
)
_PRETRAIN_STORE = "gs://marin-us-central2/datakit/store/june-67b-a2b-length64k/2026.08.24"
_RUN_NAME = "grug/moe_67b_a2b_step157k_sft_datakit80_pretrain20_ctx262k"
_TOKENIZER = marin_tokenizer
_SEQ_LEN = 262_144
_BATCH_SIZE = 256
_TRAIN_STEPS = 1_888
_MIXTURE_BLOCK_SIZE = 49_152
_SFT_FRACTION = 0.8
_PRETRAIN_FRACTION = 0.2
_LONG_CONTEXT_SKEW = 4
_TAIL_BUCKETS_WITHOUT_LONG = frozenset(
    {"c07q3", "c38q1", "c38q3", "c38q4", "c39q0", "c39q1", "c39q2", "c39q3", "c39q4"}
)


def _normalized_parquet_glob(step: StepSpec) -> str:
    return f"{step.output_path}/outputs/main/*.parquet"


def _sft_components() -> tuple[dict[str, DatasetComponent], dict[str, float], list[StepSpec]]:
    sources = all_sft_sources()
    total_tokens = sum(source.rough_token_count_b for source in sources.values())
    components: dict[str, DatasetComponent] = {}
    weights: dict[str, float] = {}
    deps: list[StepSpec] = []
    for name, source in sources.items():
        rendered = render_chat_source(source, tokenizer=_TOKENIZER)
        terminal = rendered.normalized
        source_config = UrlDatasetSourceConfig(
            train_urls=[_normalized_parquet_glob(terminal)],
            cache_dir=f"{terminal.output_path}/levanter-cache",
            format=TextLmDatasetFormat(),
        )
        components[f"sft/{name}"] = DatasetComponent(
            source=source_config,
            cache_dir=source_config.cache_dir,
            format=source_config.format,
            tags=["sft", name],
            pack=True,
            packing_slice_strategy="drop",
        )
        weights[f"sft/{name}"] = _SFT_FRACTION * source.rough_token_count_b / total_tokens
        deps.append(terminal)
    return components, weights, deps


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
    return (
        {f"pretrain/{bucket}": _pretrain_component(bucket) for bucket in phase_weights},
        {f"pretrain/{bucket}": _PRETRAIN_FRACTION * weight for bucket, weight in phase_weights.items()},
    )


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
    sft_components, sft_weights, deps = _sft_components()
    pretrain_components, pretrain_weights = _pretrain_components()
    weights = {**sft_weights, **pretrain_weights}
    assert abs(sum(weights.values()) - 1.0) < 1e-9

    data = LmDataConfig(
        tokenizer=_TOKENIZER,
        cache_dir=None,
        components={**sft_components, **pretrain_components},
        train_weights=weights,
        auto_build_caches=True,
        mixture_block_size=_MIXTURE_BLOCK_SIZE,
    )

    def train(output_path: str) -> None:
        run_grug_moe_sft_trial(
            GrugMoeSFTConfig(
                model=_model_config(),
                data=data,
                output_path=output_path,
                run_id=_RUN_NAME.removeprefix("grug/"),
                resources=ResourceConfig.with_tpu("v4-2048", preemptible=False),
                steps=_TRAIN_STEPS,
                batch_size=_BATCH_SIZE,
                seed=0,
                mp="params=float32,compute=bfloat16,output=bfloat16",
                tracker=WandbConfig(
                    project="marin_moe_sft",
                    name=_RUN_NAME.removeprefix("grug/"),
                    tags=["sft", "datakit", "v4-2048", "ctx262k", "sft80-pretrain20"],
                ),
                optimizer=_optimizer,
                init_from_path=_BASE_CHECKPOINT,
                expert_parallel=1,
                context_parallel=4,
                per_device_parallelism=1,
                save_interval_minutes=30,
                checkpoint_keep=[{"every": 250}],
                grug_trainer=GrugTrainerConfig(
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
        deps=deps,
        fn=train,
        hash_attrs={
            "base_checkpoint": _BASE_CHECKPOINT,
            "tokenizer": _TOKENIZER,
            "seq_len": _SEQ_LEN,
            "batch_size": _BATCH_SIZE,
            "steps": _TRAIN_STEPS,
            "sft_fraction": _SFT_FRACTION,
            "pretrain_fraction": _PRETRAIN_FRACTION,
            "long_context_skew": _LONG_CONTEXT_SKEW,
            "packing_slice_strategy": "drop",
        },
    )


if __name__ == "__main__":
    configure_logging(logging.INFO)
    StepRunner().run([build()], max_concurrent=12)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Repeat the seven smallest Delphi recipes, with two seed-0 controls.

Source: marin-community/marin W&B configs and execution logs, recovered 2026-09-22
(commit 9e946e47cfa74efe43cca1b74360c27f64dd1226). The source model class was
Qwen3 despite its logged use_qk_norm=False. Change VERSION for any recipe change.

After auditing the materialized configs, run this from the repository root with
WANDB_API_KEY exported. It submits all 16 runs through seven separate top-level
Iris jobs: one for the ten single-host runs and one per multi-host run. Separate
top-level jobs avoid JAX coordinator name collisions.

    (
      set -e
      delphi_batch="$(date -u +%Y%m%d-%H%M%S)-$$"

      submit_delphi() {
        local delphi_label="$1"
        shift
        uv run iris --cluster=marin job run --no-wait --no-cancel-on-exit \
          --priority batch --region us-central2 --cpu 1 --memory 2G --extra cpu \
          --job-name "delphi-seeds-${delphi_label}-${delphi_batch}" \
          -- python experiments/downstream_scaling/train_delphi_seeds.py "$@"
      }

      submit_delphi single-host --slugs 3e18 9e18 2e19 3e19
      for delphi_slug in 9e19 2e20 3e20; do
        for delphi_seed in 42 62746; do
          submit_delphi "${delphi_slug}-seed${delphi_seed}" \
            --slugs "$delphi_slug" --seeds "$delphi_seed"
        done
      done
    )

--slugs and --seeds select a subset of these 16 runs. Retry only groups whose
previous coordinator has stopped. Existing successful artifacts are skipped;
interrupted runs resume from their own checkpoints. Fresh coordinator names
preserve the training run IDs and output paths.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta

import draccus
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.adamh import AdamHConfig
from levanter.store.cache import CacheOptions
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import (
    LevanterCheckpoint,
    TrainLmOnPodConfig,
    apply_output_path,
    run_levanter_train_lm,
)
from rigging.filesystem.cluster_config import marin_prefix

VERSION = "2026.09.22"
EXPERIMENT_NAME = "delphi-seed-repeats"
TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
SEQUENCE_LENGTH = 4096
SOURCE_CHECKPOINT_ROOT = "gs://marin-us-central2/checkpoints/isoflop/"


@dataclass(frozen=True)
class ModelSpec:
    source_id: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    intermediate_dim: int
    batch_size: int
    per_device_batch: int
    num_train_steps: int
    tpu_type: str
    learning_rate: float
    adam_lr: float
    beta2: float
    epsilon: float
    tags: list[str]


MODELS = {
    "3e18": ModelSpec(
        source_id="isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6",
        hidden_dim=1024,
        num_layers=11,
        num_heads=8,
        intermediate_dim=4096,
        batch_size=8,
        per_device_batch=2,
        num_train_steps=37335,
        tpu_type="v4-8",
        learning_rate=0.00275999052746201,
        adam_lr=0.00033154735825338737,
        beta2=0.9999,
        epsilon=3.6604122149949323e-8,
        tags=["FLOPs=3.0e+18", "N=4.5e+08", "B=8", "steps=37335", "tokens=1.2e+09", "optimizer=completed-adamh"],
    ),
    "9e18": ModelSpec(
        source_id="isoflop-9e+18-d1152-L12-B16-adamh_scaling_v6",
        hidden_dim=1152,
        num_layers=12,
        num_heads=9,
        intermediate_dim=4608,
        batch_size=16,
        per_device_batch=4,
        num_train_steps=44317,
        tpu_type="v4-8",
        learning_rate=0.003011461785505323,
        adam_lr=0.000304311605183897,
        beta2=0.9999,
        epsilon=3.988017477238883e-8,
        tags=["FLOPs=9.0e+18", "N=5.5e+08", "B=16", "steps=44317", "tokens=2.9e+09", "optimizer=completed-adamh"],
    ),
    "2e19": ModelSpec(
        source_id="isoflop-2e+19-d1408-L15-B16-adamh_scaling_v6",
        hidden_dim=1408,
        num_layers=15,
        num_heads=11,
        intermediate_dim=5632,
        batch_size=16,
        per_device_batch=4,
        num_train_steps=55125,
        tpu_type="v4-8",
        learning_rate=0.002820610414485248,
        adam_lr=0.0002728525996258053,
        beta2=0.9999,
        epsilon=4.4478227499549274e-8,
        tags=["FLOPs=1.8e+19", "N=8.4e+08", "B=16", "steps=55125", "tokens=3.6e+09", "optimizer=completed-adamh"],
    ),
    "3e19": ModelSpec(
        source_id="isoflop-3e+19-d1536-L16-B32-adamh_scaling_v6",
        hidden_dim=1536,
        num_layers=16,
        num_heads=12,
        intermediate_dim=6144,
        batch_size=32,
        per_device_batch=8,
        num_train_steps=38014,
        tpu_type="v4-8",
        learning_rate=0.0036221806669679214,
        adam_lr=0.0003285714083231954,
        beta2=0.9999,
        epsilon=3.693565445007487e-8,
        tags=["FLOPs=3.0e+19", "N=1.0e+09", "B=32", "steps=38014", "tokens=5.0e+09", "optimizer=completed-adamh"],
    ),
    "9e19": ModelSpec(
        source_id="isoflop-9e+19-d1792-L18-B64-adamh_scaling_v6",
        hidden_dim=1792,
        num_layers=18,
        num_heads=14,
        intermediate_dim=7168,
        batch_size=64,
        per_device_batch=8,
        num_train_steps=40283,
        tpu_type="v4-16",
        learning_rate=0.004089070809779834,
        adam_lr=0.0003191861039276992,
        beta2=0.9999,
        epsilon=3.802170536455749e-8,
        tags=["FLOPs=9.0e+19", "N=1.4e+09", "B=64", "steps=40283", "tokens=1.1e+10", "optimizer=completed-adamh"],
    ),
    "2e20": ModelSpec(
        source_id="isoflop-2e+20-d2048-L21-B64-adamh_scaling_v6",
        hidden_dim=2048,
        num_layers=21,
        num_heads=16,
        intermediate_dim=8192,
        batch_size=64,
        per_device_batch=4,
        num_train_steps=56477,
        tpu_type="v4-32",
        learning_rate=0.003694876216643079,
        adam_lr=0.0002695686680789442,
        beta2=0.9999,
        epsilon=4.502006886217921e-8,
        tags=["FLOPs=1.8e+20", "N=1.9e+09", "B=64", "steps=56477", "tokens=1.5e+10", "optimizer=completed-adamh"],
    ),
    "3e20": ModelSpec(
        source_id="isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6",
        hidden_dim=2304,
        num_layers=23,
        num_heads=18,
        intermediate_dim=9216,
        batch_size=128,
        per_device_batch=4,
        num_train_steps=35510,
        tpu_type="v4-64",
        learning_rate=0.004878221521523273,
        adam_lr=0.00033996075743610645,
        beta2=0.99980001,
        epsilon=3.5698237912888785e-8,
        tags=["FLOPs=3.0e+20", "N=2.5e+09", "B=128", "steps=35510", "tokens=1.9e+10", "optimizer=completed-adamh"],
    ),
}

DEFAULT_RUNS: tuple[tuple[str, int], ...] = (
    *((slug, seed) for slug in MODELS for seed in (42, 62746)),
    ("3e18", 0),
    ("9e18", 0),
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    cache_path: str
    source_url: str
    weight: float = 0.0
    text_key: str = "text"


# Historical paths are read-only. Adoption writes records under our namespace.
# Preserve this execution-log order: W&B alphabetized the component dictionary.
DATASETS = (
    DatasetSpec(
        name="nemotron_cc/hq_actual",
        cache_path="gs://marin-us-central2/tokenized/nemotron_cc/hq_actual-5af4cc",
        source_url="gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl/quality=high/kind=actual/**/*.jsonl.gz",
        weight=0.91351,
    ),
    DatasetSpec(
        name="nemotron_cc/hq_synth",
        cache_path="gs://marin-us-central2/tokenized/nemotron_cc/hq_synth-3525e2",
        source_url="gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl/quality=high/kind=synthetic/**/*.jsonl.gz",
        weight=2.72,
    ),
    DatasetSpec(
        name="nemotron_cc/medium_high",
        cache_path="gs://marin-us-central2/tokenized/nemotron_cc/medium_high-d21701",
        source_url="gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl/quality=medium-high/**/*.jsonl.gz",
        weight=0.82471,
    ),
    DatasetSpec(
        name="nemotron_cc/medium",
        cache_path="gs://marin-us-central2/tokenized/nemotron_cc/medium-d86506",
        source_url="gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl/quality=medium/**/*.jsonl.gz",
        weight=3.38,
    ),
    DatasetSpec(
        name="nemotron_cc/medium_low",
        cache_path="gs://marin-us-central2/tokenized/nemotron_cc/medium_low-0fdb07",
        source_url="gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl/quality=medium-low/**/*.jsonl.gz",
        weight=1.54,
    ),
    DatasetSpec(
        name="nemotron_cc/low_actual",
        cache_path="gs://marin-us-central2/tokenized/nemotron_cc/low_actual-cb3f2c",
        source_url="gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl/quality=low/kind=actual/**/*.jsonl.gz",
        weight=0.70123,
    ),
    DatasetSpec(
        name="nemotron_cc/low_synth",
        cache_path="gs://marin-us-central2/tokenized/nemotron_cc/low_synth-3c57b3",
        source_url="gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl/quality=low/kind=synthetic/**/*.jsonl.gz",
        weight=0.62771,
    ),
    DatasetSpec(
        name="starcoderdata",
        cache_path="gs://marin-us-central2/tokenized/starcoderdata-12f018/",
        source_url="gs://marin-us-central2/raw/starcoderdata-720c8c",
        weight=0.25,
        text_key="content",
    ),
    DatasetSpec(
        name="proofpile_2",
        cache_path="gs://marin-us-central2/tokenized/proofpile_2-4a35c7/",
        source_url="gs://marin-us-central2/raw/proof-pile-2-f1b1d8/901a927/huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/901a927",
        weight=0.055,
    ),
    DatasetSpec(
        name="paloma/4chan",
        cache_path="gs://marin-us-central2/tokenized/paloma/4chan-496ad5",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/4chan_meta_sep/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/c4_100_domains",
        cache_path="gs://marin-us-central2/tokenized/paloma/c4_100_domains-2b6db7",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/c4_100_domains/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/c4_en",
        cache_path="gs://marin-us-central2/tokenized/paloma/c4_en-cf1f79",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/c4_en/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/dolma-v1_5",
        cache_path="gs://marin-us-central2/tokenized/paloma/dolma-v1_5-d3bed7",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/dolma-v1_5/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/dolma_100_programing_languages",
        cache_path="gs://marin-us-central2/tokenized/paloma/dolma_100_programing_languages-369132",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/dolma_100_programing_languages/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/dolma_100_subreddits",
        cache_path="gs://marin-us-central2/tokenized/paloma/dolma_100_subreddits-f25f70",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/dolma_100_subreddits/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/falcon-refinedweb",
        cache_path="gs://marin-us-central2/tokenized/paloma/falcon-refinedweb-75d43b",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/falcon-refinedweb/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/gab",
        cache_path="gs://marin-us-central2/tokenized/paloma/gab-ccaced",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/gab/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/m2d2_s2orc_unsplit",
        cache_path="gs://marin-us-central2/tokenized/paloma/m2d2_s2orc_unsplit-7dbcc1",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/m2d2_s2orc_unsplit/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/m2d2_wikipedia_unsplit",
        cache_path="gs://marin-us-central2/tokenized/paloma/m2d2_wikipedia_unsplit-b33d23",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/m2d2_wikipedia_unsplit/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/manosphere_meta_sep",
        cache_path="gs://marin-us-central2/tokenized/paloma/manosphere_meta_sep-a07891",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/manosphere_meta_sep/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/mc4",
        cache_path="gs://marin-us-central2/tokenized/paloma/mc4-ea36a2",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/mc4/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/ptb",
        cache_path="gs://marin-us-central2/tokenized/paloma/ptb-628036",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/ptb/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/redpajama",
        cache_path="gs://marin-us-central2/tokenized/paloma/redpajama-9d4ddd",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/redpajama/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/twitterAAE_HELM_fixed",
        cache_path="gs://marin-us-central2/tokenized/paloma/twitterAAE_HELM_fixed-2e17c1",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/twitterAAE_HELM_fixed/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="paloma/wikitext_103",
        cache_path="gs://marin-us-central2/tokenized/paloma/wikitext_103-1f5636",
        source_url="gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/wikitext_103/val/val*.jsonl.gz",
    ),
    DatasetSpec(
        name="uncheatable_eval/wikipedia_english",
        cache_path="gs://marin-us-central2/tokenized/uncheatable_eval/wikipedia_english-ba27aa",
        source_url="gs://marin-us-central2/raw/uncheatable-eval/latest-364401/wikipedia_english_*.jsonl.gz",
    ),
    DatasetSpec(
        name="uncheatable_eval/github_python",
        cache_path="gs://marin-us-central2/tokenized/uncheatable_eval/github_python-00e7de",
        source_url="gs://marin-us-central2/raw/uncheatable-eval/latest-364401/github_python_*.jsonl.gz",
    ),
    DatasetSpec(
        name="uncheatable_eval/github_cpp",
        cache_path="gs://marin-us-central2/tokenized/uncheatable_eval/github_cpp-d0da6b",
        source_url="gs://marin-us-central2/raw/uncheatable-eval/latest-364401/github_cpp_*.jsonl.gz",
    ),
    DatasetSpec(
        name="uncheatable_eval/bbc_news",
        cache_path="gs://marin-us-central2/tokenized/uncheatable_eval/bbc_news-2ff739",
        source_url="gs://marin-us-central2/raw/uncheatable-eval/latest-364401/bbc_news_*.jsonl.gz",
    ),
    DatasetSpec(
        name="uncheatable_eval/arxiv_physics",
        cache_path="gs://marin-us-central2/tokenized/uncheatable_eval/arxiv_physics-713363",
        source_url="gs://marin-us-central2/raw/uncheatable-eval/latest-364401/arxiv_physics_*.jsonl.gz",
    ),
    DatasetSpec(
        name="uncheatable_eval/arxiv_computer_science",
        cache_path="gs://marin-us-central2/tokenized/uncheatable_eval/arxiv_computer_science-9760a5",
        source_url="gs://marin-us-central2/raw/uncheatable-eval/latest-364401/arxiv_computer_science_*.jsonl.gz",
    ),
    DatasetSpec(
        name="uncheatable_eval/ao3_english",
        cache_path="gs://marin-us-central2/tokenized/uncheatable_eval/ao3_english-55e735",
        source_url="gs://marin-us-central2/raw/uncheatable-eval/latest-364401/ao3_english_*.jsonl.gz",
    ),
)


def _data_config(ctx: StepContext, caches: tuple[ArtifactStep[TokenizedCache], ...]) -> LmDataConfig:
    components = {}
    for spec, cache in zip(DATASETS, caches, strict=True):
        cache_path = ctx.artifact_path(cache)
        fmt = TextLmDatasetFormat(text_key=spec.text_key)
        source = UrlDatasetSourceConfig(
            cache_dir=cache_path,
            format=fmt,
            tags=[],
            train_urls=[spec.source_url] if spec.weight else [],
            validation_urls=[] if spec.weight else [spec.source_url],
        )
        components[spec.name] = DatasetComponent(
            source=source,
            cache_dir=cache_path,
            format=fmt,
            pack=None,
            split="validation",
            flat_cache=False,
            tags=[],
        )
    return LmDataConfig(
        tokenizer=TOKENIZER,
        cache_dir=None,
        components=components,
        train_weights={spec.name: spec.weight for spec in DATASETS},
        auto_build_caches=False,
        cache_options=CacheOptions(batch_size=128),
        shuffle=True,
        permutation_type="feistel",
        block_cross_document_attention=True,
        enforce_eos=True,
        mixture_block_size=2048,
        stop_strategy="restart",
        shuffle_before_trainval_split=True,
    )


def _training_step(
    slug: str, seed: int, caches: tuple[ArtifactStep[TokenizedCache], ...]
) -> ArtifactStep[LevanterCheckpoint]:
    spec = MODELS[slug]
    name = f"{EXPERIMENT_NAME}-{slug}-seed{seed}"
    run_id = f"{name}-{VERSION}"
    resources = ResourceConfig.with_tpu(spec.tpu_type, regions=["us-central2"])

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        model = Qwen3Config(
            hidden_dim=spec.hidden_dim,
            intermediate_dim=spec.intermediate_dim,
            num_layers=spec.num_layers,
            num_heads=spec.num_heads,
            num_kv_heads=spec.num_heads,
            head_dim=None,
            max_seq_len=SEQUENCE_LENGTH,
            activation_function=ActivationFunctionEnum.silu,
            initializer_range=0.02,
            layer_norm_epsilon=1e-5,
            tie_word_embeddings=False,
            hybrid_norm=False,
            input_embedding_norm=False,
            use_qk_norm=False,  # Qwen3 enables QK norm in attention_config regardless of this inherited field.
            use_bias=False,
            use_layer_norm_weight=True,
            upcast_attn=False,
            scan_layers=True,
            gradient_checkpointing=True,
            attn_backend=None,
            flash_attention_block_size=None,
            reference_checkpoint="NousResearch/Llama-2-7b-hf",
            tokenizer=None,
            use_sliding_window=False,
            sliding_window=4096,
            rope=Llama3RotaryEmbeddingsConfig(
                theta=500000,
                factor=8,
                low_freq_factor=1,
                high_freq_factor=4,
                original_max_position_embeddings=8192,
            ),
        )
        optimizer = AdamHConfig(
            learning_rate=spec.learning_rate,
            adam_lr=spec.adam_lr,
            beta1=0.9,
            beta2=spec.beta2,
            epsilon=spec.epsilon,
            max_grad_norm=0.1,
            weight_decay=0.1,
            lr_schedule="linear",
            warmup=0.1,
            decay=0.2,
            min_lr_ratio=0.0,
            rewarmup=0.0,
        )
        trainer = TrainerConfig(
            seed=seed,
            id=run_id,
            tracker=WandbConfig(
                entity="marin-community",
                project="marin",
                id=run_id,
                name=run_id,
                replicate_path=ctx.output_path,
                resume="allow",
                tags=list(spec.tags),
                save_code=True,
            ),
            mp=jmp.get_policy("p=f32,c=bf16,o=bf16"),
            train_batch_size=spec.batch_size,
            per_device_parallelism=spec.per_device_batch,
            per_device_eval_parallelism=spec.per_device_batch,
            num_train_steps=spec.num_train_steps,
            allow_nondivisible_batch_size=True,
            mesh=MeshConfig(
                axes={"data": -1, "model": 1, "replica": 1},
                dcn_axes={"replica_dcn": -1},
                batch_axis_name="batch",
                compute_mapping={
                    "token": ["replica_dcn", "replica", "data"],
                    "token_repeat": ["replica_dcn", "replica", "data"],
                },
                param_mapping={"embed": "data"},
            ),
            use_explicit_mesh_axes=False,
            jax_config={"jax_softmax_custom_jvp": True, "jax_threefry_partitionable": True},
            require_accelerator=True,
            steps_per_eval=1000,
            max_eval_batches=None,
            watch=WatchConfig(
                watch_targets=["grads", "params"],
                interval=10,
                include_norms=True,
                include_per_parameter_norms=True,
                include_histograms=False,
                split_scan_layers=True,
            ),
            profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=25, perfetto_link=False),
            checkpointer=CheckpointerConfig(
                base_path=ctx.output_path,
                keep=None,
                save_interval=timedelta(minutes=10),
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=True,
                keep_last_temporary_checkpoints=1,
            ),
            initialize_from=None,
            load_checkpoint=None,
            load_checkpoint_path=None,
            allow_partial_checkpoint=False,
        )
        train_config = TrainLmConfig(
            data=_data_config(ctx, caches),
            trainer=trainer,
            model=model,
            optimizer=optimizer,
            train_seq_len=SEQUENCE_LENGTH,
            data_seed=None,
            z_loss_weight=1e-7,
            initialize_from_hf=False,
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=None,
            use_hf_model_config=False,
            pad_tokenizer_to_match_model=False,
            hf_save_steps=spec.num_train_steps,
            hf_upload=None,
            merged_hf_upload=None,
            eval_harness=None,
            eval_harness_steps=10000,
            log_entropy=False,
        )
        return TrainLmOnPodConfig(
            train_config=train_config,
            resources=resources,
            output_path=ctx.output_path,
            auto_build_caches=False,
        )

    return ArtifactStep(
        name=name,
        version=VERSION,
        artifact_type=LevanterCheckpoint,
        run=remote(run_levanter_train_lm, resources=resources),
        build_config=build_config,
        deps=caches,
    )


def build(runs: tuple[tuple[str, int], ...] = DEFAULT_RUNS) -> list[ArtifactStep[LevanterCheckpoint]]:
    """Build the selected training graph without reading or writing storage."""
    if not runs or len(set(runs)) != len(runs):
        raise ValueError("Select at least one run, with no duplicate (slug, seed) pairs.")
    if invalid := set(runs) - set(DEFAULT_RUNS):
        raise ValueError(f"Runs outside DEFAULT_RUNS: {sorted(invalid)}")
    caches = tuple(
        ArtifactStep.adopt(
            name=f"{EXPERIMENT_NAME}-cache-{spec.name}",
            version=VERSION,
            source=spec.cache_path,
            kind=TokenizedCache,
            config={"tokenizer": TOKENIZER, "format": {"text_key": spec.text_key}, "tags": []},
        )
        for spec in DATASETS
    )
    return [_training_step(slug, seed, caches) for slug, seed in runs]


def selected_runs(
    slugs: tuple[str, ...] | None = None, seeds: tuple[int, ...] | None = None
) -> tuple[tuple[str, int], ...]:
    """Filter the fixed run list, preserving its order and rejecting invalid selections."""
    for label, values, allowed in (
        ("slugs", slugs, set(MODELS)),
        ("seeds", seeds, {seed for _, seed in DEFAULT_RUNS}),
    ):
        if values is not None:
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{label} must be nonempty with no duplicates.")
            if invalid := set(values) - allowed:
                raise ValueError(f"Unknown {label}: {sorted(invalid)}")
    runs = tuple(
        (slug, seed)
        for slug, seed in DEFAULT_RUNS
        if (slugs is None or slug in slugs) and (seeds is None or seed in seeds)
    )
    if not runs:
        raise ValueError("The filters select no planned run.")
    return runs


def validate_outputs(steps: list[ArtifactStep[LevanterCheckpoint]], prefix: str) -> None:
    """Check resolved run identities and write paths before submitting any step."""
    run_ids: set[str] = set()
    for step in steps:
        pod_config = materialized_config(step, prefix)
        config = apply_output_path(pod_config.train_config, pod_config.output_path)
        tracker = config.trainer.tracker
        if not isinstance(tracker, WandbConfig):
            raise ValueError(f"{step.name}: expected one W&B tracker.")
        encoded = json.dumps(draccus.encode(config))
        forbidden = (SOURCE_CHECKPOINT_ROOT, *(spec.source_id for spec in MODELS.values()))
        if any(source in encoded for source in forbidden):
            raise ValueError(f"{step.name}: a config field still references an original run.")
        if tracker.replicate_path != step.path(prefix) or pod_config.output_path != step.path(prefix):
            raise ValueError(f"{step.name}: output and tracker paths must match the artifact path.")
        expected_id = f"{step.name}-{VERSION}"
        if config.trainer.id != expected_id or tracker.id != expected_id or expected_id in run_ids:
            raise ValueError(f"{step.name}: trainer and W&B IDs must be unique and match the artifact.")
        if config.trainer.checkpointer.keep is not None or config.hf_save_steps != config.trainer.num_train_steps:
            raise ValueError(f"{step.name}: expected final-only permanent and HF checkpoints.")
        run_ids.add(expected_id)


def main() -> None:
    """Submit the selected runs after validating their identities and checkpoint policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slugs", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=16)
    args = parser.parse_args()
    if args.max_concurrent <= 0:
        parser.error("--max-concurrent must be positive.")
    steps = build(
        selected_runs(
            tuple(args.slugs) if args.slugs is not None else None,
            tuple(args.seeds) if args.seeds is not None else None,
        )
    )
    validate_outputs(steps, marin_prefix())
    StepRunner().run([step.lower() for step in steps], max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()

"""Programmatic replica of config/memorize/splice_comma_600m_multi_40k_central2.yaml
with an explicit multi-document P(z) eval loop that logs:
  - A histogram per document (pz_multi/doc_k/hist)
  - A combined histogram across all docs (pz_multi/all/hist)

Launchable via Marin's Executor on a TPU pod.

Usage:
  uv run python experiments/memorize/splice_comma_600m_multi_central2.py
"""

from __future__ import annotations

from datetime import timedelta

import jmp

from levanter.data.splice_dataset import SpliceMultiDocumentLMConfig
from levanter.data.text import (
    LMMixtureDatasetConfig,
    UrlDatasetSourceConfig,
    TextLmDatasetFormat,
)
from levanter.eval_pz_multi_doc import PzMultiDocConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.llama import LlamaConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.optim import AdamConfig
from levanter.trainer import TrainerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import RayConfig
from levanter.store.cache import CacheOptions

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.resources import TpuPodConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm


def build_config() -> TrainLmConfig:
    # --------------------
    # Data (multi-document splice from Wikimedia with coverage-balanced placement)
    # --------------------
    wikimedia = UrlDatasetSourceConfig(
        cache_dir="gs://marin-us-central2/tokenized/common_pile/wikimedia-53a667",
        train_urls=[
            "gs://marin-us-central2/raw/common_pile/wikimedia_filtered-0641bb8",
        ],
        validation_urls=[],
        tags=[],
        format=TextLmDatasetFormat(text_key="text"),
    )

    base_mixture = LMMixtureDatasetConfig(
        tokenizer="meta-llama/Meta-Llama-3.1-8B",
        cache_dir=None,
        shuffle=False,
        shuffle_per_epoch=True,
        configs={
            "common_pile/wikimedia": wikimedia,
        },
        train_weights={
            "common_pile/wikimedia": 1.0,
        },
        max_train_batches={
            "common_pile/wikimedia": 1,
        },
        mixture_block_size=2048,
        cache_options=CacheOptions(
            batch_size=128,
            num_shard_groups=128,
            target_size_per_flush="512MB",
        ),
    )

    # Multi-document splice configuration: 20 docs around ~40k tokens
    data = SpliceMultiDocumentLMConfig(
        tokenizer="meta-llama/Meta-Llama-3.1-8B",
        base=base_mixture,
        shuffle_per_epoch=True,
        dataset_name="common_pile/wikimedia",

        num_docs=20,
        min_doc_length=38_000,
        max_doc_length=45_000,
        doc_select_mode="longest",
        strict_num_docs=True,

        content_length=4096,
        content_stride=1,
        offset_stride=1,
        content_start_mode="coverage_balanced",
        min_copy_len=128,
        alpha=0.8,

        balance_mode="by_temperature",
        balance_tau=0.7,
        adaptive_k=True,
        offset_jitter=2,
    )

    # --------------------
    # Model (~600M Llama, Llama3 rotary, seq_len=4096)
    # --------------------
    model = LlamaConfig(
        seq_len=4096,
        hidden_dim=1024,
        intermediate_dim=3584,
        num_heads=16,
        num_kv_heads=8,
        num_layers=24,
        gradient_checkpointing=True,
        tie_word_embeddings=False,
        activation_function=ActivationFunctionEnum.silu,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        scan_layers=True,
        rope=Llama3RotaryEmbeddingsConfig(
            theta=500000,
            factor=8,
            low_freq_factor=1,
            high_freq_factor=4,
            original_max_position_embeddings=8192,
        ),
        reference_checkpoint="NousResearch/Llama-2-7b-hf",
    )

    # --------------------
    # Trainer
    # --------------------
    trainer = TrainerConfig(
        seed=0,
        ray=RayConfig(auto_start_cluster=False, start_workers=False),
        tracker=WandbConfig(
            project="marin",
            tags=["memorize", "splice", "600m", "wikimedia", "central2", "multi", "40k"],
            resume="allow",
            save_code=True,
            name="llama_600m_splice_wikimedia_multi_40k_tau07_central2",
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        model_axis_size=1,
        per_device_parallelism=-1,
        train_batch_size=256,
        num_train_steps=2000,
        allow_nondivisible_batch_size=True,
        steps_per_eval=1_000_000,
        max_eval_batches=10,
        checkpointer=CheckpointerConfig(
            save_interval=timedelta(hours=100000),
            keep=[dict(every=1_000_000)],
        ),
        tensor_parallel_axes=["heads", "mlp"],
        fsdp_axis="embed",
        batch_axis="batch",
        axis_resources={
            "token": ("replica", "data"),
            "token_repeat": ("replica", "data"),
        },
    )

    # --------------------
    # Optimizer (Adam, cosine)
    # --------------------
    optimizer = AdamConfig(
        learning_rate=0.002,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        warmup=0.01,
        min_lr_ratio=0.0,
        epsilon=1e-08,
        lr_schedule="cosine",
        max_grad_norm=1,
    )

    # --------------------
    # Multi-document P(z) callback: per-doc histograms + combined histogram
    # --------------------
    pz_md_cfg = PzMultiDocConfig(
        chunk_size=100,
        prompt_tokens=50,
        cursor_inc_tokens=5,
        eval_batch_size=128,
        verbose=True,
        max_docs=20,
    )

    return TrainLmConfig(
        data=data,
        trainer=trainer,
        model=model,
        optimizer=optimizer,
        data_seed=1,
        z_loss_weight=0,
        log_entropy=False,
        pz_multi_doc=pz_md_cfg,
        pz_multi_doc_steps=999,
    )


if __name__ == "__main__":
    cfg = build_config()

    # Launch via Marin on a single-slice v5p-64 in us-central2
    pod = TpuPodConfig(tpu_type="v5p-64", slice_count=1)

    step = ExecutorStep(
        name="memorize/splice_comma_600m_multi_40k_tau07_central2",
        fn=run_levanter_train_lm,
        config=TrainLmOnPodConfig(
            train_config=cfg,
            resources=pod,
            output_path=this_output_path(),
        ),
        pip_dependency_groups=["tokenize_train"],
    )

    executor_main(
        steps=[step],
        description="splice_comma_600m_multi_40k_tau07 (v5p-64, central2)",
    )


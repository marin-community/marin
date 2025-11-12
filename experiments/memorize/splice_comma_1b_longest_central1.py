"""1B model version of splice_comma_600m_longest_central1.py.

This single-file experiment trains a ~1B Llama-3.2 model on the longest spliced document
from Wikimedia (Common Pile) with coverage-balanced placement, with P(z) single-doc eval
callbacks enabled at the specified intervals.

Differences from 600M version:
- Uses llama_3_2_1b config (1B parameters) from experiments.llama
- Reduced training steps to 1000 (vs 15000)
- Uses v5p-64 TPU in us-central1

Usage (Executor):
  uv run python experiments/memorize/splice_comma_1b_longest_central1.py
"""

from __future__ import annotations

from datetime import timedelta

import jmp

from levanter.data.splice_dataset import SpliceSingleDocumentLMConfig
from levanter.data.text import (
    LMMixtureDatasetConfig,
    UrlDatasetSourceConfig,
    TextLmDatasetFormat,
)
from levanter.eval_pz_single_doc import PzSingleDocConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.trainer import TrainerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import RayConfig
from levanter.store.cache import CacheOptions

from experiments.llama import llama_3_2_1b
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.resources import TpuPodConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm


def build_config() -> TrainLmConfig:
    # --------------------
    # Data (splice longest doc from Wikimedia with coverage-balanced placement)
    # --------------------
    # Base mixture includes only the dataset we will draw the document from.
    wikimedia = UrlDatasetSourceConfig(
        cache_dir="gs://marin-us-central1/tokenized/common_pile/wikimedia-53a667",
        train_urls=[
            "gs://marin-us-central1/raw/common_pile/wikimedia_filtered-0641bb8",
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

    # Build the splice config itself. Important: enable shuffle_per_epoch at the splice layer
    # to stream indefinitely (rather than ending after one pass of splice pairs).
    # This config uses coverage_balanced mode for better coverage of the document.
    data = SpliceSingleDocumentLMConfig(
        tokenizer="meta-llama/Meta-Llama-3.1-8B",
        base=base_mixture,
        shuffle_per_epoch=True,
        dataset_name="common_pile/wikimedia",
        # Selection policy - automatically select the longest document in the dataset
        doc_index=13915895,  # Hardcoded to longest document (discovered via doc_select_mode="longest")
        min_doc_length=None,  # No minimum constraint - find truly longest
        max_doc_length=None,
        doc_select_mode="longest",
        # Balanced mode: use a constant content length per example and coverage-balanced placement
        content_length=4096,
        content_stride=1,
        offset_stride=1,
        content_start_mode="coverage_balanced",
        min_copy_len=128,
        alpha=0.8,
    )

    # --------------------
    # Model (~1B Llama-3.2, imported from experiments.llama)
    # --------------------
    model = llama_3_2_1b

    # --------------------
    # Trainer (reduced steps to 1000)
    # --------------------
    trainer = TrainerConfig(
        seed=0,
        ray=RayConfig(auto_start_cluster=False, start_workers=False),
        tracker=WandbConfig(
            project="marin",
            tags=["memorize", "splice", "1b", "wikimedia", "central1", "balanced", "longest"],
            resume="allow",
            save_code=True,
            name="llama_1b_splice_wikimedia_longest_doc_balanced_central1",
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        model_axis_size=1,
        per_device_parallelism=-1,
        train_batch_size=256,
        num_train_steps=1000,  # Reduced from 15000
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
    # Optimizer (Adam, cosine; exact params from original)
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
    # Single-document P(z) callback (exact settings from original)
    # --------------------
    pz_cfg = PzSingleDocConfig(
        chunk_size=100,
        prompt_tokens=50,
        cursor_inc_tokens=5,
        eval_batch_size=128,
        verbose=True,
        # Write under this step's output path, in a stable subdir
        gcp_prefix=this_output_path("pz_single_doc"),
    )

    return TrainLmConfig(
        data=data,
        trainer=trainer,
        model=model,
        optimizer=optimizer,
        data_seed=1,
        z_loss_weight=0,
        log_entropy=False,
        pz_single_doc=pz_cfg,
        pz_single_doc_steps=250,
    )


if __name__ == "__main__":
    cfg = build_config()

    # Launch via Marin on a single-slice v5p-64 in us-central1
    pod = TpuPodConfig(tpu_type="v5p-64", slice_count=1)

    step = ExecutorStep(
        name="memorize/splice_comma_1b_longest_balanced_central1",
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
        description="splice_comma_1b_longest_balanced (1B model, 1000 steps, v5p-64, central1)",
    )

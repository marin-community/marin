# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Unified image-text model pre-training experiment.

Trains a Qwen3-architecture model from scratch on a mixture of text-only data
(Nemotron-CC) and pre-tokenized multimodal data (dual-ordering image-caption
sequences). Uses an extended Llama3 tokenizer that adds TokLIP visual tokens
to the base vocabulary.

The multimodal cache must be generated first via vlm_tokenize_captions.py.
The extended tokenizer must be created first via create_unified_tokenizer().

Usage:
    # 1. Create the extended tokenizer (once)
    uv run python -c "from experiments.unified.unified_pretrain import create_unified_tokenizer; \
        create_unified_tokenizer()"

    # 2. Generate multimodal cache (see vlm_tokenize_captions.py)

    # 3. Run training (WANDB_API_KEY and TPU_TYPE are passed as env vars)
    uv run python -m marin.run.ray_run \
        --cluster infra/marin-us-central2.yaml \
        -e WANDB_API_KEY ${WANDB_API_KEY} \
        -e TPU_TYPE v4-64 \
        -- python experiments/unified/unified_pretrain_demo.py
"""

import dataclasses
import logging
import os
import tempfile

from levanter.data.text import DatasetComponent, LmDataConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat

from experiments.defaults import default_train, default_validation_sets
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets import NEMOTRON_WEIGHTS, tokenize_nemotron
from experiments.qwen3 import qwen3_0_6b, qwen3_1_7b, qwen3_4b
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution import executor_main
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

logger = logging.getLogger(__name__)

# --- Constants ---

LLAMA3_GCS_TOKENIZER = "gs://marin-us-central1/tokenizers/llama-3.1-8b"
LLAMA3_VOCAB_SIZE = 128_256
TOKLIP_CODEBOOK_SIZE = 16_384
UNIFIED_VOCAB_SIZE = LLAMA3_VOCAB_SIZE + TOKLIP_CODEBOOK_SIZE  # 144,640

# Reserved Llama3 special token IDs repurposed for vision sentinels
VISION_START_ID = 128_004  # <|reserved_special_token_2|> → <|vision_start|>
VISION_END_ID = 128_005  # <|reserved_special_token_3|> → <|vision_end|>

UNIFIED_TOKENIZER_PATH = "gs://marin-us-central1/tokenizers/llama3-unified-144k"
UNIFIED_CACHE_PATH = "gs://marin-vlm/hf_85m_levanter_cache_test"
UNIFIED_EVAL_CACHE_PATH = "gs://marin-vlm/unified_eval_cache"

# Eval benchmark categories (used for tagging in eval metrics)
UNDERSTANDING_BENCHMARKS = {"textvqa", "chartqa", "ai2d", "mmmu"}
GENERATION_BENCHMARKS = {"cifar10_small", "cifar10", "imagenet_small", "imagenet"}
MC_BENCHMARKS = ["ai2d", "mmmu"]
VLM_EVAL_BENCHMARKS = ["ai2d", "mmmu", "textvqa", "chartqa"]


def _enable_single_vlm_eval_hlo_dump() -> None:
    """Append one-shot VLM-eval-focused HLO dump flags to XLA_FLAGS."""
    required_flags = [
        "--xla_dump_to=/tmp/xla_dumps_vlm_once",
        "--xla_dump_hlo_as_text",
        "--xla_dump_hlo_pass_re=.*(before_optimizations|after_optimizations).*",
        "--xla_dump_hlo_module_re=.*vlm_eval_loglikelihood.*",
    ]
    existing = os.environ.get("XLA_FLAGS", "").strip()
    merged = existing
    for flag in required_flags:
        if flag not in merged:
            merged = f"{merged} {flag}".strip()
    os.environ["XLA_FLAGS"] = merged
    logger.warning("Enabled one-shot VLM eval HLO dump via XLA_FLAGS=%s", merged)


# --- Extended Tokenizer Creation ---


def create_unified_tokenizer(output_path: str = UNIFIED_TOKENIZER_PATH) -> str:
    """Create Llama3 tokenizer extended with vision special tokens and TokLIP visual tokens.

    Repurposes reserved_special_token_2/3 as <|vision_start|>/<|vision_end|>,
    then appends 16,384 visual placeholder tokens (<|v_0|> ... <|v_16383|>).

    The resulting tokenizer has vocab size 144,640 = 128,256 (Llama3) + 16,384 (TokLIP-L).
    """
    import fsspec

    from experiments.create_marin_tokenizer import _inject_special_tokens

    from levanter.compat.hf_checkpoints import load_tokenizer

    tok = load_tokenizer(LLAMA3_GCS_TOKENIZER)
    assert len(tok) == LLAMA3_VOCAB_SIZE, f"Expected {LLAMA3_VOCAB_SIZE}, got {len(tok)}"

    # Rename reserved tokens for vision sentinels
    tok = _inject_special_tokens(
        tok,
        {
            VISION_START_ID: "<|vision_start|>",
            VISION_END_ID: "<|vision_end|>",
        },
    )

    # Add visual placeholder tokens (IDs 128,256 ... 144,639)
    visual_tokens = [f"<|v_{i}|>" for i in range(TOKLIP_CODEBOOK_SIZE)]
    tok.add_tokens(visual_tokens, special_tokens=True)
    assert len(tok) == UNIFIED_VOCAB_SIZE, f"Expected {UNIFIED_VOCAB_SIZE}, got {len(tok)}"

    # Save to GCS
    with tempfile.TemporaryDirectory() as tmp:
        tok.save_pretrained(tmp)
        fs = fsspec.filesystem("gs")
        fs.put(tmp, output_path, recursive=True)

    logger.info("Saved unified tokenizer (%d tokens) to %s", len(tok), output_path)
    return output_path


# --- Data Config ---


def unified_data_config(
    text_weight: float = 0.7,
    multimodal_weight: float = 0.3,
    multimodal_cache_path: str = UNIFIED_CACHE_PATH,
    eval_benchmarks: list[str] | None = None,
    eval_cache_path: str = UNIFIED_EVAL_CACHE_PATH,
) -> LmDataConfig:
    """Build data mixture with Nemotron text + multimodal cache for unified model training.

    Args:
        text_weight: Scaling factor applied to each Nemotron split's weight (controls r1).
        multimodal_weight: Weight for the multimodal caption data component.
        multimodal_cache_path: GCS path to the pre-built multimodal Levanter cache.
        eval_benchmarks: List of eval benchmark names to include as validation-only
            components (train_weight=0.0). Set to None to disable eval.
        eval_cache_path: GCS path to eval benchmark Levanter caches.
    """
    # Text: use existing Llama3-tokenized Nemotron caches (no re-tokenization needed)
    nemotron_steps = tokenize_nemotron()
    text_components = {
        name: step_to_lm_mixture_component(step, include_raw_paths=True) for name, step in nemotron_steps.items()
    }
    text_weights = {k: text_weight * v for k, v in NEMOTRON_WEIGHTS.items()}

    # Multimodal: pre-built cache with per-token loss weights.
    # pack=True to avoid wasting compute padding short sequences (~600 tokens) to 4096.
    # NOTE: cache_dir should NOT include the split suffix — Levanter's build_caches()
    # appends /{split} automatically (e.g., cache_dir/train, cache_dir/validation).
    multimodal_component = DatasetComponent(
        cache_dir=multimodal_cache_path,
        format=PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
        ),
        pack=True,
    )

    components = {**text_components, "multimodal_captions": multimodal_component}
    weights = {**text_weights, "multimodal_captions": multimodal_weight}

    # Multimodal validation: separate understanding and generation val loss.
    # These are produced by vlm_tokenize_captions.py with val_fraction > 0.
    prebuilt_format = PrebuiltLmDatasetFormat(input_ids_key="input_ids", loss_weights_key="loss_weights")
    components["val_understanding"] = DatasetComponent(
        cache_dir=f"{multimodal_cache_path}/val_understanding",
        format=prebuilt_format,
        pack=True,
        tags=["multimodal", "understanding"],
    )
    weights["val_understanding"] = 0.0

    components["val_generation"] = DatasetComponent(
        cache_dir=f"{multimodal_cache_path}/val_generation",
        format=prebuilt_format,
        pack=True,
        tags=["multimodal", "generation"],
    )
    weights["val_generation"] = 0.0

    # Eval benchmarks: weight 0.0 → eval-only (not used in training data).
    # Levanter's validation_sets() loads these for periodic eval during training.
    if eval_benchmarks is not None:
        for bench in eval_benchmarks:
            tags = ["eval"]
            if bench in UNDERSTANDING_BENCHMARKS:
                tags.append("understanding")
            elif bench in GENERATION_BENCHMARKS:
                tags.append("generation")

            components[f"eval_{bench}"] = DatasetComponent(
                cache_dir=f"{eval_cache_path}/{bench}",
                format=PrebuiltLmDatasetFormat(
                    input_ids_key="input_ids",
                    loss_weights_key="loss_weights",
                ),
                pack=True,
                tags=tags,
            )
            weights[f"eval_{bench}"] = 0.0

    data_config = LmDataConfig(
        tokenizer=UNIFIED_TOKENIZER_PATH,
        components=components,
        train_weights=weights,
        shuffle=True,
        permutation_type="feistel",
        block_cross_document_attention=True,
    )

    # Text-only validation sets (Paloma, uncheatable_eval) use the base Llama3
    # tokenizer since the unified tokenizer only adds visual tokens — text
    # tokenization is identical, so we reuse existing Llama3-tokenized caches.
    validation_sets = default_validation_sets(tokenizer=llama3_tokenizer)
    data_config = add_validation_sets_to_mixture(data_config, validation_sets)

    return data_config


# --- Train Configs ---

# 1 epoch ≈ 1,582,102 records / 256 batch_size ≈ 6,180 steps
DEMO_TRAIN_STEPS = 6_180
TPU_TYPE = os.environ.get("TPU_TYPE", "v4-64")
EXP_NAME = os.environ.get("EXP_NAME", "")


def _demo_train_config(learning_rate: float = 3e-4) -> SimpleTrainConfig:
    return SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(TPU_TYPE, slice_count=1),
        train_batch_size=256,
        num_train_steps=DEMO_TRAIN_STEPS,
        learning_rate=learning_rate,
        warmup=0.01,
        lr_schedule="cosine",
        weight_decay=0.1,
        max_grad_norm=1.0,
        per_device_parallelism=4,
        steps_per_eval=500,
        vlm_mc_eval_benchmarks=VLM_EVAL_BENCHMARKS,
        vlm_mc_eval_steps=500,
    )


# --- Experiment Step Factories ---
# These are functions (not module-level calls) because default_train() loads
# the tokenizer eagerly to compute parameter counts. The unified tokenizer
# lives on GCS and must be created first via create_unified_tokenizer().


DEFAULT_EVAL_BENCHMARKS = [
    "textvqa",
    "chartqa",
    "ai2d",
    "mmmu",
    "cifar10_small",
    "imagenet_small",
]


def make_unified_0_6b(
    text_weight: float = 0.7,
    multimodal_weight: float = 0.3,
    eval_benchmarks: list[str] | None = DEFAULT_EVAL_BENCHMARKS,
):
    step = default_train(
        name=EXP_NAME or "unified-qwen3-0.6b-demo",
        tokenized=unified_data_config(
            text_weight=text_weight,
            multimodal_weight=multimodal_weight,
            eval_benchmarks=eval_benchmarks,
        ),
        model_config=qwen3_0_6b,
        train_config=_demo_train_config(learning_rate=3e-4),
        tags=["unified", "scaling", "qwen3", "0.6b", "demo"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    # Multimodal caches live in gs://marin-vlm (us multi-region); allow cross-region read
    return dataclasses.replace(step, config=dataclasses.replace(step.config, allow_out_of_region=("data.components",)))


def make_unified_1_7b(
    text_weight: float = 0.7,
    multimodal_weight: float = 0.3,
    eval_benchmarks: list[str] | None = DEFAULT_EVAL_BENCHMARKS,
):
    step = default_train(
        name=EXP_NAME or "unified-qwen3-1.7b-demo",
        tokenized=unified_data_config(
            text_weight=text_weight,
            multimodal_weight=multimodal_weight,
            eval_benchmarks=eval_benchmarks,
        ),
        model_config=qwen3_1_7b,
        train_config=_demo_train_config(learning_rate=3e-4),
        tags=["unified", "scaling", "qwen3", "1.7b", "demo"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    return dataclasses.replace(step, config=dataclasses.replace(step.config, allow_out_of_region=("data.components",)))


def make_unified_4b(
    text_weight: float = 0.7,
    multimodal_weight: float = 0.3,
    eval_benchmarks: list[str] | None = DEFAULT_EVAL_BENCHMARKS,
):
    step = default_train(
        name=EXP_NAME or "unified-qwen3-4b-demo",
        tokenized=unified_data_config(
            text_weight=text_weight,
            multimodal_weight=multimodal_weight,
            eval_benchmarks=eval_benchmarks,
        ),
        model_config=qwen3_4b,
        train_config=_demo_train_config(learning_rate=1.5e-4),
        tags=["unified", "scaling", "qwen3", "4b", "demo"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    return dataclasses.replace(step, config=dataclasses.replace(step.config, allow_out_of_region=("data.components",)))


if __name__ == "__main__":
    _enable_single_vlm_eval_hlo_dump()
    steps = [make_unified_0_6b()]
    executor_main(
        steps,
        description="Unified image-text model pre-training with Qwen3 architecture",
    )

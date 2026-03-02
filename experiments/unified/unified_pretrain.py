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

    # 3. Run training
    uv run lib/marin/src/marin/run/ray_run.py -- python experiments/unified/unified_pretrain.py
"""

import dataclasses
import logging
import tempfile
from collections.abc import Callable

import numpy as np
from levanter.data.text import DatasetComponent, LmDataConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat

from experiments.defaults import default_train, default_validation_sets
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets import tokenize_nemotron
from experiments.qwen3 import qwen3_0_6b, qwen3_1_7b, qwen3_4b
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution import executor_main
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

logger = logging.getLogger(__name__)

# --- Constants ---

LLAMA3_GCS_TOKENIZER = "gs://marin-us-central2/tokenizers/llama-3.1-8b"
LLAMA3_VOCAB_SIZE = 128_256
TOKLIP_CODEBOOK_SIZE = 16_384
UNIFIED_VOCAB_SIZE = LLAMA3_VOCAB_SIZE + TOKLIP_CODEBOOK_SIZE  # 144,640

# Reserved Llama3 special token IDs repurposed for vision sentinels
VISION_START_ID = 128_004  # <|reserved_special_token_2|> → <|vision_start|>
VISION_END_ID = 128_005  # <|reserved_special_token_3|> → <|vision_end|>

UNIFIED_TOKENIZER_PATH = "gs://marin-us-central2/tokenizers/llama3-unified-144k"
UNIFIED_CACHE_PATH = "gs://marin-vlm/hf_85m_levanter_cache_v2"
UNIFIED_EVAL_CACHE_PATH = "gs://marin-vlm/unified_eval_cache"
TEXT_EVAL_CACHE_PATH = "gs://marin-vlm/text_eval_cache"

# Eval benchmark categories (used for tagging in eval metrics)
UNDERSTANDING_BENCHMARKS = {"textvqa", "chartqa", "ai2d", "mmmu", "seedbench_image"}
GENERATION_BENCHMARKS = {"cifar10_small", "cifar10", "imagenet_small", "imagenet"}
MC_BENCHMARKS = ["ai2d", "mmmu", "seedbench_image"]
VLM_EVAL_BENCHMARKS = ["ai2d", "mmmu", "seedbench_image", "textvqa", "chartqa"]


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


def _make_visual_weight_transform(new_w_visual: float) -> Callable[[np.ndarray], np.ndarray]:
    """Create a loss_weight_transform that replaces fractional weights with new_w_visual.

    In the preprocessing cache, weights are either w_visual (a value in (0, 1)) for the
    "secondary" modality or 1.0 for the "primary" modality. This transform replaces all
    fractional weights with the new value, allowing w_visual to be tuned at training time
    without re-running preprocessing.
    """

    def transform(weights: np.ndarray) -> np.ndarray:
        mask = (weights > 0) & (weights < 1.0)
        if not mask.any():
            return weights
        result = weights.copy()
        result[mask] = new_w_visual
        return result

    return transform


def _zero_fractional_transform(weights: np.ndarray) -> np.ndarray:
    """Zero out fractional weights, keeping only the primary modality (weight=1.0 tokens).

    In understanding caches this removes visual tokens; in generation caches this removes text tokens.
    """
    mask = (weights > 0) & (weights < 1.0)
    if not mask.any():
        return weights
    result = weights.copy()
    result[mask] = 0.0
    return result


def _only_fractional_transform(weights: np.ndarray) -> np.ndarray:
    """Keep only the secondary modality: fractional → 1.0, primary (1.0) → 0.

    In understanding caches this keeps only visual tokens; in generation caches only text tokens.
    """
    result = np.zeros_like(weights)
    mask = (weights > 0) & (weights < 1.0)
    result[mask] = 1.0
    return result


def unified_data_config(
    text_weight: float = 1.0,
    multimodal_weight: float = 1.0,
    multimodal_cache_path: str = UNIFIED_CACHE_PATH,
    eval_benchmarks: list[str] | None = None,
    eval_cache_path: str = UNIFIED_EVAL_CACHE_PATH,
    w_visual: float | None = 1.0,
    und_gen_ratio: float = 1.0,
    text_eval_benchmarks: list[str] | None = None,
    text_eval_cache_path: str = TEXT_EVAL_CACHE_PATH,
) -> LmDataConfig:
    """Build data mixture with Nemotron text + multimodal cache for unified model training.

    Args:
        text_weight: Scaling factor applied to each Nemotron split's weight (controls r1).
        multimodal_weight: Total weight for multimodal data, split between understanding
            and generation according to und_gen_ratio.
        multimodal_cache_path: GCS path to the pre-built multimodal Levanter cache.
        eval_benchmarks: List of VLM eval benchmark names to include as validation-only
            components (train_weight=0.0). Set to None to disable eval.
        eval_cache_path: GCS path to VLM eval benchmark Levanter caches.
        w_visual: Override for the visual token loss weight. When set, replaces
            the preprocessing-baked w_visual at data loading time. None uses the
            original preprocessing value unchanged.
        und_gen_ratio: Ratio of understanding to generation weight. E.g. 3.0 means
            understanding gets 3/(3+1) of multimodal_weight, generation gets 1/(3+1).
        text_eval_benchmarks: List of text eval benchmark names (e.g. hellaswag, mmlu)
            to include as validation-only components. Set to None to disable.
        text_eval_cache_path: GCS path to text eval benchmark Levanter caches.
    """
    # Text: only hq_actual from Nemotron-CC
    nemotron_steps = tokenize_nemotron()
    hq_key = "nemotron_cc/hq_actual"
    text_components = {hq_key: step_to_lm_mixture_component(nemotron_steps[hq_key], include_raw_paths=True)}
    text_weights = {hq_key: text_weight}

    # Multimodal: pre-built cache with per-token loss weights.
    # pack=True to avoid wasting compute padding short sequences (~600 tokens) to 4096.
    # NOTE: cache_dir should NOT include the split suffix — Levanter's build_caches()
    # appends /{split} automatically (e.g., cache_dir/train, cache_dir/validation).
    loss_weight_transform = _make_visual_weight_transform(w_visual) if w_visual is not None else None
    prebuilt_format = PrebuiltLmDatasetFormat(
        input_ids_key="input_ids",
        loss_weights_key="loss_weights",
        loss_weight_transform=loss_weight_transform,
    )
    und_weight = multimodal_weight * und_gen_ratio / (und_gen_ratio + 1)
    gen_weight = multimodal_weight / (und_gen_ratio + 1)

    components = {
        **text_components,
        "multimodal_understanding": DatasetComponent(
            cache_dir=f"{multimodal_cache_path}/understanding",
            format=prebuilt_format,
            pack=True,
            tags=["multimodal", "understanding"],
        ),
        "multimodal_generation": DatasetComponent(
            cache_dir=f"{multimodal_cache_path}/generation",
            format=prebuilt_format,
            pack=True,
            tags=["multimodal", "generation"],
        ),
    }
    weights = {
        **text_weights,
        "multimodal_understanding": und_weight,
        "multimodal_generation": gen_weight,
    }

    # Multimodal validation: separate understanding and generation val loss.
    components["val_understanding"] = DatasetComponent(
        cache_dir=f"{multimodal_cache_path}/understanding/val_understanding",
        format=prebuilt_format,
        pack=True,
        tags=["multimodal", "understanding"],
    )
    weights["val_understanding"] = 0.0

    components["val_generation"] = DatasetComponent(
        cache_dir=f"{multimodal_cache_path}/generation/val_generation",
        format=prebuilt_format,
        pack=True,
        tags=["multimodal", "generation"],
    )
    weights["val_generation"] = 0.0

    # Per-modality val loss breakdown (text-only and visual-only).
    # In understanding cache: fractional=visual, 1.0=text.
    # In generation cache: fractional=text, 1.0=visual.
    fmt_primary_only = PrebuiltLmDatasetFormat(
        input_ids_key="input_ids",
        loss_weights_key="loss_weights",
        loss_weight_transform=_zero_fractional_transform,
    )
    fmt_secondary_only = PrebuiltLmDatasetFormat(
        input_ids_key="input_ids",
        loss_weights_key="loss_weights",
        loss_weight_transform=_only_fractional_transform,
    )

    und_val_cache = f"{multimodal_cache_path}/understanding/val_understanding"
    gen_val_cache = f"{multimodal_cache_path}/generation/val_generation"

    # Understanding: primary=text, secondary=visual
    components["val_understanding_text_only"] = DatasetComponent(
        cache_dir=und_val_cache,
        format=fmt_primary_only,
        pack=True,
        tags=["multimodal", "understanding"],
    )
    components["val_understanding_visual_only"] = DatasetComponent(
        cache_dir=und_val_cache,
        format=fmt_secondary_only,
        pack=True,
        tags=["multimodal", "understanding"],
    )
    weights["val_understanding_text_only"] = 0.0
    weights["val_understanding_visual_only"] = 0.0

    # Generation: primary=visual, secondary=text
    components["val_generation_text_only"] = DatasetComponent(
        cache_dir=gen_val_cache,
        format=fmt_secondary_only,
        pack=True,
        tags=["multimodal", "generation"],
    )
    components["val_generation_visual_only"] = DatasetComponent(
        cache_dir=gen_val_cache,
        format=fmt_primary_only,
        pack=True,
        tags=["multimodal", "generation"],
    )
    weights["val_generation_text_only"] = 0.0
    weights["val_generation_visual_only"] = 0.0

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
                format=prebuilt_format,
                pack=True,
                tags=tags,
            )
            weights[f"eval_{bench}"] = 0.0

    # Text eval benchmarks: weight 0.0 → eval-only.
    # These are pure text (no visual tokens), so no loss_weight_transform needed.
    if text_eval_benchmarks is not None:
        text_prebuilt_format = PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
        )
        for bench in text_eval_benchmarks:
            components[f"text_eval_{bench}"] = DatasetComponent(
                cache_dir=f"{text_eval_cache_path}/{bench}",
                format=text_prebuilt_format,
                pack=True,
                tags=["eval", "text"],
            )
            weights[f"text_eval_{bench}"] = 0.0

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

unified_0_6b_train = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v4-64", slice_count=1),
    train_batch_size=256,
    num_train_steps=50_000,
    learning_rate=3e-4,
    warmup=0.01,
    lr_schedule="cosine",
    weight_decay=0.1,
    max_grad_norm=1.0,
)

unified_1_7b_train = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v4-128", slice_count=1),
    train_batch_size=256,
    num_train_steps=50_000,
    learning_rate=3e-4,
    warmup=0.01,
    lr_schedule="cosine",
    weight_decay=0.1,
    max_grad_norm=1.0,
)

unified_4b_train = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v4-256", slice_count=1),
    train_batch_size=256,
    num_train_steps=50_000,
    learning_rate=1.5e-4,
    warmup=0.01,
    lr_schedule="cosine",
    weight_decay=0.1,
    max_grad_norm=1.0,
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
    "seedbench_image",
    "cifar10_small",
    "imagenet_small",
]

DEFAULT_TEXT_EVAL_BENCHMARKS = [
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "mmlu",
]


def make_unified_0_6b(
    text_weight: float = 1.0,
    multimodal_weight: float = 1.0,
    eval_benchmarks: list[str] | None = DEFAULT_EVAL_BENCHMARKS,
    w_visual: float | None = None,
    und_gen_ratio: float = 1.0,
    text_eval_benchmarks: list[str] | None = DEFAULT_TEXT_EVAL_BENCHMARKS,
):
    step = default_train(
        name="unified-qwen3-0.6b",
        tokenized=unified_data_config(
            text_weight=text_weight,
            multimodal_weight=multimodal_weight,
            eval_benchmarks=eval_benchmarks,
            w_visual=w_visual,
            und_gen_ratio=und_gen_ratio,
            text_eval_benchmarks=text_eval_benchmarks,
        ),
        model_config=qwen3_0_6b,
        train_config=unified_0_6b_train,
        tags=["unified", "scaling", "qwen3", "0.6b"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    # Multimodal caches live in gs://marin-vlm (us multi-region); allow cross-region read
    return dataclasses.replace(step, config=dataclasses.replace(step.config, allow_out_of_region=("data.components",)))


def make_unified_1_7b(
    text_weight: float = 1.0,
    multimodal_weight: float = 1.0,
    eval_benchmarks: list[str] | None = DEFAULT_EVAL_BENCHMARKS,
    w_visual: float | None = None,
    und_gen_ratio: float = 1.0,
    text_eval_benchmarks: list[str] | None = DEFAULT_TEXT_EVAL_BENCHMARKS,
):
    step = default_train(
        name="unified-qwen3-1.7b",
        tokenized=unified_data_config(
            text_weight=text_weight,
            multimodal_weight=multimodal_weight,
            eval_benchmarks=eval_benchmarks,
            w_visual=w_visual,
            und_gen_ratio=und_gen_ratio,
            text_eval_benchmarks=text_eval_benchmarks,
        ),
        model_config=qwen3_1_7b,
        train_config=unified_1_7b_train,
        tags=["unified", "scaling", "qwen3", "1.7b"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    step.config.allow_out_of_region = ("data.components",)
    return step


def make_unified_4b(
    text_weight: float = 1.0,
    multimodal_weight: float = 1.0,
    eval_benchmarks: list[str] | None = DEFAULT_EVAL_BENCHMARKS,
    w_visual: float | None = None,
    und_gen_ratio: float = 1.0,
    text_eval_benchmarks: list[str] | None = DEFAULT_TEXT_EVAL_BENCHMARKS,
):
    step = default_train(
        name="unified-qwen3-4b",
        tokenized=unified_data_config(
            text_weight=text_weight,
            multimodal_weight=multimodal_weight,
            eval_benchmarks=eval_benchmarks,
            w_visual=w_visual,
            und_gen_ratio=und_gen_ratio,
            text_eval_benchmarks=text_eval_benchmarks,
        ),
        model_config=qwen3_4b,
        train_config=unified_4b_train,
        tags=["unified", "scaling", "qwen3", "4b"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    step.config.allow_out_of_region = ("data.components",)
    return step


if __name__ == "__main__":
    steps = [make_unified_0_6b(), make_unified_1_7b(), make_unified_4b()]
    executor_main(
        steps,
        description="Unified image-text model pre-training with Qwen3 architecture",
    )

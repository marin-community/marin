# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
from collections.abc import Callable

import numpy as np

from levanter.data.text import DatasetComponent, LmDataConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from levanter.optim import MuonConfig

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

LLAMA3_GCS_TOKENIZER = "gs://marin-eu-west4/tokenizers/llama-3.1-8b"
LLAMA3_VOCAB_SIZE = 128_256
TOKLIP_CODEBOOK_SIZE = 16_384
UNIFIED_VOCAB_SIZE = LLAMA3_VOCAB_SIZE + TOKLIP_CODEBOOK_SIZE  # 144,640

# Reserved Llama3 special token IDs repurposed for vision sentinels
VISION_START_ID = 128_004  # <|reserved_special_token_2|> → <|vision_start|>
VISION_END_ID = 128_005  # <|reserved_special_token_3|> → <|vision_end|>
ENDOFTEXT_ID = 128_001  # <|end_of_text|> (Llama3 EOS)
VISUAL_TOKEN_OFFSET = 128_256  # Llama3 vocab size; TokLIP index c → unified ID c + 128256

UNIFIED_TOKENIZER_PATH = "gs://marin-eu-west4/tokenizers/llama3-unified-144k"
UNIFIED_CACHE_PATH = "gs://marin-vlm-eu/hf_85m_levanter_cache_v2"
VISUAL_ONLY_CACHE_PATH = "gs://marin-vlm-eu/hf_85m_levanter_cache_v2/visual_only"
UNIFIED_EVAL_CACHE_PATH = "gs://marin-vlm-eu/unified_eval_cache"
TEXT_EVAL_CACHE_PATH = "gs://marin-vlm-eu/text_eval_cache"

# Eval benchmark categories (used for tagging in eval metrics)
UNDERSTANDING_BENCHMARKS = {"textvqa", "chartqa", "ai2d", "mmmu"}
GENERATION_BENCHMARKS = {"cifar10_small", "cifar10", "imagenet_small", "imagenet"}
MC_BENCHMARKS = ["ai2d", "mmmu"]
VLM_EVAL_BENCHMARKS = ["ai2d", "mmmu", "textvqa", "chartqa"]


_VLM_HLO_DUMP_XLA_FLAGS = [
    "--xla_dump_to=/tmp/xla_dumps_vlm_once",
    "--xla_dump_hlo_as_text",
    "--xla_dump_hlo_pass_re=.*(before_optimizations|after_optimizations).*",
    "--xla_dump_hlo_module_re=.*vlm_eval_loglikelihood.*",
]

# --- Train Configs ---

# 1 epoch ≈ 1,582,102 records / 256 batch_size ≈ 6,180 steps
TEXT_STEPS = int(os.environ.get("TEXT_STEPS", "5000"))
UND_STEPS = int(os.environ.get("UND_STEPS", "5000"))
GEN_STEPS = int(os.environ.get("GEN_STEPS", "5000"))
VIS_STEPS = int(os.environ.get("VIS_STEPS", "0"))
DEMO_TRAIN_STEPS = TEXT_STEPS + UND_STEPS + GEN_STEPS + VIS_STEPS
TPU_TYPE = os.environ.get("TPU_TYPE", "v4-64")
EXP_NAME = os.environ.get("EXP_NAME", "")
OPTIMIZER = os.environ.get("OPTIMIZER", "muon")  # "muon" or "adam"
MUON_LR = float(os.environ.get("MUON_LR", "0.004"))
ADAM_LR = float(os.environ.get("ADAM_LR", "0.0012"))
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "cosine")
W_VISUAL = float(os.environ.get("W_VISUAL", "1.0"))
ABLATION_MODE = os.environ.get("ABLATION_MODE", "")

VALID_ABLATION_MODES = {
    "",
    "mask_und_visual",
    "mask_und_text",
    "mask_gen_visual",
    "mask_gen_text",
    "isolate_und_attn",
    "isolate_gen_attn",
}


def _merge_xla_flags(existing: str, required_flags: list[str]) -> str:
    merged = existing.strip()
    for flag in required_flags:
        if flag not in merged:
            merged = f"{merged} {flag}".strip()
    return merged


def _with_vlm_hlo_dump_env(step):
    """Attach one-shot VLM HLO dump flags to the remote worker runtime env."""
    env_vars = dict(step.env_vars or {})
    env_vars["XLA_FLAGS"] = _merge_xla_flags(env_vars.get("XLA_FLAGS", ""), _VLM_HLO_DUMP_XLA_FLAGS)
    logger.warning("Configured step env XLA_FLAGS for one-shot VLM HLO dump: %s", env_vars["XLA_FLAGS"])
    return dataclasses.replace(step, env_vars=env_vars)


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


def _swap_primary_secondary_transform(weights: np.ndarray) -> np.ndarray:
    """Swap fractional and 1.0 weights to fix generation cache convention.

    The generation cache was built with w_visual applied to text tokens instead of
    visual tokens. This transform swaps fractional↔1.0 so that visual tokens get
    the fractional weight and text tokens get 1.0, matching the understanding convention.
    """
    result = weights.copy()
    fractional_mask = (weights > 0) & (weights < 1.0)
    primary_mask = weights == 1.0
    if fractional_mask.any():
        frac_val = weights[fractional_mask][0]
        result[fractional_mask] = 1.0
        result[primary_mask] = frac_val
    return result


def _compose_transforms(
    *transforms: Callable[[np.ndarray], np.ndarray] | None,
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Chain multiple optional loss_weight_transforms, skipping Nones."""
    active = [t for t in transforms if t is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def composed(weights: np.ndarray) -> np.ndarray:
        for t in active:
            weights = t(weights)
        return weights

    return composed


def _replace_visual_with_eos(input_ids: np.ndarray) -> np.ndarray:
    """Replace visual tokens with EOS so block_cross_document_attention prevents
    real tokens from attending to them.

    Each replaced position becomes its own segment boundary, so the remaining
    text tokens (in a later segment) cannot attend to any visual position.
    """
    result = input_ids.copy()
    visual_mask = (result >= VISUAL_TOKEN_OFFSET) | (result == VISION_START_ID) | (result == VISION_END_ID)
    result[visual_mask] = ENDOFTEXT_ID
    return result


def _replace_text_with_eos(input_ids: np.ndarray) -> np.ndarray:
    """Replace text tokens with EOS so block_cross_document_attention prevents
    real tokens from attending to them.

    Each replaced position becomes its own segment boundary, so the remaining
    visual tokens (in a later segment) cannot attend to any text position.
    """
    result = input_ids.copy()
    text_mask = (
        (result > 0)
        & (result < VISUAL_TOKEN_OFFSET)
        & (result != VISION_START_ID)
        & (result != VISION_END_ID)
        & (result != ENDOFTEXT_ID)
    )
    result[text_mask] = ENDOFTEXT_ID
    return result


def _make_modality_segment_ids(input_ids: np.ndarray) -> np.ndarray:
    """Return per-token modality offset: 0 for text tokens, 1 for visual tokens.

    Visual tokens are identified by ID >= LLAMA3_VOCAB_SIZE (128256) or being a
    vision sentinel token (vision_start / vision_end). This approach is robust to
    chunk boundaries in concatenation-chunking since it doesn't depend on matching
    start/end pairs.
    """
    is_visual = (input_ids >= LLAMA3_VOCAB_SIZE).astype(np.int32)
    is_sentinel = ((input_ids == VISION_START_ID) | (input_ids == VISION_END_ID)).astype(np.int32)
    return np.clip(is_visual + is_sentinel, 0, 1)


def unified_data_config(
    multimodal_cache_path: str = UNIFIED_CACHE_PATH,
    visual_only_cache_path: str = VISUAL_ONLY_CACHE_PATH,
    eval_benchmarks: list[str] | None = None,
    eval_cache_path: str = UNIFIED_EVAL_CACHE_PATH,
    w_visual: float | None = 1.0,
    text_eval_benchmarks: list[str] | None = None,
    text_eval_cache_path: str = TEXT_EVAL_CACHE_PATH,
    ablation_mode: str = "",
) -> LmDataConfig:
    """Build data mixture with Nemotron text + multimodal cache for unified model training.

    Mixing weights are derived from the global TEXT_STEPS, UND_STEPS, GEN_STEPS
    environment variables.

    Args:
        multimodal_cache_path: GCS path to the pre-built multimodal Levanter cache.
        eval_benchmarks: List of VLM eval benchmark names to include as validation-only
            components (train_weight=0.0). Set to None to disable eval.
        eval_cache_path: GCS path to VLM eval benchmark Levanter caches.
        w_visual: Override for the visual token loss weight. When set, replaces
            the preprocessing-baked w_visual at data loading time. None uses the
            original preprocessing value unchanged.
        text_eval_benchmarks: List of text eval benchmark names (e.g. hellaswag, mmlu)
            to include as validation-only components. Set to None to disable.
        text_eval_cache_path: GCS path to text eval benchmark Levanter caches.
        ablation_mode: Ablation mode for modality masking or attention isolation.
            One of: "", "mask_und_visual", "mask_und_text", "mask_gen_visual",
            "mask_gen_text", "isolate_und_attn", "isolate_gen_attn".
    """
    if ablation_mode and ablation_mode not in VALID_ABLATION_MODES:
        raise ValueError(f"Unknown ablation_mode={ablation_mode!r}. Valid: {VALID_ABLATION_MODES}")

    # Mixing weights derived from step counts: each component's weight is
    # proportional to the number of steps allocated to it.
    total_steps = TEXT_STEPS + UND_STEPS + GEN_STEPS + VIS_STEPS
    text_w = TEXT_STEPS / total_steps if total_steps > 0 else 1.0
    und_w = UND_STEPS / total_steps if total_steps > 0 else 0.0
    gen_w = GEN_STEPS / total_steps if total_steps > 0 else 0.0
    vis_w = VIS_STEPS / total_steps if total_steps > 0 else 0.0

    # Text: only hq_actual from Nemotron-CC
    nemotron_steps = tokenize_nemotron()
    hq_key = "nemotron_cc/hq_actual"
    text_components = {hq_key: step_to_lm_mixture_component(nemotron_steps[hq_key], include_raw_paths=True)}
    text_weights = {hq_key: text_w}

    # Multimodal: pre-built cache with per-token loss weights.
    # pack=True to avoid wasting compute padding short sequences (~600 tokens) to 4096.
    # NOTE: cache_dir should NOT include the split suffix — Levanter's build_caches()
    # appends /{split} automatically (e.g., cache_dir/train, cache_dir/validation).
    w_visual_transform = _make_visual_weight_transform(w_visual) if w_visual is not None else None

    # Determine per-component ablation transforms.
    # Understanding cache: fractional = visual, 1.0 = text (correct convention).
    # Generation cache: fractional = text, 1.0 = visual (bug — swap fixes this).
    und_ablation: Callable[[np.ndarray], np.ndarray] | None = None
    gen_ablation: Callable[[np.ndarray], np.ndarray] | None = None
    und_seg_transform: Callable[[np.ndarray], np.ndarray] | None = None
    gen_seg_transform: Callable[[np.ndarray], np.ndarray] | None = None

    if ablation_mode == "mask_und_visual":
        und_ablation = _zero_fractional_transform
    elif ablation_mode == "mask_und_text":
        und_ablation = _only_fractional_transform
    elif ablation_mode == "mask_gen_visual":
        gen_ablation = _zero_fractional_transform
    elif ablation_mode == "mask_gen_text":
        gen_ablation = _only_fractional_transform
    elif ablation_mode == "isolate_und_attn":
        und_seg_transform = _make_modality_segment_ids
    elif ablation_mode == "isolate_gen_attn":
        gen_seg_transform = _make_modality_segment_ids

    # Compose transforms: ablation runs before w_visual (ablation needs to distinguish
    # fractional vs 1.0; w_visual replaces fractional values, destroying that signal).
    # Generation additionally prepends a swap to fix the cache convention bug.
    und_format = PrebuiltLmDatasetFormat(
        input_ids_key="input_ids",
        loss_weights_key="loss_weights",
        loss_weight_transform=_compose_transforms(und_ablation, w_visual_transform),
        segment_ids_transform=und_seg_transform,
    )
    gen_format = PrebuiltLmDatasetFormat(
        input_ids_key="input_ids",
        loss_weights_key="loss_weights",
        loss_weight_transform=_compose_transforms(_swap_primary_secondary_transform, gen_ablation, w_visual_transform),
        segment_ids_transform=gen_seg_transform,
    )

    components = {
        **text_components,
        "multimodal_understanding": DatasetComponent(
            cache_dir=f"{multimodal_cache_path}/understanding",
            format=und_format,
            pack=True,
            tags=["multimodal", "understanding"],
        ),
        "multimodal_generation": DatasetComponent(
            cache_dir=f"{multimodal_cache_path}/generation",
            format=gen_format,
            pack=True,
            tags=["multimodal", "generation"],
        ),
    }
    weights = {
        **text_weights,
        "multimodal_understanding": und_w,
        "multimodal_generation": gen_w,
    }

    # Visual-only: pure visual AR sequences (no text).
    # Only included when VIS_STEPS > 0; all loss_weights in the cache are 1.0.
    if VIS_STEPS > 0:
        vis_format = PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
        )
        components["visual_only"] = DatasetComponent(
            cache_dir=f"{visual_only_cache_path}/train",
            format=vis_format,
            pack=True,
            tags=["visual"],
        )
        weights["visual_only"] = vis_w

        components["val_visual_only"] = DatasetComponent(
            cache_dir=f"{visual_only_cache_path}/validation",
            format=vis_format,
            pack=True,
            tags=["visual"],
        )
        weights["val_visual_only"] = 0.0

    # Multimodal validation: separate understanding and generation val loss.
    # These are produced by vlm_tokenize_captions.py with val_fraction > 0.
    # Validation formats: no ablation transforms, but generation still needs the swap fix.
    und_val_format = PrebuiltLmDatasetFormat(
        input_ids_key="input_ids",
        loss_weights_key="loss_weights",
        loss_weight_transform=w_visual_transform,
    )
    gen_val_format = PrebuiltLmDatasetFormat(
        input_ids_key="input_ids",
        loss_weights_key="loss_weights",
        loss_weight_transform=_compose_transforms(_swap_primary_secondary_transform, w_visual_transform),
    )

    components["val_understanding"] = DatasetComponent(
        cache_dir=f"{multimodal_cache_path}/understanding/val_understanding",
        format=und_val_format,
        pack=True,
        tags=["multimodal", "understanding"],
    )
    weights["val_understanding"] = 0.0

    components["val_generation"] = DatasetComponent(
        cache_dir=f"{multimodal_cache_path}/generation/val_generation",
        format=gen_val_format,
        pack=True,
        tags=["multimodal", "generation"],
    )
    weights["val_generation"] = 0.0

    # Per-modality val loss breakdown (text-only and visual-only).
    # After the swap fix, both caches have: fractional=visual, 1.0=text.
    # _zero_fractional_transform: zeros visual (fractional), keeps text (1.0).
    # _only_fractional_transform: keeps visual (fractional→1.0), zeros text (1.0→0).
    und_val_cache = f"{multimodal_cache_path}/understanding/val_understanding"
    gen_val_cache = f"{multimodal_cache_path}/generation/val_generation"

    # Understanding: no swap needed (convention already correct)
    components["val_understanding_text_only"] = DatasetComponent(
        cache_dir=und_val_cache,
        format=PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
            loss_weight_transform=_zero_fractional_transform,
        ),
        pack=True,
        tags=["multimodal", "understanding"],
    )
    components["val_understanding_visual_only"] = DatasetComponent(
        cache_dir=und_val_cache,
        format=PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
            loss_weight_transform=_only_fractional_transform,
        ),
        pack=True,
        tags=["multimodal", "understanding"],
    )
    weights["val_understanding_text_only"] = 0.0
    weights["val_understanding_visual_only"] = 0.0

    # Generation: swap first to normalize convention, then filter
    components["val_generation_text_only"] = DatasetComponent(
        cache_dir=gen_val_cache,
        format=PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
            loss_weight_transform=_compose_transforms(_swap_primary_secondary_transform, _zero_fractional_transform),
        ),
        pack=True,
        tags=["multimodal", "generation"],
    )
    components["val_generation_visual_only"] = DatasetComponent(
        cache_dir=gen_val_cache,
        format=PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
            loss_weight_transform=_compose_transforms(_swap_primary_secondary_transform, _only_fractional_transform),
        ),
        pack=True,
        tags=["multimodal", "generation"],
    )
    weights["val_generation_text_only"] = 0.0
    weights["val_generation_visual_only"] = 0.0

    # Ablation val: replace one modality with EOS → segment-based attention blocking
    # prevents real tokens from attending to the replaced positions.
    fmt_und_wo_visual = PrebuiltLmDatasetFormat(
        input_ids_key="input_ids",
        loss_weights_key="loss_weights",
        loss_weight_transform=_zero_fractional_transform,
        input_ids_transform=_replace_visual_with_eos,
    )
    fmt_gen_wo_language = PrebuiltLmDatasetFormat(
        input_ids_key="input_ids",
        loss_weights_key="loss_weights",
        loss_weight_transform=_compose_transforms(_swap_primary_secondary_transform, _zero_fractional_transform),
        input_ids_transform=_replace_text_with_eos,
    )

    components["val_understanding_wo_visual"] = DatasetComponent(
        cache_dir=und_val_cache,
        format=fmt_und_wo_visual,
        pack=True,
        tags=["multimodal", "understanding"],
    )
    weights["val_understanding_wo_visual"] = 0.0

    components["val_generation_wo_language"] = DatasetComponent(
        cache_dir=gen_val_cache,
        format=fmt_gen_wo_language,
        pack=True,
        tags=["multimodal", "generation"],
    )
    weights["val_generation_wo_language"] = 0.0

    # Eval benchmarks: weight 0.0 → eval-only (not used in training data).
    # Levanter's validation_sets() loads these for periodic eval during training.
    if eval_benchmarks is not None:
        for bench in eval_benchmarks:
            tags = ["eval"]
            if bench in UNDERSTANDING_BENCHMARKS:
                tags.append("understanding")
                eval_fmt = und_val_format
            elif bench in GENERATION_BENCHMARKS:
                tags.append("generation")
                eval_fmt = gen_val_format
            else:
                eval_fmt = und_val_format

            components[f"eval_{bench}"] = DatasetComponent(
                cache_dir=f"{eval_cache_path}/{bench}",
                format=eval_fmt,
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
        metadata={
            "text_steps": TEXT_STEPS,
            "und_steps": UND_STEPS,
            "gen_steps": GEN_STEPS,
            "vis_steps": VIS_STEPS,
            "text_w": text_w,
            "und_w": und_w,
            "gen_w": gen_w,
            "vis_w": vis_w,
            "w_visual": w_visual,
        },
    )

    # Text-only validation sets (Paloma, uncheatable_eval) use the base Llama3
    # tokenizer since the unified tokenizer only adds visual tokens — text
    # tokenization is identical, so we reuse existing Llama3-tokenized caches.
    validation_sets = default_validation_sets(tokenizer=llama3_tokenizer)
    data_config = add_validation_sets_to_mixture(data_config, validation_sets)

    return data_config


def _demo_train_config() -> SimpleTrainConfig:
    if OPTIMIZER == "muon":
        optimizer = MuonConfig(
            learning_rate=MUON_LR,
            adam_lr=ADAM_LR,
            weight_decay=0.1,
            momentum=0.98,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1.0,
            lr_schedule=LR_SCHEDULE,
            decay=1.0,
            min_lr_ratio=0,
            warmup=0,
        )
        lr = MUON_LR
        warmup = 0
    elif OPTIMIZER == "adam":
        optimizer = None
        lr = ADAM_LR
        warmup = 0.01
    else:
        raise ValueError(f"Unknown OPTIMIZER={OPTIMIZER!r}. Valid: 'muon', 'adam'")

    return SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(TPU_TYPE, slice_count=1),
        train_batch_size=256,
        num_train_steps=DEMO_TRAIN_STEPS,
        learning_rate=lr,
        warmup=warmup,
        lr_schedule=LR_SCHEDULE,
        weight_decay=0.1,
        max_grad_norm=1.0,
        per_device_parallelism=1,
        steps_per_eval=500,
        steps_per_export=1000,
        optimizer_config=optimizer,
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

DEFAULT_TEXT_EVAL_BENCHMARKS = [
    "hellaswag",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "mmlu",
]


def make_unified_0_6b(
    eval_benchmarks: list[str] | None = DEFAULT_EVAL_BENCHMARKS,
    w_visual: float | None = 1.0,
    text_eval_benchmarks: list[str] | None = DEFAULT_TEXT_EVAL_BENCHMARKS,
    ablation_mode: str = "",
):
    step = default_train(
        name=EXP_NAME or "unified-qwen3-0.6b-demo",
        tokenized=unified_data_config(
            eval_benchmarks=eval_benchmarks,
            w_visual=w_visual,
            text_eval_benchmarks=text_eval_benchmarks,
            ablation_mode=ablation_mode,
        ),
        model_config=qwen3_0_6b,
        train_config=_demo_train_config(),
        tags=["unified", "scaling", "qwen3", "0.6b", "demo"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    # Multimodal caches live in gs://marin-vlm (us multi-region); allow cross-region read
    step = dataclasses.replace(step, config=dataclasses.replace(step.config, allow_out_of_region=("data.components",)))
    return _with_vlm_hlo_dump_env(step)


def make_unified_1_7b(
    eval_benchmarks: list[str] | None = DEFAULT_EVAL_BENCHMARKS,
    w_visual: float | None = 1.0,
    text_eval_benchmarks: list[str] | None = DEFAULT_TEXT_EVAL_BENCHMARKS,
    ablation_mode: str = "",
):
    step = default_train(
        name=EXP_NAME or "unified-qwen3-1.7b-demo",
        tokenized=unified_data_config(
            eval_benchmarks=eval_benchmarks,
            w_visual=w_visual,
            text_eval_benchmarks=text_eval_benchmarks,
            ablation_mode=ablation_mode,
        ),
        model_config=qwen3_1_7b,
        train_config=_demo_train_config(),
        tags=["unified", "scaling", "qwen3", "1.7b", "demo"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    step = dataclasses.replace(step, config=dataclasses.replace(step.config, allow_out_of_region=("data.components",)))
    return _with_vlm_hlo_dump_env(step)


def make_unified_4b(
    eval_benchmarks: list[str] | None = DEFAULT_EVAL_BENCHMARKS,
    w_visual: float | None = 1.0,
    text_eval_benchmarks: list[str] | None = DEFAULT_TEXT_EVAL_BENCHMARKS,
    ablation_mode: str = "",
):
    step = default_train(
        name=EXP_NAME or "unified-qwen3-4b-demo",
        tokenized=unified_data_config(
            eval_benchmarks=eval_benchmarks,
            w_visual=w_visual,
            text_eval_benchmarks=text_eval_benchmarks,
            ablation_mode=ablation_mode,
        ),
        model_config=qwen3_4b,
        train_config=_demo_train_config(),
        tags=["unified", "scaling", "qwen3", "4b", "demo"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    step = dataclasses.replace(step, config=dataclasses.replace(step.config, allow_out_of_region=("data.components",)))
    return _with_vlm_hlo_dump_env(step)


if __name__ == "__main__":
    # steps = [make_unified_0_6b(w_visual=W_VISUAL, ablation_mode=ABLATION_MODE)]
    steps = [
        make_unified_1_7b(
            w_visual=W_VISUAL,
            ablation_mode=ABLATION_MODE,
        )
    ]
    # steps = [make_unified_4b(w_visual=W_VISUAL, ablation_mode=ABLATION_MODE)]
    executor_main(
        steps,
        description="Unified image-text model pre-training with Qwen3 architecture",
    )

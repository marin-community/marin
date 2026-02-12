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

import logging
import tempfile

from levanter.data.text import DatasetComponent, LmDataConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat

from experiments.defaults import default_train
from experiments.pretraining_datasets import NEMOTRON_WEIGHTS, tokenize_nemotron
from experiments.qwen3 import qwen3_0_6b, qwen3_1_7b, qwen3_4b
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution import executor_main
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

UNIFIED_TOKENIZER_PATH = "gs://marin-vlm/tokenizers/llama3-unified-144k"
UNIFIED_CACHE_PATH = "gs://marin-vlm/unified_pretraining_cache"


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
) -> LmDataConfig:
    """Build data mixture with Nemotron text + multimodal cache for unified model training.

    Args:
        text_weight: Scaling factor applied to each Nemotron split's weight (controls r1).
        multimodal_weight: Weight for the multimodal caption data component.
        multimodal_cache_path: GCS path to the pre-built multimodal Levanter cache.
    """
    # Text: use existing Llama3-tokenized Nemotron caches (no re-tokenization needed)
    nemotron_steps = tokenize_nemotron()
    text_components = {
        name: step_to_lm_mixture_component(step, include_raw_paths=True) for name, step in nemotron_steps.items()
    }
    text_weights = {k: text_weight * v for k, v in NEMOTRON_WEIGHTS.items()}

    # Multimodal: pre-built cache with per-token loss weights.
    # pack=True to avoid wasting compute padding short sequences (~600 tokens) to 4096.
    multimodal_component = DatasetComponent(
        cache_dir=f"{multimodal_cache_path}/train",
        format=PrebuiltLmDatasetFormat(
            input_ids_key="input_ids",
            loss_weights_key="loss_weights",
        ),
        pack=True,
    )

    components = {**text_components, "multimodal_captions": multimodal_component}
    weights = {**text_weights, "multimodal_captions": multimodal_weight}

    return LmDataConfig(
        tokenizer=UNIFIED_TOKENIZER_PATH,
        components=components,
        train_weights=weights,
        shuffle=True,
        permutation_type="feistel",
        block_cross_document_attention=True,
    )


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


def make_unified_0_6b(text_weight: float = 0.7, multimodal_weight: float = 0.3):
    return default_train(
        name="unified-qwen3-0.6b",
        tokenized=unified_data_config(text_weight=text_weight, multimodal_weight=multimodal_weight),
        model_config=qwen3_0_6b,
        train_config=unified_0_6b_train,
        tags=["unified", "scaling", "qwen3", "0.6b"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )


def make_unified_1_7b(text_weight: float = 0.7, multimodal_weight: float = 0.3):
    return default_train(
        name="unified-qwen3-1.7b",
        tokenized=unified_data_config(text_weight=text_weight, multimodal_weight=multimodal_weight),
        model_config=qwen3_1_7b,
        train_config=unified_1_7b_train,
        tags=["unified", "scaling", "qwen3", "1.7b"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )


def make_unified_4b(text_weight: float = 0.7, multimodal_weight: float = 0.3):
    return default_train(
        name="unified-qwen3-4b",
        tokenized=unified_data_config(text_weight=text_weight, multimodal_weight=multimodal_weight),
        model_config=qwen3_4b,
        train_config=unified_4b_train,
        tags=["unified", "scaling", "qwen3", "4b"],
        eval_harness_tasks=[],
        use_default_validation=False,
    )


if __name__ == "__main__":
    steps = [make_unified_0_6b(), make_unified_1_7b(), make_unified_4b()]
    executor_main(
        steps,
        description="Unified image-text model pre-training with Qwen3 architecture",
    )

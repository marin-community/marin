# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import replace
from functools import lru_cache
from typing import Literal

from levanter.data.text.datasets import (
    DEFAULT_LM_DATA_SHUFFLE,
    BlockShuffleConfig,
    DatasetComponent,
    LmDataConfig,
    LmDatasetSourceConfigBase,
)
from levanter.tokenizers import load_tokenizer

from marin.processing.tokenize.tokenize import TokenizeConfig

logger = logging.getLogger(__name__)

_KNOWN_VOCAB_SIZES: dict[str, int] = {
    "EleutherAI/gpt-neox-20b": 50_257,
    "meta-llama/Meta-Llama-3.1-8B": 128_256,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 128_256,
    "marin-community/marin-tokenizer": 128_256,
    "meta-llama/Llama-2-7b": 32_000,
    "gpt2": 50_257,
}

# The marin tokenizer is a re-upload of the llama3 tokenizer with a custom chat template; the
# Llama-3.1 base and Instruct checkpoints likewise share the same vocabulary and token IDs and
# differ only in their chat template. Listing all of them lets _are_tokenizers_equivalent
# short-circuit without loading the gated meta-llama tokenizers from the Hub.
_EQUIVALENT_TOKENIZERS = frozenset(
    {
        "meta-llama/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "marin-community/marin-tokenizer",
    }
)


def dataset_component(source: LmDatasetSourceConfigBase) -> DatasetComponent:
    """Wrap a resolved dataset source as a Levanter mixture component, carrying its
    cache dir, format, and tags."""
    return DatasetComponent(source=source, cache_dir=source.cache_dir, format=source.format, tags=source.tags)


def with_pack(data: LmDataConfig, pack: bool | int | Literal["pad"]) -> LmDataConfig:
    """Override the packing strategy on every cache-backed component of a mixture.

    Packing is a load-time view over the tokenized cache, so this re-tokenizes nothing.
    Components without a ``pack`` field (concat/direct) are returned unchanged.
    """
    return replace(
        data,
        components={
            name: replace(component, pack=pack) if isinstance(component, DatasetComponent) else component
            for name, component in data.components.items()
        },
    )


def step_to_lm_mixture_component(step: TokenizeConfig, include_raw_paths: bool) -> DatasetComponent:
    """Convert a tokenize config to a Levanter dataset component, for building data mixtures."""
    source = step.as_lm_dataset_source_config(step.cache_path, include_raw_paths=include_raw_paths)
    return dataset_component(source)


def lm_mixture_data_config(
    components: dict[str, TokenizeConfig],
    weights: dict[str, float],
    *,
    shuffle: bool | BlockShuffleConfig = DEFAULT_LM_DATA_SHUFFLE,
    missing_weights_are_validation: bool = True,
    include_raw_paths: bool = True,
    max_train_batches: dict[str, int] | None = None,
    num_validation_sequences: dict[str, int] | None = None,
    shuffle_before_trainval_split: bool = True,
    mixture_block_size: int | None = None,
    block_cross_document_attention: bool = True,
) -> LmDataConfig:
    """
    Creates a training config from a mixture of datasources.

    Args:
        components: dict from names of datasets to the steps that produced them.
        weights: dict from names of datasets to their weights.
        shuffle: shuffling policy. Defaults to hierarchical block shuffle.
            `True` enables a full permutation shuffle; `BlockShuffleConfig` enables hierarchical block shuffling.
        missing_weights_are_validation: whether to pad out missing weights with 0's, indicating validation-only sets
        include_raw_paths: whether to include raw paths in the dataset config. This is mostly for logging purposes.
        max_train_batches: Maximum number of batches to use for the training set per dataset.
        num_validation_sequences: Number of validation sequences to take from the training set per dataset.
        shuffle_before_trainval_split: Whether to shuffle before splitting into train/val. Defaults to True.
        block_cross_document_attention: Whether to mask attention across document boundaries.
    """
    component_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=include_raw_paths)
        for name, step in components.items()
    }

    if missing_weights_are_validation:
        missing_keys = {k: 0.0 for k in components if k not in weights}
        weights = {**weights, **missing_keys}

    tokenizer = _verify_tokenizers_same(components)

    kwargs = {}
    if mixture_block_size is not None:
        kwargs["mixture_block_size"] = mixture_block_size

    return LmDataConfig(
        components=component_configs,
        train_weights=weights,
        tokenizer=tokenizer,
        cache_dir=None,
        shuffle=shuffle,
        permutation_type="feistel",
        max_train_batches=max_train_batches,
        num_validation_sequences=num_validation_sequences,
        shuffle_before_trainval_split=shuffle_before_trainval_split,
        block_cross_document_attention=block_cross_document_attention,
        **kwargs,
    )


@lru_cache(maxsize=128)
def get_vocab_size_for_tokenizer(tokenizer_name: str) -> int:
    """Return the vocabulary size for a tokenizer name.

    Args:
        tokenizer_name: HuggingFace tokenizer name or path.

    Returns:
        Vocabulary size for the tokenizer.
    """
    if tokenizer_name in _KNOWN_VOCAB_SIZES:
        return _KNOWN_VOCAB_SIZES[tokenizer_name]

    logger.warning(
        "Tokenizer %r not found in _KNOWN_VOCAB_SIZES; loading from HuggingFace. "
        "Consider adding it to _KNOWN_VOCAB_SIZES in data_configs.py to avoid network calls during dry-runs.",
        tokenizer_name,
    )
    tokenizer = load_tokenizer(tokenizer_name)
    return tokenizer.vocab_size


def _are_tokenizers_equivalent(tokenizer1: str, tokenizer2: str) -> bool:
    """Compare two tokenizers by loading them and comparing their vocabularies and token IDs"""
    # The marin tokenizer is a re-upload of the llama3 tokenizer with a custom chat template,
    # so they share the same vocabulary and token IDs.
    if tokenizer1 in _EQUIVALENT_TOKENIZERS and tokenizer2 in _EQUIVALENT_TOKENIZERS:
        return True

    t1 = load_tokenizer(tokenizer1)
    t2 = load_tokenizer(tokenizer2)

    # Compare vocab sizes
    if len(t1.get_vocab()) != len(t2.get_vocab()):
        return False

    # Compare vocab contents and IDs
    vocab1 = t1.get_vocab()
    vocab2 = t2.get_vocab()

    # Check that all tokens exist in both vocabs with the same IDs
    for token, id1 in vocab1.items():
        if token not in vocab2:
            return False
        if vocab2[token] != id1:
            return False

    if t1.chat_template is not None and t2.chat_template is not None:
        if t1.chat_template != t2.chat_template:
            return False

    return True


def _verify_tokenizers_same(components: dict[str, TokenizeConfig]):
    first_name, first_step = next(iter(components.items()))
    tokenizer = first_step.tokenizer
    for name, step in components.items():
        step_tokenizer = step.tokenizer
        if step_tokenizer != tokenizer:
            if not _are_tokenizers_equivalent(step_tokenizer, tokenizer):
                raise ValueError(
                    "All components must have the same tokenizer, but got:"
                    f" {step_tokenizer} ({name}) vs {tokenizer} ({first_name})"
                )
            else:
                logger.warning(
                    f"Tokenizers ({name}) and {tokenizer} ({first_name}) have equivalent vocabularies but are not the"
                    " same tokenizer. This may cause issues with training."
                )
    return tokenizer

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lazy data primitives: tokenized datasets and mixtures as artifact handles.

``tokenize`` returns a :class:`~marin.execution.lazy.Dataset` handle that produces
a tokenized cache; ``mixture`` assembles a Levanter ``LmDataConfig`` from those
handles, resolving each component's cache path with ``ctx.path(dataset)``.

This replaces the ``ExecutorStep`` + ``this_output_path()`` + ``InputName`` dataset
catalog: the tokenize step's ``cache_path`` is ``ctx.out``, and a consumer wires a
dependency by passing the handle to ``mixture`` (which reads ``ctx.path(handle)``).
"""

from collections.abc import Mapping, Sequence

from levanter.data.text import (
    DEFAULT_LM_DATA_SHUFFLE,
    BlockShuffleConfig,
    DatasetComponent,
    LmDataConfig,
    LmDatasetFormatBase,
    TextLmDatasetFormat,
)
from levanter.data.text.datasets import LmDatasetSourceConfigBase

from marin.execution.lazy import BuildContext, Dataset, Recipe
from marin.processing.tokenize.tokenize import HfTokenizeConfig, TokenizeConfig, TokenizeConfigBase
from marin.processing.tokenize.tokenize import tokenize as _tokenize


def _looks_like_hf_id(source: str) -> bool:
    """A bare ``org/name`` HuggingFace id (one slash, not a filesystem/URL path)."""
    return source.count("/") == 1 and "://" not in source and not source.startswith("/")


def _tokenize_config(
    source: str,
    tokenizer: str,
    format: LmDatasetFormatBase,  # noqa: A002 (matches default_tokenize's public arg name)
    *,
    cache_path: str,
    tags: Sequence[str],
) -> TokenizeConfigBase:
    if _looks_like_hf_id(source):
        return HfTokenizeConfig(id=source, cache_path=cache_path, tokenizer=tokenizer, format=format, tags=[*tags])
    return TokenizeConfig(
        train_paths=[source],
        validation_paths=[],
        cache_path=cache_path,
        tokenizer=tokenizer,
        format=format,
        tags=[*tags],
    )


def tokenize(
    name: str,
    version: str,
    *,
    source: str,
    tokenizer: str,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),  # noqa: A002
    tags: Sequence[str] = (),
    resources=None,
) -> Dataset:
    """A tokenized-dataset handle. ``source`` is a HuggingFace id (``org/name``) or a
    raw data path/glob; the tokenized cache is written at ``ctx.out``."""

    def build_config(ctx: BuildContext) -> TokenizeConfigBase:
        return _tokenize_config(source, tokenizer, format, cache_path=ctx.out, tags=tags)

    return Dataset(
        name=name,
        version=version,
        recipe=Recipe(fn=_tokenize, build_config=build_config, resources=resources),
    )


def _component_for(dataset: Dataset, cache_path: str) -> DatasetComponent:
    """Build a Levanter mixture component for ``dataset`` rooted at ``cache_path``."""
    config = dataset.recipe.build_config(BuildContext.for_fingerprint())
    if not isinstance(config, TokenizeConfigBase):
        raise TypeError(f"{dataset.name}: mixture component must be a tokenize dataset, got {type(config).__name__}")
    source: LmDatasetSourceConfigBase = config.as_lm_dataset_source_config(cache_path)
    return DatasetComponent(source=source, cache_dir=source.cache_dir, format=source.format, tags=source.tags)


def mixture(
    ctx: BuildContext,
    components: Mapping[Dataset, float],
    *,
    validation: Mapping[str, Dataset] | None = None,
    shuffle: bool | BlockShuffleConfig = DEFAULT_LM_DATA_SHUFFLE,
) -> LmDataConfig:
    """Assemble an ``LmDataConfig`` from dataset handles.

    Each training component's cache path is resolved with ``ctx.path(dataset)``;
    validation handles are added with weight 0. Call this inside a consumer's
    ``build_config`` and pass the same handles as the recipe's ``deps`` so they are
    materialized first.
    """
    component_configs: dict[str, DatasetComponent] = {}
    weights: dict[str, float] = {}

    tokenizers: set[str] = set()
    for dataset, weight in components.items():
        component_configs[dataset.name] = _component_for(dataset, ctx.path(dataset))
        weights[dataset.name] = weight
        tokenizers.add(dataset.recipe.build_config(BuildContext.for_fingerprint()).tokenizer)

    for name, dataset in (validation or {}).items():
        component_configs[name] = _component_for(dataset, ctx.path(dataset))
        weights[name] = 0.0
        tokenizers.add(dataset.recipe.build_config(BuildContext.for_fingerprint()).tokenizer)

    if len(tokenizers) != 1:
        raise ValueError(f"mixture components must share one tokenizer, got {sorted(tokenizers)}")

    return LmDataConfig(
        components=component_configs,
        train_weights=weights,
        tokenizer=tokenizers.pop(),
        cache_dir=None,
        shuffle=shuffle,
        permutation_type="feistel",
    )

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lazy data primitives: tokenized datasets and mixtures as artifact handles.

``tokenize`` returns a :class:`~marin.execution.lazy.Dataset` handle that produces
a tokenized cache; ``mixture`` assembles a Levanter ``LmDataConfig`` from those
handles, resolving each component's cache path with ``ctx.path(dataset)``.

This replaces the ``ExecutorStep`` + ``this_output_path()`` + ``InputName`` dataset
catalog: the tokenize step's ``cache_path`` is ``ctx.out``, raw input globs resolve
against ``ctx.prefix``, and a consumer wires a dependency by passing the handle to
``mixture`` (which reads ``ctx.path(handle)``). ``pinned_path`` references already
tokenized data at its existing location instead of recomputing it.
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

from marin.execution.lazy import BuildContext, Dataset, Recipe
from marin.execution.step_spec import _is_relative_path
from marin.processing.tokenize.tokenize import HfTokenizeConfig, TokenizeConfig, TokenizeConfigBase
from marin.processing.tokenize.tokenize import tokenize as _tokenize


def _looks_like_hf_id(source: str) -> bool:
    """A bare ``org/name`` HuggingFace id (one slash, not a filesystem/URL path)."""
    return source.count("/") == 1 and "://" not in source and not source.startswith("/")


def _resolve(prefix: str, path: str) -> str:
    """Make a prefix-relative raw path absolute; leave absolute/URL paths alone."""
    return f"{prefix}/{path}" if _is_relative_path(path) else path


def _build_tokenize_config(
    ctx: BuildContext,
    *,
    source: str | None,
    train_paths: Sequence[str] | None,
    tokenizer: str,
    format: LmDatasetFormatBase,  # noqa: A002 (matches default_tokenize's public arg name)
    tags: Sequence[str],
) -> TokenizeConfigBase:
    if source is not None and _looks_like_hf_id(source):
        return HfTokenizeConfig(id=source, cache_path=ctx.out, tokenizer=tokenizer, format=format, tags=[*tags])
    raw = [source] if source is not None else list(train_paths or [])
    return TokenizeConfig(
        train_paths=[_resolve(ctx.prefix, p) for p in raw],
        validation_paths=[],
        cache_path=ctx.out,
        tokenizer=tokenizer,
        format=format,
        tags=[*tags],
    )


def tokenize(
    name: str,
    version: str,
    *,
    source: str | None = None,
    train_paths: Sequence[str] | None = None,
    tokenizer: str,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),  # noqa: A002
    tags: Sequence[str] = (),
    resources=None,
    pinned_path: str | None = None,
) -> Dataset:
    """A tokenized-dataset handle.

    Provide either ``source`` (a HuggingFace id ``org/name`` or a single raw path)
    or ``train_paths`` (raw globs, resolved against ``ctx.prefix``). The tokenized
    cache is written at ``ctx.out``. ``pinned_path`` references already-tokenized
    data at an existing location instead of recomputing it.
    """
    if (source is None) == (train_paths is None):
        raise ValueError(f"{name}: provide exactly one of source or train_paths")

    def build_config(ctx: BuildContext) -> TokenizeConfigBase:
        return _build_tokenize_config(
            ctx, source=source, train_paths=train_paths, tokenizer=tokenizer, format=format, tags=tags
        )

    return Dataset(
        name=name,
        version=version,
        recipe=Recipe(fn=_tokenize, build_config=build_config, resources=resources),
        override_path=pinned_path,
    )


def _component_for(dataset: Dataset, ctx: BuildContext) -> DatasetComponent:
    """Build a Levanter mixture component for ``dataset``, rooted at its resolved path."""
    cache_path = ctx.path(dataset)
    config = dataset.recipe.build_config(BuildContext.for_run(out=cache_path, prefix=ctx.prefix))
    if not isinstance(config, TokenizeConfigBase):
        raise TypeError(f"{dataset.name}: mixture component must be a tokenize dataset, got {type(config).__name__}")
    source = config.as_lm_dataset_source_config(cache_path)
    return DatasetComponent(source=source, cache_dir=source.cache_dir, format=source.format, tags=source.tags)


def _tokenizer_of(dataset: Dataset) -> str:
    return dataset.recipe.build_config(BuildContext.for_fingerprint()).tokenizer


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
        component_configs[dataset.name] = _component_for(dataset, ctx)
        weights[dataset.name] = weight
        tokenizers.add(_tokenizer_of(dataset))

    for name, dataset in (validation or {}).items():
        component_configs[name] = _component_for(dataset, ctx)
        weights[name] = 0.0
        tokenizers.add(_tokenizer_of(dataset))

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

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lazy data builders: tokenized datasets and mixtures as artifact handles.

The library provides *mechanism* — concise builders — while an experiment states
the *policy*: which datasets, at what weights. The split is deliberate, so mixture
weights live in the experiment that chose them, not buried in a catalog constant.

- :func:`tokenized` returns a :class:`~marin.execution.lazy.Dataset` handle. Its raw
  input is one of: ``source`` (a HuggingFace id or single path), ``paths`` (raw globs
  resolved against the run prefix), or ``raw=`` a download handle + ``glob`` within it
  (a download -> tokenize dependency). ``pin`` references already-tokenized data at an
  existing location instead of recomputing it.
- :func:`raw_download` wraps a download function as a raw-data handle that
  :func:`tokenized` can depend on.
- :func:`mixture` assembles a Levanter ``LmDataConfig`` from ``{handle: weight}``
  training components plus weight-0 ``validation`` handles, resolving each cache path
  with ``ctx.path(handle)``.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from fray.types import ResourceConfig
from levanter.data.text import (
    DEFAULT_LM_DATA_SHUFFLE,
    BlockShuffleConfig,
    DatasetComponent,
    LmDataConfig,
    TextLmDatasetFormat,
)

from marin.execution.lazy import Dataset, Recipe, RunContext
from marin.execution.step_spec import _is_relative_path
from marin.processing.tokenize.tokenize import HfTokenizeConfig, TokenizeConfig, TokenizeConfigBase
from marin.processing.tokenize.tokenize import tokenize as _tokenize_fn

DEFAULT_VERSION = "v1"


def _looks_like_hf_id(source: str) -> bool:
    """A bare ``org/name`` HuggingFace id (one slash, not a filesystem/URL path)."""
    return source.count("/") == 1 and "://" not in source and not source.startswith("/")


def _resolve(prefix: str, path: str) -> str:
    """Make a prefix-relative raw path absolute; leave absolute/URL paths alone."""
    return f"{prefix}/{path}" if _is_relative_path(path) else path


def raw_download(
    name: str,
    *,
    fn: Callable[[Any], Any],
    build_config: Callable[[RunContext], Any],
    version: str = DEFAULT_VERSION,
    pin: str | None = None,
    resources: ResourceConfig | None = None,
) -> Dataset:
    """A raw-data download handle: ``build_config(ctx)`` writes the download to ``ctx.out``.

    Returned as a :class:`Dataset` so :func:`tokenized` can depend on it; it is not a
    tokenized cache itself.
    """
    return Dataset(
        name=name,
        version=version,
        recipe=Recipe(fn=fn, build_config=build_config, resources=resources),
        override_path=pin,
    )


def tokenized(
    name: str,
    *,
    tokenizer: str,
    source: str | None = None,
    paths: Sequence[str] | None = None,
    raw: Dataset | None = None,
    glob: str | None = None,
    validation: bool = False,
    pin: str | None = None,
    text_key: str = "text",
    version: str = DEFAULT_VERSION,
    tags: Sequence[str] = (),
    resources: ResourceConfig | None = None,
) -> Dataset:
    """A tokenized-dataset handle.

    Provide exactly one raw input: ``source`` (a HuggingFace id ``org/name`` or a single
    raw path), ``paths`` (raw globs resolved against the run prefix), or ``raw`` + ``glob``
    (a download handle and a subpath glob within it). ``validation=True`` routes the data
    to the cache's validation split. ``pin`` references already-tokenized data at an existing
    location instead of recomputing it.
    """
    if sum(x is not None for x in (source, paths, raw)) != 1:
        raise ValueError(f"{name}: provide exactly one of source, paths, or raw")
    if (raw is None) != (glob is None):
        raise ValueError(f"{name}: raw and glob must be given together")

    fmt = TextLmDatasetFormat(text_key=text_key)

    def build_config(ctx: RunContext) -> TokenizeConfigBase:
        if source is not None and _looks_like_hf_id(source):
            return HfTokenizeConfig(id=source, cache_path=ctx.out, tokenizer=tokenizer, format=fmt, tags=[*tags])
        if raw is not None:
            resolved = [f"{ctx.path(raw)}/{glob}"]
        elif paths is not None:
            resolved = [_resolve(ctx.prefix, p) for p in paths]
        else:
            resolved = [_resolve(ctx.prefix, source)]
        return TokenizeConfig(
            train_paths=[] if validation else resolved,
            validation_paths=resolved if validation else [],
            cache_path=ctx.out,
            tokenizer=tokenizer,
            format=fmt,
            tags=[*tags],
        )

    return Dataset(
        name=name,
        version=version,
        recipe=Recipe(
            fn=_tokenize_fn, build_config=build_config, deps=(raw,) if raw is not None else (), resources=resources
        ),
        override_path=pin,
    )


def _component_for(dataset: Dataset, ctx: RunContext) -> DatasetComponent:
    """Build a Levanter mixture component for ``dataset``, rooted at its resolved path."""
    cache_path = ctx.path(dataset)
    config = dataset.recipe.build_config(
        RunContext.for_run(out=cache_path, prefix=ctx.prefix, run_args=dataset.recipe.run_args)
    )
    if not isinstance(config, TokenizeConfigBase):
        raise TypeError(f"{dataset.name}: mixture component must be a tokenize dataset, got {type(config).__name__}")
    source = config.as_lm_dataset_source_config(cache_path)
    return DatasetComponent(source=source, cache_dir=source.cache_dir, format=source.format, tags=source.tags)


def _tokenizer_of(dataset: Dataset) -> str:
    return dataset.recipe.build_config(RunContext.for_fingerprint(dataset.recipe.run_args.keys())).tokenizer


def mixture(
    ctx: RunContext,
    train: Mapping[Dataset, float],
    *,
    validation: Sequence[Dataset] = (),
    shuffle: bool | BlockShuffleConfig = DEFAULT_LM_DATA_SHUFFLE,
) -> LmDataConfig:
    """Assemble an ``LmDataConfig`` from dataset handles.

    ``train`` maps each handle to its mixture weight; ``validation`` handles are added at
    weight 0. Each component's cache path is resolved with ``ctx.path(handle)``; the
    component key is the handle's ``name``. Call this inside a consumer's ``build_config``
    and pass the same handles as the recipe's ``deps`` so they materialize first.
    """
    components: dict[str, DatasetComponent] = {}
    weights: dict[str, float] = {}
    tokenizers: set[str] = set()

    for dataset, weight in train.items():
        components[dataset.name] = _component_for(dataset, ctx)
        weights[dataset.name] = weight
        tokenizers.add(_tokenizer_of(dataset))

    for dataset in validation:
        components[dataset.name] = _component_for(dataset, ctx)
        weights[dataset.name] = 0.0
        tokenizers.add(_tokenizer_of(dataset))

    if len(tokenizers) != 1:
        raise ValueError(f"mixture components must share one tokenizer, got {sorted(tokenizers)}")

    return LmDataConfig(
        components=components,
        train_weights=weights,
        tokenizer=tokenizers.pop(),
        cache_dir=None,
        shuffle=shuffle,
        permutation_type="feistel",
    )

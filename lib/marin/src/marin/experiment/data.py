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
  :func:`tokenized` can depend on; :func:`hf_download` is the HuggingFace-Hub case.
- :func:`derived` is the generic single-step builder: ``fn(build_config(ctx))`` writing
  to ``ctx.out``, with declared ``deps``. Use it for transforms/conversions/filters
  (e.g. HF-dataset-to-eval-JSONL, Dolma conversions) that are neither a tokenize nor a
  plain download.
- :func:`pretokenized` handles an already-tokenized Levanter cache hosted on
  HuggingFace (e.g. the fineweb-edu prebuilt subcaches): it downloads rather than
  re-tokenizes, and is consumed like any other tokenized :class:`Dataset`.
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

from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.lazy import Artifact, Dataset, Recipe, RunContext
from marin.execution.remote import remote
from marin.execution.step_spec import _is_relative_path
from marin.processing.tokenize.data_configs import dataset_component
from marin.processing.tokenize.download_pretokenized import PretokenizedCacheDownloadConfig, fetch_pretokenized_cache
from marin.processing.tokenize.tokenize import HfTokenizeConfig, TokenizeConfig, TokenizeConfigBase
from marin.processing.tokenize.tokenize import tokenize as _tokenize_fn

DEFAULT_VERSION = "v1"


def _on(fn: Callable[[Any], Any], resources: ResourceConfig | None) -> Callable[[Any], Any]:
    """Run ``fn`` on Fray with ``resources`` (via :func:`remote`), or inline when None.

    Resources ride with the function, so they stay off the step node and out of the
    artifact fingerprint."""
    return remote(fn, resources=resources) if resources is not None else fn


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
        recipe=Recipe(fn=_on(fn, resources), build_config=build_config),
        override_path=pin,
    )


def hf_download(
    name: str,
    *,
    hf_id: str,
    revision: str,
    urls_glob: Sequence[str] = (),
    version: str = DEFAULT_VERSION,
    pin: str | None = None,
    resources: ResourceConfig | None = None,
) -> Dataset:
    """A HuggingFace-Hub dataset download as a raw-data handle.

    Wraps :func:`marin.datakit.download.huggingface.download_hf` into a handle that
    :func:`tokenized` (via ``raw=``) or :func:`derived` can depend on. ``urls_glob``
    restricts which files in the repo are fetched (empty = all). ``pin`` references an
    existing download at a fixed location instead of re-fetching it.
    """

    def build_config(ctx: RunContext) -> DownloadConfig:
        return DownloadConfig(
            hf_dataset_id=hf_id,
            revision=revision,
            hf_urls_glob=[*urls_glob],
            gcs_output_path=ctx.out,
            wait_for_completion=True,
        )

    return raw_download(name, fn=download_hf, build_config=build_config, version=version, pin=pin, resources=resources)


def derived(
    name: str,
    *,
    fn: Callable[[Any], Any],
    build_config: Callable[[RunContext], Any],
    deps: Sequence[Artifact] = (),
    version: str = DEFAULT_VERSION,
    pin: str | None = None,
    resources: ResourceConfig | None = None,
    kind: type[Artifact] = Artifact,
) -> Artifact:
    """A single derived artifact: ``fn(build_config(ctx))`` writes its result to ``ctx.out``.

    The generic builder behind transforms, conversions, and filters — anything that is
    neither a tokenize nor a plain download (e.g. HF-dataset-to-eval-JSONL, Dolma
    conversions, extension filters). ``deps`` are the upstream handles the build consumes;
    resolve each with ``ctx.path(dep)`` inside ``build_config`` and pass the same handles
    here so they materialize first. ``kind`` selects the handle type for consumer routing
    (:class:`~marin.execution.lazy.Dataset` for a tokenizable corpus,
    :class:`~marin.execution.lazy.Checkpoint` for a model; the base
    :class:`~marin.execution.lazy.Artifact` otherwise). ``pin`` references existing data
    instead of recomputing it.
    """
    return kind(
        name=name,
        version=version,
        recipe=Recipe(fn=_on(fn, resources), build_config=build_config, deps=tuple(deps)),
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
    sample_count: int | None = None,
    version: str = DEFAULT_VERSION,
    tags: Sequence[str] = (),
    resources: ResourceConfig | None = None,
) -> Dataset:
    """A tokenized-dataset handle.

    Provide exactly one raw input: ``source`` (a HuggingFace id ``org/name`` or a single
    raw path), ``paths`` (raw globs resolved against the run prefix), or ``raw`` + ``glob``
    (a download handle and a subpath glob within it). ``validation=True`` routes the data
    to the cache's validation split. ``sample_count`` caps the documents tokenized per shard
    (it bears identity — a sampled cache differs from the full one). ``pin`` references
    already-tokenized data at an existing location instead of recomputing it.
    """
    if sum(x is not None for x in (source, paths, raw)) != 1:
        raise ValueError(f"{name}: provide exactly one of source, paths, or raw")
    if (raw is None) != (glob is None):
        raise ValueError(f"{name}: raw and glob must be given together")

    fmt = TextLmDatasetFormat(text_key=text_key)

    def build_config(ctx: RunContext) -> TokenizeConfigBase:
        if source is not None and _looks_like_hf_id(source):
            return HfTokenizeConfig(
                id=source, cache_path=ctx.out, tokenizer=tokenizer, format=fmt, sample_count=sample_count, tags=[*tags]
            )
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
            sample_count=sample_count,
            tags=[*tags],
        )

    return Dataset(
        name=name,
        version=version,
        recipe=Recipe(
            fn=_on(_tokenize_fn, resources), build_config=build_config, deps=(raw,) if raw is not None else ()
        ),
        override_path=pin,
    )


def pretokenized(
    name: str,
    *,
    repo_id: str,
    tokenizer: str,
    revision: str | None = None,
    version: str = DEFAULT_VERSION,
    pin: str | None = None,
    tags: Sequence[str] = (),
    resources: ResourceConfig | None = None,
) -> Dataset:
    """A handle to an already-tokenized Levanter cache hosted on HuggingFace.

    ``build_config(ctx)`` downloads the HF dataset repo ``repo_id`` into ``ctx.out`` as
    a Levanter cache; the handle then reads as a tokenized :class:`Dataset` with no
    re-tokenization. Use it where a tokenizing :func:`tokenized` handle would be too
    slow — e.g. the fineweb-edu prebuilt subcaches. ``pin`` references an
    already-downloaded cache at an existing location instead of fetching it again.
    """

    def build_config(ctx: RunContext) -> PretokenizedCacheDownloadConfig:
        return PretokenizedCacheDownloadConfig(
            cache_path=ctx.out,
            tokenizer=tokenizer,
            hf_repo_id=repo_id,
            hf_revision=revision,
            tags=[*tags],
        )

    return Dataset(
        name=name,
        version=version,
        recipe=Recipe(fn=_on(fetch_pretokenized_cache, resources), build_config=build_config),
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
    return dataset_component(config.as_lm_dataset_source_config(cache_path))


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
    component key is the handle's ``name``, so two handles that share a name are rejected
    (rather than silently collapsing). Call this inside a consumer's ``build_config`` and
    pass the same handles as the recipe's ``deps`` so they materialize first.
    """
    handles = [*train, *validation]
    if not handles:
        raise ValueError("mixture needs at least one training or validation component")
    names = [dataset.name for dataset in handles]
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"mixture components are keyed by name, but these collide: {duplicates}")

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

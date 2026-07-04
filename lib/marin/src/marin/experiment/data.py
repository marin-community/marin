# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lazy data builders: tokenized datasets and mixtures as artifact handles.

The library provides *mechanism* — concise builders — while an experiment states
the *policy*: which datasets, at what weights. The split is deliberate, so mixture
weights live in the experiment that chose them, not buried in a catalog constant.

- :func:`tokenized` returns an ``ArtifactStep[TokenizedCache]`` handle. Its raw input is one of:
  ``source`` (a HuggingFace id or single path), ``paths`` (raw globs resolved against
  the run prefix), or ``raw=`` a download handle + ``glob`` within it (a download ->
  tokenize dependency). ``pin`` references already-tokenized data at an existing location
  instead of recomputing it.
- :func:`hf_download` is a HuggingFace-Hub download as a raw-data ``ArtifactStep[Artifact]`` that
  :func:`tokenized` (via ``raw=``) can depend on; :func:`raw_download` is the same for a
  non-Hub source.
- :func:`pretokenized` handles an already-tokenized Levanter cache hosted on
  HuggingFace (e.g. the fineweb-edu prebuilt subcaches): it downloads rather than
  re-tokenizes, and is consumed like any other ``TokenizedCache``.
- :func:`mixture` assembles a Levanter ``LmDataConfig`` from ``{handle: weight}``
  training components plus weight-0 ``validation`` handles, reading each cache's
  tokenizer/format from its :class:`~marin.processing.tokenize.tokenize.TokenizedCache`
  record at run time.
- :func:`dataset_main` is the ``python -m`` entry point for a dataset module: it prints
  the build plan by default and builds the module's handles under ``--run``.
"""

from collections.abc import Callable, Mapping, Sequence

import click
from fray.types import ResourceConfig
from levanter.data.text import (
    DEFAULT_LM_DATA_SHUFFLE,
    BlockShuffleConfig,
    LmDataConfig,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
)
from rigging.filesystem import prefix_join

from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, lower, run
from marin.execution.remote import remote
from marin.execution.step_spec import _is_relative_path
from marin.processing.tokenize.data_configs import dataset_component
from marin.processing.tokenize.download_pretokenized import PretokenizedCacheDownloadConfig, fetch_pretokenized_cache
from marin.processing.tokenize.tokenize import (
    HfTokenizeConfig,
    TokenizeConfig,
    TokenizeConfigBase,
    TokenizedCache,
)
from marin.processing.tokenize.tokenize import tokenize as _tokenize_fn


def _on(fn: Callable[..., object], resources: ResourceConfig | None) -> Callable[..., object]:
    """Run ``fn`` on Fray with ``resources`` (via :func:`remote`), or inline when None.

    Resources ride with the function, so they stay off the step node and out of the
    artifact fingerprint."""
    return remote(fn, resources=resources) if resources is not None else fn


def _looks_like_hf_id(source: str) -> bool:
    """A bare ``org/name`` HuggingFace id (one slash, not a filesystem/URL path)."""
    return source.count("/") == 1 and "://" not in source and not source.startswith("/")


def _resolve(prefix: str, path: str) -> str:
    """Make a prefix-relative raw path absolute; leave absolute/URL paths alone."""
    return prefix_join(prefix, path) if _is_relative_path(path) else path


def hf_download(
    name: str,
    *,
    hf_id: str,
    revision: str,
    version: str,
    urls_glob: Sequence[str] = (),
    pin: str | None = None,
    resources: ResourceConfig | None = None,
) -> ArtifactStep[Artifact]:
    """A HuggingFace-Hub dataset download as a raw-data handle.

    Wraps :func:`marin.datakit.download.huggingface.download_hf` into a handle that
    :func:`tokenized` (via ``raw=``) or :func:`marin.execution.lazy.apply` can depend on.
    ``urls_glob`` restricts which files in the repo are fetched (empty = all). ``pin``
    references an existing download at a fixed location instead of re-fetching it.
    """

    def build_config(ctx: StepContext) -> DownloadConfig:
        return DownloadConfig(
            hf_dataset_id=hf_id,
            revision=revision,
            hf_urls_glob=[*urls_glob],
            gcs_output_path=ctx.output_path,
            wait_for_completion=True,
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=Artifact,
        run=_on(download_hf, resources),
        build_config=build_config,
        override_path=pin,
    )


def raw_download(
    name: str,
    *,
    fn: Callable[[object], object],
    build_config: Callable[[StepContext], object],
    version: str,
    pin: str | None = None,
    resources: ResourceConfig | None = None,
) -> ArtifactStep[Artifact]:
    """A raw-data download as an ``ArtifactStep[Artifact]`` that :func:`tokenized` can depend on.

    The generic download builder for a source that is not a HuggingFace-Hub dataset (use
    :func:`hf_download` for that): ``fn(build_config(ctx))`` writes the download to
    ``ctx.output_path``. Returned as a raw :class:`~marin.execution.artifact.Artifact` (not a
    tokenized cache). ``pin`` references an existing download instead of re-fetching it.
    """
    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=Artifact,
        run=_on(fn, resources),
        build_config=build_config,
        override_path=pin,
    )


def tokenized(
    name: str,
    *,
    tokenizer: str,
    version: str,
    source: str | None = None,
    paths: Sequence[str] | None = None,
    raw: ArtifactStep[Artifact] | None = None,
    glob: str | None = None,
    validation: bool = False,
    pin: str | None = None,
    text_key: str = "text",
    sample_count: int | None = None,
    tags: Sequence[str] = (),
    resources: ResourceConfig | None = None,
) -> ArtifactStep[TokenizedCache]:
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

    def build_config(ctx: StepContext) -> TokenizeConfigBase:
        if source is not None and _looks_like_hf_id(source):
            return HfTokenizeConfig(
                id=source,
                cache_path=ctx.output_path,
                tokenizer=tokenizer,
                format=fmt,
                sample_count=sample_count,
                tags=[*tags],
            )
        if raw is not None:
            resolved = [f"{ctx.artifact_path(raw)}/{glob}"]
        elif paths is not None:
            resolved = [_resolve(ctx.prefix, p) for p in paths]
        else:
            resolved = [_resolve(ctx.prefix, source)]
        return TokenizeConfig(
            train_paths=[] if validation else resolved,
            validation_paths=resolved if validation else [],
            cache_path=ctx.output_path,
            tokenizer=tokenizer,
            format=fmt,
            sample_count=sample_count,
            tags=[*tags],
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=TokenizedCache,
        run=_on(_tokenize_fn, resources),
        build_config=build_config,
        deps=(raw,) if raw is not None else (),
        override_path=pin,
    )


def pretokenized(
    name: str,
    *,
    repo_id: str,
    tokenizer: str,
    version: str,
    revision: str | None = None,
    pin: str | None = None,
    tags: Sequence[str] = (),
    resources: ResourceConfig | None = None,
) -> ArtifactStep[TokenizedCache]:
    """A handle to an already-tokenized Levanter cache hosted on HuggingFace.

    ``build_config(ctx)`` downloads the HF dataset repo ``repo_id`` into ``ctx.output_path`` as
    a Levanter cache; the handle then reads as a ``TokenizedCache`` with no
    re-tokenization. Use it where a tokenizing :func:`tokenized` handle would be too
    slow — e.g. the fineweb-edu prebuilt subcaches. ``pin`` references an
    already-downloaded cache at an existing location instead of fetching it again.
    """

    def build_config(ctx: StepContext) -> PretokenizedCacheDownloadConfig:
        return PretokenizedCacheDownloadConfig(
            cache_path=ctx.output_path,
            tokenizer=tokenizer,
            hf_repo_id=repo_id,
            hf_revision=revision,
            tags=[*tags],
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=TokenizedCache,
        run=_on(fetch_pretokenized_cache, resources),
        build_config=build_config,
        override_path=pin,
    )


def _placeholder_component(cache_dir: str):
    """A fingerprint-time mixture component: just a cache-dir placeholder + constant format.

    At fingerprint time no record exists; the cache dir renders to ``name@version``, so the
    component carries the dataset's identity and nothing read from disk.
    """
    source = UrlDatasetSourceConfig(
        tags=[], train_urls=[], validation_urls=[], cache_dir=cache_dir, format=TextLmDatasetFormat()
    )
    return dataset_component(source)


def mixture(
    ctx: StepContext,
    train: Mapping[ArtifactStep[TokenizedCache], float],
    *,
    validation: Sequence[ArtifactStep[TokenizedCache]] = (),
    shuffle: bool | BlockShuffleConfig = DEFAULT_LM_DATA_SHUFFLE,
) -> LmDataConfig:
    """Assemble an ``LmDataConfig`` from dataset handles.

    ``train`` maps each handle to its mixture weight; ``validation`` handles are added at
    weight 0. The component key is the handle's ``name`` (two handles sharing a name are
    rejected). At run time each component is built from its ``TokenizedCache`` record
    (tokenizer/format/path), never from the producing recipe — so adopted and pinned caches
    work the same as freshly tokenized ones. At fingerprint time (no records yet) the data
    contribution is the sorted ``{name@version: weight}`` map; the tokenizer is determined by
    the chosen datasets and verified at run time. Call this inside a consumer's ``build_config``
    and pass the same handles as the step's ``deps`` so they materialize first.
    """
    handles = [*train, *validation]
    if not handles:
        raise ValueError("mixture needs at least one training or validation component")
    names = [dataset.name for dataset in handles]
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"mixture components are keyed by name, but these collide: {duplicates}")

    if ctx.is_fingerprint:
        components = {dataset.name: _placeholder_component(ctx.artifact_path(dataset)) for dataset in handles}
        weights = {dataset.name: weight for dataset, weight in train.items()}
        weights.update({dataset.name: 0.0 for dataset in validation})
        return LmDataConfig(
            components=components,
            train_weights=weights,
            tokenizer="<tokenizer>",
            cache_dir=None,
            shuffle=shuffle,
            permutation_type="feistel",
        )

    components = {}
    weights = {}
    tokenizers: set[str] = set()
    for dataset, weight in train.items():
        cache = ctx.resolved(dataset)
        components[dataset.name] = cache.as_component()
        weights[dataset.name] = weight
        tokenizers.add(cache.tokenizer)
    for dataset in validation:
        cache = ctx.resolved(dataset)
        components[dataset.name] = cache.as_component()
        weights[dataset.name] = 0.0
        tokenizers.add(cache.tokenizer)

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


def dataset_main(datasets: Mapping[str, ArtifactStep[TokenizedCache]]) -> None:
    """CLI entry point for a runnable dataset module.

    Use as the body of a module's ``if __name__ == "__main__":`` block, passing the
    module's family map keyed by subset (a single-corpus module passes a one-entry map).
    Building is opt-in so ``python -m experiments.datasets.<name>`` never starts a large
    tokenize by accident: by default it prints the lowered plan for the selected handles;
    ``--run`` builds them, and ``--only`` restricts a family to named keys.
    """

    @click.command()
    @click.option("--run", "build", is_flag=True, help="Build the handles (default: only print the plan).")
    @click.option("--only", help="Comma-separated subset keys to select (default: all).")
    @click.option("--max-concurrent", type=int, default=8, help="Max handles built concurrently.")
    def cli(build: bool, only: str | None, max_concurrent: int) -> None:
        selected = datasets
        if only is not None:
            keys = only.split(",")
            unknown = [key for key in keys if key not in datasets]
            if unknown:
                raise click.BadParameter(f"unknown keys {unknown}; available: {sorted(datasets)}", param_hint="--only")
            selected = {key: datasets[key] for key in keys}
        handles = list(selected.values())
        if not build:
            for handle in handles:
                click.echo(lower(handle))
            return
        run(*handles, max_concurrent=max_concurrent)

    cli()

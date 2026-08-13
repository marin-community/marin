# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the hero pretraining data stages by source name.

Every Datakit stage writes to a content-addressed path, so reading one back
normally means knowing which artifact version, tokenizer pin and upstream hash
produced it. This module holds that knowledge in one place: callers name a
source and a stage, and get a :class:`StepSpec` pointing at the data, including
the Harrier embeddings.

Every accessor returns a step that points at data which already exists and
refuses to execute. Pass one to :func:`marin.execution.artifact.read_artifact`,
or use it as a dependency of a step you do intend to run::

    from experiments.datakit import hero_data
    from marin.execution.artifact import read_artifact
    from marin.processing.tokenize.attributes import TokenizedAttrData

    step = hero_data.tokenized("stack-v3", hero_data.NEMOTRON_TOKENIZER)
    data = read_artifact(step.output_path, TokenizedAttrData)

:func:`normalized` and :func:`minhash` follow current code, so they track main
as the registry moves. :func:`tokenized` pins the artifact version instead,
because the tokenize hash includes one and it has changed under the runs that
produced this data: each tokenizer was applied to the whole registry in a single
fleet run, and each of those runs wrote a different version. The dedup stages
are pinned to a specific run outright.

All paths resolve against ``MARIN_PREFIX``. The hero data currently lives on
CoreWeave, so read it with ``MARIN_PREFIX=s3://marin-us-east-02a/marin``.
"""

import contextlib
import json
import os
import pathlib
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import NoReturn

from levanter.tokenizers import TokenizerBackend
from marin.datakit.sources import all_sources
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize.attributes import tokenize_attributes_step

from experiments.datakit.embeddings.harrier.pipeline import harrier_hash_attrs
from experiments.datakit.reference_pipeline import select_sources, zephyr_datakit_steps

_MARIN_PREFIX_ENV = "MARIN_PREFIX"

MANIFEST_PATH = pathlib.Path(__file__).with_name("hero_data_paths.json")

# The manifest records the paths as they resolve under the prefix that holds the
# hero data. It is not region-independent: ``materialize_ghalogs_step`` hashes the
# absolute ``source_path`` of its download, so the whole ghalogs/public chain
# re-keys per region and the same data sits at a different path in each one.
MANIFEST_PREFIX = "s3://marin-us-east-02a/marin"


@dataclass(frozen=True)
class TokenizerPin:
    """A tokenizer identity, and the artifact version its fleet run wrote."""

    name: str
    """HuggingFace tokenizer name."""

    revision: str
    """Immutable HF commit. Identity-only -- ``load_tokenizer`` takes no revision."""

    artifact_version: int
    """``TOKENIZED_ATTR_DATA_VERSION`` as of that run. Part of the tokenize hash."""


MARIN_TOKENIZER = TokenizerPin("marin-community/marin-tokenizer", "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2", 2)
NEMOTRON_TOKENIZER = TokenizerPin(
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "624ba927cfbef0427354998700de3d51173c8c04", 3
)

# The Focus Crawl was normalized (#8111) and tokenized after both fleet runs, by
# which point #8100 had bumped TOKENIZED_ATTR_DATA_VERSION to 4. It is the only
# source whose outputs sit at the current version, under either tokenizer.
_ARTIFACT_VERSION_OVERRIDES = {"common-crawl-focus-2026-22": 4}

# Pinned dedup runs. Both cover all 292 registered sources, and both key the
# focus crawl under its pre-#8111 extraction, so their attributes do not join
# against today's normalize for that one source. Exported because consumers
# also hash the run id into their own steps.
EXACT_DUPS_ID = "global_exact_dedup_af4c6c3e"
FUZZY_DUPS_ID = "dedup_709f5997"


def _refuse_to_run(output_path: str) -> NoReturn:
    """Fail loudly: these steps describe data that already exists."""
    raise AssertionError(
        f"hero_data steps point at existing data and must never execute. "
        f"Something asked to produce {output_path}, which would overwrite it."
    )


def _frozen_step(name: str, path: str) -> StepSpec:
    """A step pinned to a relative ``path`` that raises if a runner executes it.

    ``hash_id`` ignores ``override_output_path``, and a dependent's cache key is
    built from its deps' ``name_with_hash``. Carrying the pin in ``hash_attrs``
    is what makes repointing one of these invalidate the steps that consumed it.
    """
    return StepSpec(name=name, override_output_path=path, hash_attrs={"path": path}, fn=_refuse_to_run)


def _read_only(step: StepSpec) -> StepSpec:
    """The same step, stripped of its ability to run.

    Keeps the original identity, so ``output_path`` still resolves
    ``marin_prefix()`` lazily the way the pipeline does.
    """
    return replace(step, fn=_refuse_to_run)


def source_names() -> list[str]:
    """Every registered Datakit source name, sorted."""
    return sorted(all_sources())


def _normalize_step(source: str) -> StepSpec:
    return select_sources([source])[source]


def normalized(source: str) -> StepSpec:
    """Return the normalized dataset for ``source``, as current code resolves it."""
    return _read_only(_normalize_step(source))


def tokenized(source: str, tokenizer: TokenizerPin = NEMOTRON_TOKENIZER) -> StepSpec:
    """Return the tokenized attributes for ``source`` under ``tokenizer``."""
    step = tokenize_attributes_step(
        name=f"datakit/tokenize/{source}",
        train_normalize=_normalize_step(source),
        tokenizer=tokenizer.name,
        tokenizer_backend=TokenizerBackend.HF,
        tokenizer_revision=tokenizer.revision,
    )
    version = _ARTIFACT_VERSION_OVERRIDES.get(source, tokenizer.artifact_version)
    return _read_only(replace(step, hash_attrs={**step.hash_attrs, "artifact_version": version}))


def minhash(source: str) -> StepSpec:
    """Return the MinHash signatures for ``source``, keyed off its normalized output."""
    steps = zephyr_datakit_steps({source: _normalize_step(source)})
    return _read_only(steps.minhash[source])


def exact_dups() -> StepSpec:
    """Return the pinned global exact-duplicate attributes covering every source."""
    return _frozen_step("hero/exact_dups", f"datakit/{EXACT_DUPS_ID}")


def fuzzy_dups() -> StepSpec:
    """Return the pinned fuzzy-duplicate attributes covering every source."""
    return _frozen_step("hero/fuzzy_dups", f"datakit/{FUZZY_DUPS_ID}")


def harrier(source: str) -> StepSpec:
    """Return the Harrier embeddings for ``source``."""
    return StepSpec(
        name=f"datakit/embed/harrier/{source}",
        deps=[_normalize_step(source)],
        hash_attrs=harrier_hash_attrs(FUZZY_DUPS_ID),
        fn=_refuse_to_run,
    )


def all_paths() -> dict[str, str]:
    """Every hero data path, keyed ``<stage>/<source>`` for the per-source stages.

    ``tests/datakit/test_hero_data.py`` pins this against
    ``experiments/datakit/hero_data_paths.json``, so a change that moves a hero
    path fails there instead of silently repointing whoever reads it.
    """
    sources = select_sources(None)
    minhash_steps = zephyr_datakit_steps(sources).minhash
    paths = {
        "exact_dups": exact_dups().output_path,
        "fuzzy_dups": fuzzy_dups().output_path,
    }
    for source in sorted(sources):
        paths[f"normalized/{source}"] = _read_only(sources[source]).output_path
        paths[f"minhash/{source}"] = _read_only(minhash_steps[source]).output_path
        paths[f"tokenize.marin/{source}"] = tokenized(source, MARIN_TOKENIZER).output_path
        paths[f"tokenize.nemotron/{source}"] = tokenized(source, NEMOTRON_TOKENIZER).output_path
        paths[f"harrier/{source}"] = harrier(source).output_path
    return paths


@contextlib.contextmanager
def _pinned_prefix(prefix: str) -> Iterator[None]:
    """Bind ``MARIN_PREFIX`` for the block.

    ``DataConfig.resolved_root`` gives the environment variable precedence over
    every other source, including ``use_data_config``, so this is what it takes
    to pin the prefix for code that reads it lazily.
    """
    previous = os.environ.get(_MARIN_PREFIX_ENV)
    os.environ[_MARIN_PREFIX_ENV] = prefix
    try:
        yield
    finally:
        if previous is None:
            del os.environ[_MARIN_PREFIX_ENV]
        else:
            os.environ[_MARIN_PREFIX_ENV] = previous


def write_manifest() -> dict[str, str]:
    """Rewrite :data:`MANIFEST_PATH` with the paths as they resolve on CoreWeave.

    Pins :data:`MANIFEST_PREFIX` rather than reading the caller's, so the file is
    the same whatever the ambient config. Entries are stored relative to it.
    Returns what it wrote.
    """
    with _pinned_prefix(MANIFEST_PREFIX):
        relative = {key: path.removeprefix(f"{MANIFEST_PREFIX}/") for key, path in all_paths().items()}

    escaped = sorted(key for key, path in relative.items() if "://" in path)
    if escaped:
        raise ValueError(f"paths outside {MANIFEST_PREFIX} cannot be stored relative: {escaped}")

    MANIFEST_PATH.write_text(json.dumps(relative, indent=1, sort_keys=True) + "\n")
    return relative


if __name__ == "__main__":
    written = write_manifest()
    print(f"wrote {len(written)} paths to {MANIFEST_PATH}")

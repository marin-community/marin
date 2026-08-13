# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the hero pretraining data stages by source name.

Every Datakit stage writes to a content-addressed path, so reading one back
normally means knowing which artifact version, tokenizer pin and upstream hash
produced it. This module holds that knowledge in one place: callers name a
source and a stage, and get a :class:`StepSpec` pointing at the data.

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
are pinned to a specific run outright, and :func:`quality` is pinned to the
scorer and calibration that produced the scores.

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


@dataclass(frozen=True)
class QualityPin:
    """The scorer that produced the quality scores, and the calibration that buckets them."""

    tokenizer: TokenizerPin
    """The tokenization the scores were computed from. Its step hash is part of
    the scores' path, so this is what resolves them."""

    model_path: str
    """Folded scorer artifact. Provenance only -- the weights determined every
    score but are never read again, so moving them is not a change."""

    model_sha256: str
    """Digest of the folded artifact. Part of the quality hash: this, not the
    path, identifies the scorer."""

    calibration_path: str
    """Bucket edges, as ``apply_calibration`` knots. Read by any consumer that
    buckets, so the location is part of the contract. Part of the quality hash."""

    calibration_sha256: str
    """Digest of that file. Part of the quality hash, so a refit is a change
    whether or not it lands on a new path."""

    version: int
    """Bumped when the score data changes under an unchanged path, as when
    scores for the currently-undeduplicated documents are merged in. Part of
    the quality hash."""


# One scorer produced the corpus scores. Unlike the tokenizers, it does not
# appear in the output path, so it is a constant rather than a parameter:
# a second scorer would need its own output tree, not a second pin here.
#
# ``model_sha256`` digests the artifact directory, which is not a single file.
# The recipe: list every file under ``model_path`` recursively, take each path
# relative to that root, sort those relative paths bytewise, and fold them into
# one sha256 as ``rel.encode() + b"\0" + sha256(content).digest()`` per file, in
# that order. ``calibration_sha256`` is a plain sha256 of the one file.
NEMOTRON_88K = QualityPin(
    tokenizer=NEMOTRON_TOKENIZER,
    model_path="user/muchanem/quality_scores_run/model/nemotron_88k_folded",
    model_sha256="e1be09c903bf7a926046bac97d40db050ca399fbb336c7541df42dc8cf6eda10",
    calibration_path="user/muchanem/quality_scores_run/model/nemotron_88k_folded/calib_bme.json",
    calibration_sha256="1a1dcfd31c20d9f3879878d617f0f8fb0b6898c4445885ffae29ebea63738fd8",
    version=1,
)

# The Focus Crawl was scored from the same pre-#8111 extraction the dedup runs
# key it under, so its scores sit beside a tokenization that current code no
# longer resolves and its ids do not join against today's normalize. Every other
# scored source sits at exactly the leaf :func:`tokenized` computes.
_QUALITY_LEAF_OVERRIDES = {"common-crawl-focus-2026-22": "common-crawl-focus-2026-22_fe127aa9"}


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


def quality(source: str, quality_model: QualityPin = NEMOTRON_88K) -> StepSpec:
    """Return the quality scores for ``source``.

    Raw sigmoid, one row per scored document, keyed by the same content-derived
    ``id`` as every other stage. Bucket with the edges at
    ``quality_model.calibration_path``; that remap is monotonic, so it changes
    bucketing but never ranking.

    The scores sit beside the tokenization they were computed from, so their
    leaf carries that step's hash and is resolved through :func:`tokenized`.
    """
    leaf = _QUALITY_LEAF_OVERRIDES.get(source)
    if leaf is None:
        leaf = tokenized(source, quality_model.tokenizer).name_with_hash.removeprefix("datakit/tokenize/")
    step = _frozen_step(f"hero/quality/{source}", f"datakit/quality-scores/{leaf}")
    return replace(
        step,
        hash_attrs={
            **step.hash_attrs,
            "model": quality_model.model_sha256,
            "calibration": quality_model.calibration_sha256,
            "version": quality_model.version,
        },
    )


def all_paths() -> dict[str, str]:
    """Every hero data path, keyed ``<stage>/<source>`` for the per-source stages.

    ``tests/datakit/test_hero_data.py`` pins this against
    ``experiments/datakit/hero_data_paths.json``, so a change that moves a hero
    path fails there instead of silently repointing whoever reads it.

    Emitted per stage for every source, including the ``quality`` entries of the
    150 sources that have no scores yet. This is a drift pin, not a catalogue of
    what exists: the point is that a path cannot move unnoticed, and a quality
    path moves exactly when the tokenize path it is derived from moves.
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
        paths[f"quality/{source}"] = quality(source, NEMOTRON_88K).output_path
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

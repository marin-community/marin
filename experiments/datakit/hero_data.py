# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the hero pretraining data stages by source name.

Every Datakit stage writes to a content-addressed path, so reading one back
normally means knowing which artifact version, tokenizer pin and upstream hash
produced it. This module holds that knowledge in one place: callers name a
source and a stage, and get a fixed reference to the data.

Most accessors return a step that points at data which already exists and
refuses to execute. Pass one to :func:`marin.execution.artifact.read_artifact`,
or use it as a dependency of a step you do intend to run::

    from experiments.datakit import hero_data
    from marin.execution.artifact import read_artifact
    from marin.processing.tokenize.attributes import TokenizedAttrData

    step = hero_data.tokenized("stack-v3", hero_data.NEMOTRON_TOKENIZER)
    data = read_artifact(step.output_path, TokenizedAttrData)

:func:`harrier` returns a path string. It reads the fixed source-to-path
map in ``hero_data_emb_paths.json`` and adds the active Marin prefix.

:func:`normalized` and :func:`minhash` follow current code, so they track main
as the registry moves. :func:`tokenized` pins the artifact version instead,
because the tokenize hash includes one and it has changed under the runs that
produced this data: each tokenizer was applied to the whole registry in a single
fleet run, and each of those runs wrote a different version. The dedup stages
and domain cluster assignment are pinned to specific runs outright.
:func:`decontam` is pinned per source, like :func:`harrier`, because its step
folds in an eval bloom and a drop-set stage that have both moved since the run.

A stage whose producing job has not finished raises :class:`PendingRegistration`
when something asks for it. :func:`experiments.datakit.produce_store.pending`
collects those into one list, so the store entry point can say everything it is
waiting on at once instead of failing on the first one.

All paths resolve against ``MARIN_PREFIX``. CoreWeave Datakit has one storage
root, ``s3://marin-us-east-02a/marin``; use it regardless of worker placement.
"""

import contextlib
import json
import os
import pathlib
from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import cache
from typing import NoReturn

from levanter.tokenizers import TokenizerBackend
from marin.datakit.sources import all_sources
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize.attributes import tokenize_attributes_step
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.cluster.domain.v0.assign import assign_hash_attrs
from experiments.datakit.reference_pipeline import select_sources, zephyr_datakit_steps

_MARIN_PREFIX_ENV = "MARIN_PREFIX"


@cache
def manifest_path() -> pathlib.Path:
    """Return the path to the generated Hero Data manifest."""
    return pathlib.Path(__file__).with_name("hero_data_paths.json")


@cache
def harrier_paths_path() -> pathlib.Path:
    """Return the path to the complete Harrier path map."""
    return pathlib.Path(__file__).with_name("hero_data_emb_paths.json")


@cache
def harrier_paths() -> dict[str, str]:
    """Load the complete Harrier path map."""
    return json.loads(harrier_paths_path().read_text())


@cache
def decon_paths_path() -> pathlib.Path:
    """Return the path to the per-source decontamination path map."""
    return pathlib.Path(__file__).with_name("hero_data_decon_paths.json")


class PendingRegistration(LookupError):
    """A hero stage the run needs, whose producing job has not been registered yet.

    Raised at the point of use rather than at import, so a caller that does not
    touch the stage still works, and one that does gets told what to do about it
    instead of a path that resolves to nothing.
    """

    def __init__(self, stage: str, remedy: str) -> None:
        super().__init__(f"{stage} is not registered in hero_data yet: {remedy}")
        self.stage = stage
        self.remedy = remedy


@cache
def decon_paths() -> dict[str, str]:
    """Load the per-source decontamination path map."""
    path = decon_paths_path()
    if not path.exists():
        raise PendingRegistration(
            "decontamination",
            f"run experiments/datakit/scripts/register_decontam.py to write {path.name} "
            "from the completed decontamination run",
        )
    return json.loads(path.read_text())


# The manifest records paths relative to the sole CoreWeave Datakit root.
# Regeneration pins this prefix because some step hashes include resolved paths.
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

DOMAIN_CLUSTER_ASSIGNMENT_PATH = "datakit/cluster/domain/v1/harrier-all-sources-10m/train_fe81b456"

# The knobs the assign stage hashes. The pinned model ships one coarse view,
# so widening ``K_VIEWS`` asks for a lookup that does not exist.
DOMAIN_ASSIGN_K_TRAIN = 5000
DOMAIN_ASSIGN_K_VIEWS = (40,)
DOMAIN_ASSIGN_BATCH_SIZE = 4096

# Pinned dedup runs. Both cover all 292 registered sources, and both key the
# focus crawl under its pre-#8111 extraction, so their attributes do not join
# against today's normalize for that one source. Exported because consumers
# also hash the run id into their own steps.
EXACT_DUPS_ID = "global_exact_dedup_af4c6c3e"
FUZZY_DUPS_ID = "dedup_709f5997"
VERIFIED_FUZZY_DUPS_PATH = "datakit/verify_fuzzy_dups_c757e4f0"

# The key the two candidate runs filed the focus crawl under: its jusText
# extraction, read as a finished source before #8111 sent it through
# normalize_step. Fuzzy verification already repacked its half (#8237) and keys
# all 292 sources the way normalize keys them today; the exact marks still carry
# this key, and :mod:`experiments.datakit.repack_exact_dups` moves them.
FOCUS_SOURCE_NAME = "common-crawl-focus-2026-22"
LEGACY_FOCUS_SOURCE_KEY = "data/datakit/normalized/common_crawl_focus_2026_22_ed4b8bc9/outputs/main"


@dataclass(frozen=True)
class QualityPin:
    """A quality scorer's identity: the bytes it reads and the tokens it eats."""

    name: str
    """Value written into the score rows' ``model`` column."""

    model_sha256: str
    """Digest over the deployed model directory, as
    ``score_corpus.model_dir_sha256`` computes it: every file under the root,
    addressed relative to it, folded in bytewise path order. Scoring refuses to
    write from a directory that digests to anything else, so the path's claim
    about which model produced the scores under it is checked, not asserted."""

    calibration_sha256: str
    """Digest over the calibration file, by the same recipe over one file."""

    tokenizer: TokenizerPin
    """Which tokenization the scorer reads. Folded in structurally: :func:`quality`
    makes :func:`tokenized` a dependency rather than naming its leaf."""

    version: int
    """Bump to rescore the corpus under an unchanged model and calibration."""


NEMOTRON_88K = QualityPin(
    name="nemotron88k_v1",
    model_sha256="e1be09c903bf7a926046bac97d40db050ca399fbb336c7541df42dc8cf6eda10",
    calibration_sha256="1a1dcfd31c20d9f3879878d617f0f8fb0b6898c4445885ffae29ebea63738fd8",
    tokenizer=NEMOTRON_TOKENIZER,
    version=1,
)

# Where NEMOTRON_88K's deployed bytes sit. Held as a path beside the pin rather
# than inside it: the digests are what make the claim checkable, and scoring
# refuses to run against a directory that digests to anything else, so this only
# has to say where to look.
QUALITY_MODEL_DIR = "user/muchanem/quality_scores_run/model/nemotron_88k_folded"
QUALITY_CALIBRATION_FILE = "calib_bme.json"


def quality_calibration(quality_model: QualityPin = NEMOTRON_88K) -> str:
    """Return the calibration file whose knots cut ``quality_model``'s scores."""
    if quality_model != NEMOTRON_88K:
        raise KeyError(f"no calibration path recorded for {quality_model.name!r}")
    return prefix_join(marin_prefix(), f"{QUALITY_MODEL_DIR}/{QUALITY_CALIBRATION_FILE}")


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


def quality(source: str, quality_model: QualityPin = NEMOTRON_88K) -> StepSpec:
    """Return the quality scores for ``source`` under ``quality_model``.

    Unlike the other accessors this one takes no pinned path: its output sits at
    ``name_with_hash`` like any ordinary step, so the scorer is *in* the path.
    The scorer's identity reaches the hash two ways, and it needs both. The model
    and calibration digests go in ``hash_attrs``, which covers the bytes the
    scorer reads; the tokenization goes in as a dependency, so ``hash_id`` folds
    it through ``dep_names`` rather than through string surgery on a leaf name.

    The earlier layout derived the score path from the tokenize leaf alone, which
    left the scorer invisible: two pins over one tokenization resolved to the same
    directory, so a step could claim one model and read another's bytes. Here two
    pins that differ anywhere -- model, calibration, tokenizer or version -- differ
    in ``hash_id``, and so cannot collide.
    """
    return StepSpec(
        name=f"datakit/quality/{source}",
        deps=[tokenized(source, quality_model.tokenizer)],
        hash_attrs={
            "model": quality_model.name,
            "model_sha256": quality_model.model_sha256,
            "calibration_sha256": quality_model.calibration_sha256,
            "version": quality_model.version,
        },
        fn=_refuse_to_run,
    )


def exact_dups() -> StepSpec:
    """Return the pinned global exact-duplicate attributes covering every source."""
    return _frozen_step("hero/exact_dups", f"datakit/{EXACT_DUPS_ID}")


def fuzzy_dups() -> StepSpec:
    """Return the pinned fuzzy-duplicate *candidate* attributes covering every source."""
    return _frozen_step("hero/fuzzy_dups", f"datakit/{FUZZY_DUPS_ID}")


def verified_fuzzy_dups() -> StepSpec:
    """Return the pinned verified fuzzy-duplicate attributes covering every source."""
    return _frozen_step("hero/verified_fuzzy_dups", VERIFIED_FUZZY_DUPS_PATH)


def decontam(source: str) -> StepSpec:
    """Return the decontamination marks for ``source``.

    Recorded rather than recomputed, like :func:`harrier`. The decon step folds
    the eval bloom and the cross-source drop sets into its identity through its
    dependencies, and both have moved since the run that produced this data, so
    rebuilding the step from current code addresses a directory that does not
    exist. ``scripts/register_decontam.py`` writes the map by reading the run.
    """
    paths = decon_paths()
    if source not in paths:
        raise PendingRegistration(
            f"decontamination for {source!r}",
            f"the map in {decon_paths_path().name} covers {len(paths)} sources but not this one; "
            "rerun register_decontam.py against a run that includes it",
        )
    return _frozen_step(f"hero/decontam/{source}", paths[source])


def domain_cluster_assignment() -> StepSpec:
    """Return the pinned Harrier domain cluster assignment."""
    return _frozen_step("hero/domain_cluster_assignment", DOMAIN_CLUSTER_ASSIGNMENT_PATH)


def assigned_clusters(source: str) -> StepSpec:
    """Return the cluster each document of ``source`` was assigned to.

    Named apart from :func:`domain_cluster_assignment`, which returns the
    centroid model these were assigned against, not the assignments.

    Unlike :func:`harrier`, this needs no recorded path: the assign step's
    identity is the frozen model plus the knobs above, so the output path
    recomputes exactly. Repointing the model therefore moves these too.
    """
    model = domain_cluster_assignment()
    return _read_only(
        StepSpec(
            name=f"datakit/cluster_assign/harrier/{source}",
            deps=[model],
            hash_attrs=assign_hash_attrs(
                model.name_with_hash, DOMAIN_ASSIGN_K_TRAIN, DOMAIN_ASSIGN_K_VIEWS, DOMAIN_ASSIGN_BATCH_SIZE
            ),
        )
    )


def harrier(source: str) -> str:
    """Return the fixed complete Harrier path for ``source``."""
    return prefix_join(marin_prefix(), harrier_paths()[source])


def all_paths() -> dict[str, str]:
    """Every hero data path, keyed ``<stage>/<source>`` for the per-source stages.

    ``tests/datakit/test_hero_data.py`` pins this against
    ``experiments/datakit/hero_data_paths.json``, so a change that moves a hero
    path fails there instead of silently repointing whoever reads it.
    """
    sources = select_sources(None)
    minhash_steps = zephyr_datakit_steps(sources).minhash
    paths = {
        "domain_cluster_assignment": domain_cluster_assignment().output_path,
        "exact_dups": exact_dups().output_path,
        "fuzzy_dups": fuzzy_dups().output_path,
        "verified_fuzzy_dups": verified_fuzzy_dups().output_path,
    }
    for source in sorted(sources):
        paths[f"normalized/{source}"] = _read_only(sources[source]).output_path
        paths[f"minhash/{source}"] = _read_only(minhash_steps[source]).output_path
        paths[f"tokenize.marin/{source}"] = tokenized(source, MARIN_TOKENIZER).output_path
        paths[f"tokenize.nemotron/{source}"] = tokenized(source, NEMOTRON_TOKENIZER).output_path
        paths[f"harrier/{source}"] = harrier(source)
        paths[f"cluster_assign/{source}"] = assigned_clusters(source).output_path
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
    """Rewrite :func:`manifest_path` with the paths as they resolve on CoreWeave.

    Pins :data:`MANIFEST_PREFIX` rather than reading the caller's, so the file is
    the same whatever the ambient config. Entries are stored relative to it.
    Returns what it wrote.
    """
    with _pinned_prefix(MANIFEST_PREFIX):
        relative = {key: path.removeprefix(f"{MANIFEST_PREFIX}/") for key, path in all_paths().items()}

    escaped = sorted(key for key, path in relative.items() if "://" in path)
    if escaped:
        raise ValueError(f"paths outside {MANIFEST_PREFIX} cannot be stored relative: {escaped}")

    manifest_path().write_text(json.dumps(relative, indent=1, sort_keys=True) + "\n")
    return relative


if __name__ == "__main__":
    written = write_manifest()
    print(f"wrote {len(written)} paths to {manifest_path()}")

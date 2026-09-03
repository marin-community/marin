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

:func:`normalized` and :func:`minhash` follow current code, so they track main
as the registry moves. :func:`tokenized` pins the artifact version instead,
because the tokenize hash includes one and it has changed under the runs that
produced this data: each tokenizer was applied to the whole registry in a single
fleet run, and each of those runs wrote a different version. The dedup stages
and domain cluster assignment are pinned to specific runs outright, as are the
three leaves whose producers ran from branches: :func:`harrier`,
:func:`fusion_scores` and :func:`content_type` read fixed source-to-path maps
from JSON files beside this module. :func:`quality` is the bucket step over the
pinned scores and types, so its path recomputes from that step's identity.

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

from experiments.datakit.cluster.domain.v0.assign import assign_hash_attrs
from experiments.datakit.cluster.quality.fast_transformer.bucket import quality_step
from experiments.datakit.cluster.quality.fast_transformer.quality_model import QualityPin
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
def fusion_score_paths_path() -> pathlib.Path:
    """Return the path to the fusion score path map."""
    return pathlib.Path(__file__).with_name("hero_data_fusion_score_paths.json")


@cache
def fusion_score_paths() -> dict[str, str]:
    """Load the fusion score path map."""
    return json.loads(fusion_score_paths_path().read_text())


@cache
def content_type_paths_path() -> pathlib.Path:
    """Return the path to the content-type path map."""
    return pathlib.Path(__file__).with_name("hero_data_content_type_paths.json")


@cache
def content_type_paths() -> dict[str, str]:
    """Load the content-type path map."""
    return json.loads(content_type_paths_path().read_text())


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


# The fusion quality scorer: Nemotron-tokenized text plus the Harrier document
# embedding. ``model_path`` holds the folded checkpoint, its remap and meta, and
# the per-type calibration; the digests are what a step checks before it writes
# under a path naming this pin. The tokenizer is the corpus tokenizer whose ids the
# scorer reads -- its vocabulary is shared with the Nemotron-Flash-1B tokenizer the
# checkpoint was trained on.
NEMOTRON_88K = QualityPin(
    name="nemotron88k_v1",
    model_path="datakit/models/quality/nemotron_88k",
    model_sha256="453745d4e06854eb8b9545f3014a8c5b59ad3a3072a18d1e26e0b916ca393196",
    calibration_sha256="b89b7b782e606394fd341e7705d438521e96fcb553f769c6c1dd520331da5758",
    tokenizer=NEMOTRON_TOKENIZER.name,
)


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


def fusion_scores(source: str, quality_model: QualityPin = NEMOTRON_88K) -> StepSpec:
    """Return the pinned fusion scores for ``source``: one raw score per document.

    Written by the :data:`NEMOTRON_88K` run of August 14, 2026 over all 292 sources
    from the Nemotron tokenization and the Harrier embeddings; the tokenization it
    read has since been deleted, so the leaves are pinned by path. The producer of
    record is :func:`score_fusion.fusion_score_step`, which tokenizes the normalized
    text itself; a rerun lands at that step's own identity and repoints this map.
    """
    if quality_model.model_sha256 != NEMOTRON_88K.model_sha256:
        raise ValueError(
            f"the pinned fusion scores were written by {NEMOTRON_88K.name}; score the corpus under "
            f"{quality_model.name} with run.py --stage score before bucketing it"
        )
    return _frozen_step(f"hero/fusion_scores/{source}", fusion_score_paths()[source])


def content_type(source: str) -> StepSpec:
    """Return the pinned predicted content types for ``source``.

    One row per document from the ``domain_mlp_v1`` classifier over the Harrier
    embeddings, written beside the fusion scores in their row order. The calibration
    in :data:`NEMOTRON_88K` carries one curve per predicted type, which is what
    :func:`quality` applies.
    """
    return _frozen_step(f"hero/content_type/{source}", content_type_paths()[source])


def quality(source: str, quality_model: QualityPin = NEMOTRON_88K) -> StepSpec:
    """Return the store-ready quality dataset for ``source``.

    The bucket step over :func:`fusion_scores` and :func:`content_type`: one row per
    normalized document, in the normalized order, with ``raw_score``, the per-type
    calibrated ``score`` and ``quality_bucket``. Its identity is the step's own, so
    a refit calibration or a repointed input moves the path.
    """
    return _read_only(
        quality_step(
            name=f"datakit/quality/{source}",
            source=source,
            normalized=_normalize_step(source),
            scores=fusion_scores(source, quality_model),
            content_type=content_type(source),
            quality_model=quality_model,
        )
    )


def exact_dups() -> StepSpec:
    """Return the pinned global exact-duplicate attributes covering every source."""
    return _frozen_step("hero/exact_dups", f"datakit/{EXACT_DUPS_ID}")


def fuzzy_dups() -> StepSpec:
    """Return the pinned fuzzy-duplicate attributes covering every source."""
    return _frozen_step("hero/fuzzy_dups", f"datakit/{FUZZY_DUPS_ID}")


def verified_fuzzy_dups() -> StepSpec:
    """Return the pinned verified fuzzy-duplicate attributes covering every source."""
    return _frozen_step("hero/verified_fuzzy_dups", VERIFIED_FUZZY_DUPS_PATH)


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


def harrier(source: str) -> StepSpec:
    """Return the pinned complete Harrier embeddings for ``source``."""
    return _frozen_step(f"hero/harrier/{source}", harrier_paths()[source])


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
        paths[f"harrier/{source}"] = harrier(source).output_path
        paths[f"cluster_assign/{source}"] = assigned_clusters(source).output_path
        paths[f"fusion_scores/{source}"] = fusion_scores(source).output_path
        paths[f"content_type/{source}"] = content_type(source).output_path
        paths[f"quality/{source}"] = quality(source).output_path
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

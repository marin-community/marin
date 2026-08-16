# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Produce the hero (cluster x quality) store from the registered Datakit stages.

Every input the store joins is already built. This entry point names those
artifacts through :mod:`experiments.datakit.hero_data` and hands ``StepRunner``
one terminal, ``datakit/store``. Nothing upstream is rebuilt: the dependencies
are read-only steps that refuse to execute, so a mis-registered pin fails on the
cache check instead of quietly starting a fleet run over a stage that already
exists.

    per source: tokenize x decontam x cluster_assign x quality
    globally:   exact dups, verified fuzzy dups
    -> build_clustered_store -> one Levanter cache per populated bucket

One stage does run: the focus crawl's exact-duplicate marks predate #8111 and
are addressed by the jusText extraction's shard layout, so
:mod:`experiments.datakit.repack_exact_dups` moves them onto the current
normalize first. It is a small job beside the store, and skipping it is not a
neutral choice -- sparse duplicate attributes read a layout mismatch as "no
duplicates", so the focus crawl would silently keep its exact duplicates.

Check the graph before submitting anything::

    uv run python -m experiments.datakit.produce_store --preflight

``--preflight`` resolves every pin, reads each artifact, and replays the
alignment the store itself demands, on a sample of shards per source. It costs a
few thousand small reads and no cluster time. The store's own checks run after
the fleet is already up, so a source whose shards do not line up is much cheaper
to find here.

Submit in the CoreWeave data region, where the inputs live::

    uv run iris --cluster=marin job run --no-wait \\
        --target-cluster cw-us-east-02a --priority production \\
        --cpu 2 --memory 8g --enable-extra-resources \\
        -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python -m experiments.datakit.produce_store \\
            --max-workers 48 --task-count 192

Sizing is the caller's call and the defaults are deliberately not a full-fleet
shape. ``--task-count`` sets how many map tasks the input shards are dealt into,
and each task holds one worker of ``--worker-cpu`` / ``--worker-ram`` /
``--worker-disk`` for its lifetime. ``--max-workers`` bounds how many run at
once, so ``max_workers x worker_cpu`` has to fit the CPU pool of the cluster you
target, and a task's local disk has to hold that task's spill: partitioned token
runs land on local SSD before each bucket cache is written. Ask for close to a
whole node and Kueue will not admit the gang at all.

Not every stage is registered yet. Ask what is missing::

    uv run python -m experiments.datakit.produce_store --pending

``--pending`` covers stages with no pin. ``--preflight`` covers the rest, and
reports separately on a stage that holds complete data the runner will not
serve because nothing marked it succeeded -- which is where the rescored quality
stage sits, since ``score_corpus`` writes its shards without going through
``StepRunner`` -- as does the content-type job. ``scripts/seal_derived_stage.py``
writes that marker for the sources whose output is complete.
"""

import argparse
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.decon import DeconAttributes
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner, step_is_built
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.verify_fuzzy_dups import VerifiedFuzzyDupsAttrData
from marin.processing.tokenize.attributes import TokenizedAttrData
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit import hero_data
from experiments.datakit.cluster.domain.v0.assign import AssignmentAttrData
from experiments.datakit.cluster.quality.fast_transformer.artifact import calibration_edges_by_content_type
from experiments.datakit.global_exact_dedup import GlobalExactDedupData
from experiments.datakit.reference_pipeline import SPLIT, StoreConfig
from experiments.datakit.repack_exact_dups import repack_exact_dups_source
from experiments.datakit.store.datakit_store import (
    ClusteredStoreData,
    build_clustered_store,
)

logger = logging.getLogger(__name__)

STORE_STEP_NAME = "datakit/store"

# The K the store partitions on. The pinned Harrier model ships one coarse view
# (``hero_data.DOMAIN_ASSIGN_K_VIEWS``), so this or ``DOMAIN_ASSIGN_K_TRAIN`` are
# the only widths an assignment artifact actually materialized a column for.
CLUSTER_VIEW = 40

# Bumped when this entry point changes what the store contains for a fixed set of
# inputs. Kept apart from the reference pipeline's own store version: the two
# build the same artifact but address their inputs differently, and a bump on one
# side must not silently invalidate the other's cache.
STORE_VERSION = 1

# The focus-crawl exact-duplicate repack. Cheap next to the store: one pass over
# the ``id`` column of a 36.3M-document extraction and its sparse marks.
DEFAULT_REPACK_MAX_WORKERS = 64
DEFAULT_REPACK_WORKER = ResourceConfig(cpu=4, ram="16g", disk="16g")

# What a correct focus-crawl repack emits. Global exact dedup marks every copy of
# an id but the canonical one, so summing ``f - 1`` over the source is what
# normalize independently reported as ``duplicate_records_out`` -- 14,316,834
# against 14,316,837 marks. Three documents of the focus crawl duplicate
# something outside it, and those three marks are the whole surviving set.
FOCUS_EXPECTED_MARKS_KEPT = 3


@dataclass(frozen=True)
class SourceStages:
    """The four per-source artifacts the store joins for one source."""

    tokenize: StepSpec
    decontam: StepSpec
    cluster_assign: StepSpec
    quality: StepSpec
    content_type: StepSpec
    """Which calibration the score is cut against. Not optional: cutting a mixed
    corpus at the content-blind default measurably promotes code and math."""

    def steps(self) -> list[StepSpec]:
        return [self.tokenize, self.decontam, self.cluster_assign, self.quality, self.content_type]


@dataclass(frozen=True)
class StoreInputs:
    """Every step the store reads, resolved from :mod:`hero_data`."""

    per_source: dict[str, SourceStages]
    exact_dups: StepSpec
    """What the store reads. The repack step when this run repacks, else the pin."""

    exact_dups_pin: StepSpec
    """The pinned dedup run. The same step as :attr:`exact_dups` when nothing is repacked."""

    repacked_sources: tuple[str, ...]
    """Sources whose exact marks :attr:`exact_dups` moves onto their current normalize."""

    verified_fuzzy_dups: StepSpec
    quality_model: hero_data.QualityPin

    def steps(self) -> list[StepSpec]:
        """Every dependency of the store step."""
        per_source = [step for stages in self.per_source.values() for step in stages.steps()]
        return [*per_source, self.exact_dups, self.verified_fuzzy_dups]

    def registered_steps(self) -> list[StepSpec]:
        """The inputs that must already exist, which is all of them but the repack."""
        per_source = [step for stages in self.per_source.values() for step in stages.steps()]
        return [*per_source, self.exact_dups_pin, self.verified_fuzzy_dups]


@dataclass(frozen=True)
class Pending:
    """A stage the store needs that nobody has registered yet."""

    stage: str
    remedy: str


def pending(sources: list[str] | None = None) -> list[Pending]:
    """Return every hero stage the store needs that is not registered yet.

    Collected rather than raised, so a caller waiting on two jobs learns about
    all of them from one run instead of discovering the second only after the
    first lands.

    Scoped to ``sources``, because decontamination lands per source: a run over
    the sources that already have it is exactly how the store gets exercised
    before the rest finish, and a registry-wide probe would refuse it.
    """
    names = hero_data.source_names() if sources is None else sources
    probes: list[Callable[[], object]] = [hero_data.verified_fuzzy_dups]
    probes += [lambda n=name: hero_data.decontam(n) for name in names]
    probes += [lambda n=name: hero_data.content_type(n) for name in names]
    missing: list[Pending] = []
    seen: set[str] = set()
    for probe in probes:
        try:
            probe()
        except hero_data.PendingRegistration as unregistered:
            if unregistered.stage not in seen:
                seen.add(unregistered.stage)
                missing.append(Pending(unregistered.stage, unregistered.remedy))
    return missing


def store_inputs(
    sources: list[str] | None = None,
    *,
    quality_model: hero_data.QualityPin = hero_data.NEMOTRON_88K,
) -> StoreInputs:
    """Resolve the store's inputs for ``sources`` (``None`` selects every source).

    ``quality_model`` scored a different tokenization than the corpus carries,
    and that is fine: a score is a property of the document, not of how it was
    tokenized, and the store joins the two by document id. Both leaves derive
    from the same normalize, so they share shard names and document order. The
    store checks that per shard rather than taking it on trust.
    """
    names = hero_data.source_names() if sources is None else sorted(sources)
    unknown = sorted(set(names) - set(hero_data.source_names()))
    if unknown:
        raise KeyError(f"unknown sources {unknown}")

    per_source = {
        name: SourceStages(
            tokenize=hero_data.tokenized(name),
            decontam=hero_data.decontam(name),
            cluster_assign=hero_data.assigned_clusters(name),
            quality=hero_data.quality(name, quality_model),
            content_type=hero_data.content_type(name),
        )
        for name in names
    }
    pin = hero_data.exact_dups()
    repacked = (hero_data.FOCUS_SOURCE_NAME,) if hero_data.FOCUS_SOURCE_NAME in per_source else ()
    return StoreInputs(
        per_source=per_source,
        exact_dups=build_focus_exact_repack_step(pin) if repacked else pin,
        exact_dups_pin=pin,
        repacked_sources=repacked,
        verified_fuzzy_dups=hero_data.verified_fuzzy_dups(),
        quality_model=quality_model,
    )


def build_focus_exact_repack_step(
    exact_dups: StepSpec,
    *,
    max_workers: int = DEFAULT_REPACK_MAX_WORKERS,
    worker_resources: ResourceConfig = DEFAULT_REPACK_WORKER,
) -> StepSpec:
    """Return the exact-duplicate artifact with the focus crawl moved onto its normalize.

    The only stage of this graph that runs anything besides the store. Both
    pinned dedup runs read the focus crawl's jusText extraction before #8111 sent
    it through ``normalize_step``, so their marks are addressed by shard names the
    current source does not use. Fuzzy verification repacks its own candidates
    (#8237); the exact marks have no such pass, and left alone they resolve to
    absent files, which sparse duplicate attributes read as "no duplicates".

    Not a hero pin, because the artifact does not exist yet: repointing
    ``EXACT_DUPS_ID`` or renormalizing the focus crawl moves this step, and the
    store's identity follows it.
    """
    focus = hero_data.normalized(hero_data.FOCUS_SOURCE_NAME)
    return StepSpec(
        name=f"datakit/repack_exact_dups/{hero_data.FOCUS_SOURCE_NAME}",
        deps=[exact_dups, focus],
        hash_attrs={
            "legacy_source_key": hero_data.LEGACY_FOCUS_SOURCE_KEY,
            "v": 1,
        },
        fn=lambda output_path: repack_exact_dups_source(
            exact_dups=read_artifact(exact_dups.output_path, GlobalExactDedupData),
            legacy_source_key=hero_data.LEGACY_FOCUS_SOURCE_KEY,
            normalized=read_artifact(focus.output_path, NormalizedData),
            output_path=output_path,
            max_workers=max_workers,
            worker_resources=worker_resources,
        ),
    )


def _restrict_to_sources(
    exact_dups: GlobalExactDedupData,
    verified: VerifiedFuzzyDupsAttrData,
    source_keys: set[str],
) -> tuple[GlobalExactDedupData, VerifiedFuzzyDupsAttrData]:
    """Drop dedup entries for sources outside this run.

    Both dedup artifacts cover the whole registry, and the store requires their
    source sets to *equal* the set it is building over, not contain it. A subset
    run is still correctly deduplicated: the marks in the attribute directories
    that survive were computed against every source, so a document dropped for
    matching one that is not in this run stays dropped.
    """
    for label, artifact in (("exact_dups", exact_dups), ("verified_fuzzy_dups", verified)):
        missing = sorted(source_keys - set(artifact.sources))
        if missing:
            raise KeyError(f"{label} has no entry for source keys {missing}")
    return (
        exact_dups.model_copy(update={"sources": {key: exact_dups.sources[key] for key in source_keys}}),
        verified.model_copy(update={"sources": {key: verified.sources[key] for key in source_keys}}),
    )


@dataclass(frozen=True)
class ReadInputs:
    """The store's inputs, loaded."""

    tokenize: dict[str, TokenizedAttrData]
    decontam: dict[str, DeconAttributes]
    cluster_assign: dict[str, AssignmentAttrData]
    quality_dirs: dict[str, str]
    content_type_dirs: dict[str, str]
    exact_dups: GlobalExactDedupData
    verified_fuzzy_dups: VerifiedFuzzyDupsAttrData


@dataclass(frozen=True)
class PerSourceInputs:
    """One source's four attribute stages, loaded, plus the key dedup files them under."""

    tokenize: dict[str, TokenizedAttrData]
    decontam: dict[str, DeconAttributes]
    cluster_assign: dict[str, AssignmentAttrData]
    quality_dirs: dict[str, str]
    content_type_dirs: dict[str, str]
    source_keys: dict[str, str]


def _read_per_source(inputs: StoreInputs) -> PerSourceInputs:
    """Read the per-source artifacts.

    The quality stage writes no artifact of its own -- ``score_corpus`` publishes
    narrow ``id``/``score`` shards straight into the step's directory -- so the
    score directory is the step path, and the cutpoints come from the pinned
    calibration rather than from a manifest beside the data.
    """
    tokenize = {n: read_artifact(s.tokenize.output_path, TokenizedAttrData) for n, s in inputs.per_source.items()}
    source_keys: dict[str, str] = {}
    for name, tok in tokenize.items():
        key = tok.source_keys.get(SPLIT)
        if key is None:
            raise ValueError(f"{name}: tokenize has no source_key for split={SPLIT!r}")
        source_keys[name] = key
    return PerSourceInputs(
        tokenize=tokenize,
        decontam={n: read_artifact(s.decontam.output_path, DeconAttributes) for n, s in inputs.per_source.items()},
        cluster_assign={
            n: read_artifact(s.cluster_assign.output_path, AssignmentAttrData) for n, s in inputs.per_source.items()
        },
        quality_dirs={n: s.quality.output_path for n, s in inputs.per_source.items()},
        content_type_dirs={n: s.content_type.output_path for n, s in inputs.per_source.items()},
        source_keys=source_keys,
    )


def _read_inputs(inputs: StoreInputs) -> ReadInputs:
    """Read every input artifact and restrict the dedup metadata to these sources."""
    per_source = _read_per_source(inputs)
    restricted_exact, verified = _restrict_to_sources(
        read_artifact(inputs.exact_dups.output_path, GlobalExactDedupData),
        read_artifact(inputs.verified_fuzzy_dups.output_path, VerifiedFuzzyDupsAttrData),
        set(per_source.source_keys.values()),
    )
    return ReadInputs(
        tokenize=per_source.tokenize,
        decontam=per_source.decontam,
        cluster_assign=per_source.cluster_assign,
        quality_dirs=per_source.quality_dirs,
        content_type_dirs=per_source.content_type_dirs,
        exact_dups=restricted_exact,
        verified_fuzzy_dups=verified,
    )


def build_store_step(
    inputs: StoreInputs,
    *,
    store: StoreConfig = StoreConfig(),
    max_workers: int,
    cluster_view: int = CLUSTER_VIEW,
) -> StepSpec:
    """Build the ``datakit/store`` terminal over already-registered inputs.

    The step's identity comes from its dependencies, so repointing any hero pin
    moves the store with it rather than serving a cache built from other bytes.
    """

    def run(output_path: str) -> ClusteredStoreData:
        loaded = _read_inputs(inputs)
        edges = calibration_edges_by_content_type(
            hero_data.quality_calibration(inputs.quality_model),
            expected_sha256=inputs.quality_model.calibration_sha256,
        )
        logger.info("quality cutpoints per content type: %s", {k: [round(e, 4) for e in v] for k, v in edges.items()})
        return build_clustered_store(
            tokenize=loaded.tokenize,
            decontam=loaded.decontam,
            cluster_assign=loaded.cluster_assign,
            quality=loaded.quality_dirs,
            bucket_edges=edges,
            content_type=loaded.content_type_dirs,
            exact_dedup=loaded.exact_dups,
            dedup=loaded.verified_fuzzy_dups,
            output_path=output_path,
            cluster_view=cluster_view,
            split=SPLIT,
            worker_resources=store.worker,
            max_workers=max_workers,
            task_count=store.task_count,
            partition_processes=store.partition_processes,
            max_parallel_bucket_writes=store.max_parallel_bucket_writes,
        )

    return StepSpec(
        name=STORE_STEP_NAME,
        deps=inputs.steps(),
        hash_attrs={
            "cluster_view": cluster_view,
            "split": SPLIT,
            "task_count": store.task_count,
            "v": STORE_VERSION,
        },
        fn=run,
    )


@dataclass
class PreflightReport:
    """What :func:`preflight` found. All three lists empty means the store can run."""

    sources: int
    missing: list[str] = field(default_factory=list)
    unsealed: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.missing and not self.unsealed and not self.problems


def _unbuilt_steps(steps: list[StepSpec]) -> tuple[list[str], list[str]]:
    """Split the steps the runner will not serve from cache into missing and unsealed.

    ``StepRunner`` decides from a ``SUCCESS`` status file, and a stage produced
    outside it has none, however complete its data is. That distinction matters
    here: a hero dependency refuses to execute, so an unsealed stage does not
    make the runner rebuild it, it makes the run die on the dependency. The two
    cases have different fixes, so they are reported apart.
    """
    missing: list[str] = []
    unsealed: list[str] = []
    for step in steps:
        if step_is_built(step):
            continue
        where = f"{step.name} -> {step.output_path}"
        holds_data = bool(StoragePath(f"{step.output_path.rstrip('/')}/*").glob())
        (unsealed if holds_data else missing).append(where)
    return missing, unsealed


def preflight(inputs: StoreInputs, *, shards_per_source: int) -> PreflightReport:
    """Check the inputs the way the store will, before the store costs anything.

    Five things go wrong here in practice and all five are cheap to find. A pin
    can address a directory that was never written. A stage can hold complete
    data that the runner will not serve, because whatever produced it never
    marked it succeeded. A dedup artifact can cover a different source set than
    the tokenize leaves. Two stages of one source can come from different
    normalize runs, which leaves their shards sharing neither basenames nor row
    order -- the store detects that per shard, deep into a fleet run, once it has
    already paid to read the shard.

    The fifth is the one the store cannot detect at all. Duplicate attributes are
    sparse, and a shard with no duplicates is legitimately absent, so a dedup
    directory written against an older normalize of the same source reads as "no
    duplicates anywhere" rather than as an error. The corpus would keep every
    duplicate in that source and nothing would say so, which is why the layouts
    are compared here rather than left to the run.
    """
    report = PreflightReport(sources=len(inputs.per_source))
    report.missing, report.unsealed = _unbuilt_steps(inputs.registered_steps())
    if report.missing:
        return report

    try:
        loaded = _read_per_source(inputs)
        exact_dups = read_artifact(inputs.exact_dups_pin.output_path, GlobalExactDedupData)
        verified = read_artifact(inputs.verified_fuzzy_dups.output_path, VerifiedFuzzyDupsAttrData)
        edges = calibration_edges_by_content_type(
            hero_data.quality_calibration(inputs.quality_model),
            expected_sha256=inputs.quality_model.calibration_sha256,
        )
    except (KeyError, ValueError) as failure:
        report.problems.append(str(failure))
        return report
    logger.info("quality cutpoints: %s", {k: [round(e, 4) for e in v] for k, v in edges.items()})
    report.problems.extend(_dedup_coverage_problems(inputs, loaded, exact_dups, verified))

    for name in sorted(inputs.per_source):
        report.problems.extend(
            _shard_alignment_problems(
                name=name,
                tokenize=loaded.tokenize[name],
                decontam=loaded.decontam[name],
                assign=loaded.cluster_assign[name],
                quality_dir=loaded.quality_dirs[name],
                content_type_dir=loaded.content_type_dirs[name],
                shards_per_source=shards_per_source,
            )
        )
        if name not in inputs.repacked_sources:
            report.problems.extend(
                _dedup_layout_problems(
                    name=name,
                    tokenize=loaded.tokenize[name],
                    source_key=loaded.source_keys[name],
                    exact_dups=exact_dups,
                    verified=verified,
                )
            )
    return report


def _dedup_coverage_problems(
    inputs: StoreInputs,
    loaded: PerSourceInputs,
    exact_dups: GlobalExactDedupData,
    verified: VerifiedFuzzyDupsAttrData,
) -> list[str]:
    """Check that both dedup artifacts file every source this run builds.

    A repacked source is the exception in both directions: the pinned artifact
    must still hold it under its *legacy* key, because that is what the repack
    reads, and must not hold it under the current one, because that is what the
    repack writes.
    """
    problems: list[str] = []
    for name in sorted(inputs.repacked_sources):
        current = loaded.source_keys[name]
        if hero_data.LEGACY_FOCUS_SOURCE_KEY not in exact_dups.sources:
            problems.append(
                f"{name}: exact_dups has no entry for the legacy key "
                f"{hero_data.LEGACY_FOCUS_SOURCE_KEY!r}, so there is nothing to repack"
            )
        if current in exact_dups.sources:
            problems.append(f"{name}: exact_dups already holds the current key {current!r}; the repack would refuse")

    expected = {loaded.source_keys[n] for n in inputs.per_source if n not in inputs.repacked_sources}
    for label, sources in (("exact_dups", exact_dups.sources), ("verified_fuzzy_dups", verified.sources)):
        absent = sorted(expected - set(sources))
        if absent:
            problems.append(f"{label} has no entry for {len(absent)} source keys, e.g. {absent[0]!r}")
    repacked_keys = {loaded.source_keys[n] for n in inputs.repacked_sources}
    absent_fuzzy = sorted(repacked_keys - set(verified.sources))
    if absent_fuzzy:
        problems.append(
            f"verified_fuzzy_dups has no entry for {absent_fuzzy!r}; fuzzy verification repacks the focus "
            "crawl itself (#8237), so its output should already carry the current key"
        )
    return problems


def _dedup_layout_problems(
    *,
    name: str,
    tokenize: TokenizedAttrData,
    source_key: str,
    exact_dups: GlobalExactDedupData,
    verified: VerifiedFuzzyDupsAttrData,
) -> list[str]:
    """Check one source's sparse duplicate attributes against its tokenize layout.

    Sparse means a shard with no duplicates is simply not there, so the only
    thing a listing can prove is the negative: a directory whose shards are named
    for a normalize the tokenize side does not share is addressing a different
    partitioning of the source, and every lookup against it will miss quietly.
    """
    tok_dir = tokenize.output_dirs.get(SPLIT)
    if tok_dir is None:
        return []
    tok_names = {os.path.basename(str(m)) for m in StoragePath(f"{tok_dir.rstrip('/')}/*.parquet").glob()}

    problems: list[str] = []
    for label, sources in (("exact_dups", exact_dups.sources), ("verified_fuzzy_dups", verified.sources)):
        entry = sources.get(source_key)
        if entry is None:
            continue
        attr_dir = entry.attr_dir.rstrip("/")
        attr_names = {os.path.basename(str(m)) for m in StoragePath(f"{attr_dir}/*.parquet").glob()}
        stray = attr_names - tok_names
        if stray:
            problems.append(
                f"{name}: {len(stray)} of {len(attr_names)} {label} shards are named for shards this "
                f"source does not have (e.g. {sorted(stray)[0]}) -- {label} was built against a different "
                "normalize, and its marks would be read as 'no duplicates' rather than as an error"
            )
    return problems


def _shard_alignment_problems(
    *,
    name: str,
    tokenize: TokenizedAttrData,
    decontam: DeconAttributes,
    assign: AssignmentAttrData,
    quality_dir: str,
    content_type_dir: str,
    shards_per_source: int,
) -> list[str]:
    """Report basename disagreements between one source's four attribute stages.

    Compares complete basename sets, then reads the ids of a sample of shards.
    A count that matches proves nothing on its own: the store routes the dense
    tables positionally, so equal-length shards in different row orders fail
    only once it compares them element by element.
    """

    def basenames(directory: str) -> set[str]:
        return {os.path.basename(str(m)) for m in StoragePath(f"{directory.rstrip('/')}/*.parquet").glob()}

    tok_dir = tokenize.output_dirs.get(SPLIT)
    if tok_dir is None:
        return [f"{name}: tokenize has no split={SPLIT!r}"]
    tok_names = basenames(tok_dir)
    if not tok_names:
        return [f"{name}: no tokenize shards under {tok_dir}"]

    problems: list[str] = []
    stages = {
        "decontam": decontam.main_output_dir,
        "cluster_assign": assign.output_dir,
        "quality": quality_dir,
        "content_type": content_type_dir,
    }
    for label, directory in stages.items():
        missing = tok_names - basenames(directory)
        if missing:
            problems.append(
                f"{name}: {label} is missing {len(missing)} of {len(tok_names)} tokenize shards "
                f"(e.g. {sorted(missing)[0]}) -- the two stages came from different normalize runs"
            )
    if problems:
        return problems

    def ids(directory: str, basename: str) -> list[str]:
        with StoragePath(f"{directory.rstrip('/')}/{basename}").open("rb") as handle:
            return pq.read_table(handle, columns=["id"]).column("id").to_pylist()

    # decontam is written in normalize order; cluster_assign and quality are both
    # driven off the embedding side. Equal ids in a different order is the failure
    # this is really looking for, and it is invisible to a row count.
    for basename in sorted(tok_names)[:shards_per_source]:
        reference = ids(stages["decontam"], basename)
        for label in ("cluster_assign", "quality", "content_type"):
            other = ids(stages[label], basename)
            if other == reference:
                continue
            if len(other) != len(reference):
                detail = f"{len(reference)} rows against {len(other)}"
            elif sorted(other) == sorted(reference):
                detail = "the same ids in a different row order"
            else:
                detail = "different ids"
            problems.append(
                f"{name}/{basename}: decontam and {label} disagree -- {detail}. "
                "The store routes these positionally, so it cannot join them."
            )
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = StoreConfig()
    parser.add_argument("--sources", help="comma-separated source names; omit for every registered source")
    parser.add_argument("--task-count", type=int, default=default.task_count, help="map tasks to deal shards into")
    parser.add_argument("--max-workers", type=int, default=16, help="tasks running at once; sized to the CPU pool")
    parser.add_argument("--worker-cpu", type=float, default=default.worker.cpu)
    parser.add_argument("--worker-ram", default=default.worker.ram)
    parser.add_argument("--worker-disk", default=default.worker.disk, help="local spill room per task")
    parser.add_argument("--partition-processes", type=int, default=default.partition_processes)
    parser.add_argument("--max-parallel-bucket-writes", type=int, default=default.max_parallel_bucket_writes)
    parser.add_argument("--cluster-view", type=int, default=CLUSTER_VIEW, help="K the store partitions on")
    parser.add_argument("--preflight", action="store_true", help="check the inputs and exit without submitting")
    parser.add_argument(
        "--preflight-shards",
        type=int,
        default=2,
        help="shards per source whose ids --preflight compares across stages",
    )
    parser.add_argument("--pending", action="store_true", help="list unregistered stages and exit")
    parser.add_argument(
        "--repack-only",
        action="store_true",
        help="run just the focus-crawl exact-duplicate repack, which needs no stage the store is waiting on",
    )
    parser.add_argument("--repack-workers", type=int, default=DEFAULT_REPACK_MAX_WORKERS)
    parser.add_argument("--repack-worker-cpu", type=float, default=DEFAULT_REPACK_WORKER.cpu)
    parser.add_argument("--repack-worker-ram", default=DEFAULT_REPACK_WORKER.ram)
    parser.add_argument("--dry-run", action="store_true", help="report the cache state without submitting")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    unregistered = pending(sources)
    if args.pending:
        for item in unregistered:
            logger.info("pending: %s -- %s", item.stage, item.remedy)
        if not unregistered:
            logger.info("every stage the store reads is registered")
        return

    if args.repack_only:
        # Reads nothing the store is still waiting on, so it runs before the rest
        # of the graph is registered. Same call as ``store_inputs`` makes, so the
        # step lands at the identity the store will later depend on.
        target = build_focus_exact_repack_step(
            hero_data.exact_dups(),
            max_workers=args.repack_workers,
            worker_resources=ResourceConfig(cpu=args.repack_worker_cpu, ram=args.repack_worker_ram, disk="16g"),
        )
        logger.info("exact-duplicate repack target: %s", target.output_path)
        logger.info("expect %d marks kept; anything else means the model is wrong", FOCUS_EXPECTED_MARKS_KEPT)
        StepRunner().run([target], dry_run=args.dry_run, max_concurrent=1)
        return

    if unregistered:
        raise SystemExit(
            "the store cannot be built yet:\n" + "\n".join(f"  {item.stage}: {item.remedy}" for item in unregistered)
        )

    inputs = store_inputs(
        sources,
    )
    store = StoreConfig(
        task_count=args.task_count,
        partition_processes=args.partition_processes,
        max_parallel_bucket_writes=args.max_parallel_bucket_writes,
        worker=ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk=args.worker_disk, preemptible=False),
    )

    if args.preflight:
        report = preflight(inputs, shards_per_source=args.preflight_shards)
        for path in report.missing:
            logger.error("no data: %s", path)
        for path in report.unsealed:
            logger.error("holds data but is not marked succeeded: %s", path)
        if report.unsealed:
            logger.error(
                "%d stages need sealing before StepRunner will serve them; "
                "see experiments/datakit/scripts/seal_derived_stage.py",
                len(report.unsealed),
            )
        for problem in report.problems:
            logger.error("%s", problem)
        if not report.ok():
            raise SystemExit(f"preflight failed over {report.sources} sources")
        logger.info("preflight passed over %d sources", report.sources)
        return

    target = build_store_step(inputs, store=store, max_workers=args.max_workers, cluster_view=args.cluster_view)
    logger.info("store target: %s", target.output_path)
    if inputs.repacked_sources:
        logger.info(
            "repacking exact-duplicate marks for %s first: %s",
            ", ".join(inputs.repacked_sources),
            inputs.exact_dups.output_path,
        )
    logger.info(
        "%d sources, %s tasks, %d concurrent workers of %s cpu / %s ram / %s disk",
        len(inputs.per_source),
        store.task_count,
        args.max_workers,
        store.worker.cpu,
        store.worker.ram,
        store.worker.disk,
    )
    StepRunner().run([target], dry_run=args.dry_run, max_concurrent=1)


if __name__ == "__main__":
    main()

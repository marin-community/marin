# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The pinned RL task releases the Exp3 long-loop runs trained on.

The long-loop RL runs read two staged parquet releases directly: the round-3 RL
data handoff froze them, and the launch documents pin their files by SHA-256.
Neither release was produced by a pipeline in this repository, so the catalog
adopts them: the adoption stage reads the pinned release back in region, verifies
every file's digest and row provenance against the recorded manifest, and writes
an admission receipt. It converts nothing; the releases' own seals remain the
qualification evidence.
"""

import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.lazy import ArtifactStep
from marin.rl.curriculum import PoolRow
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.models import SourceStatus
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.campaign import CampaignArtifact, PipelineResult
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.source import (
    RlDataSource,
    SourceInfo,
    SourceOrigin,
    SourceReference,
    TaskFormat,
)

AGENTIC_RELEASE_NAME = "long-loop-rl-agentic-train"
NONAGENTIC_RELEASE_NAME = "long-loop-rl-nonagentic-train"

AGENTIC_RELEASE_URL = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/rl-data/swe-verified-20261112-v4/release"
NONAGENTIC_RELEASE_URL = "s3://marin-us-east-02a/marin/users/benfeuer/long-loop/rl-data/smartie-mixed-20261110-v1"

AGENTIC_TRAIN_SOURCE = "long-loop-smartie-swesmith"
AGENTIC_TRAIN_SHA256 = "453552f9d4e1fa27710f9434a303f7ab5ff390c22deb54572f9be885a2a80f5d"
AGENTIC_MANIFEST_SHA256 = "ec645c79b1d636df20dc0575db5bd7df608eaf0a06203cd63ead8e0c310860b3"
NONAGENTIC_TRAIN_SHA256 = "c8948b3c13957ee054b4b8b5553127f75c44a50577b5b2b798261e9f8a5abc45"
NONAGENTIC_MANIFEST_SHA256 = "94f21b06fedc6dce7184e783dab1349f1c14fed05fab3d47fcace61d2386701a"

AGENTIC_TRAIN_FILE = "agentic-train.parquet"
NONAGENTIC_TRAIN_FILE = "nonagentic-train.parquet"
"""The two releases' train parquet members: the pools the runs trained on."""

AGENTIC_ROUTE = "agentic"
NONAGENTIC_ROUTE = "nonagentic"
"""The routes the long-loop runs' rollout buffers admitted groups through."""

AGENTIC_TRAIN_ROWS = 5608
EURUS_ROWS = 10_317
NEMOTRON_BLENDS_ROWS = 12_426
OPEN_SCIENCE_ROWS = 20_790
REASONING_GYM_ROWS = 20_800
SNOWBALL_SMARTIE_ROWS = 11_634
TASKTROVE_ROWS = 24_033
NONAGENTIC_TRAIN_ROWS = (
    EURUS_ROWS + NEMOTRON_BLENDS_ROWS + OPEN_SCIENCE_ROWS + REASONING_GYM_ROWS + SNOWBALL_SMARTIE_ROWS + TASKTROVE_ROWS
)

REASONING_GYM_SOURCE = (
    "https://files.pythonhosted.org/packages/42/21/"
    "bb9f6d2f76424f8b516fe83f8026e52e466b0202341bb4ee7a9861e06277/reasoning_gym-0.1.25-py3-none-any.whl"
)

RELEASE_SOURCE_ROUTES = MappingProxyType(
    {
        AGENTIC_TRAIN_SOURCE: AGENTIC_ROUTE,
        "PRIME-RL/Eurus-2-RL-Data": NONAGENTIC_ROUTE,
        "nvidia/Nemotron-RL-Ultra-Training-Blends": NONAGENTIC_ROUTE,
        "nvidia/OpenScience": NONAGENTIC_ROUTE,
        REASONING_GYM_SOURCE: NONAGENTIC_ROUTE,
        "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": NONAGENTIC_ROUTE,
        "open-thoughts/TaskTrove": NONAGENTIC_ROUTE,
    }
)
"""Each training row's ``data_source`` and the route the runs admitted it through."""

REASONING_GYM_LABEL = "Reasoning Gym:reasoning_gym"
"""The ``teacher_route`` the stamped pools label the reasoning-gym rows with: the
upstream ``data_source`` is the 0.1.25 wheel URL, which is not a valid pool step
name, so the stamped population renames it to the source identity."""

TEACHER_ROUTE_LABELS: Mapping[str, str] = MappingProxyType(
    {**{source: source for source in RELEASE_SOURCE_ROUTES}, REASONING_GYM_SOURCE: REASONING_GYM_LABEL}
)
"""Each training row's ``data_source`` and the ``teacher_route`` the stamped pools
label it with — identical except the reasoning-gym wheel URL."""


@dataclass(frozen=True)
class AdoptedFile:
    """One pinned member of an adopted release and the row provenance it must carry."""

    name: str
    sha256: str
    rows: int | None = None
    source_counts: Mapping[str, int] | None = None


@dataclass(frozen=True)
class AdoptedRelease:
    """Where an adopted long-loop RL release lives and what its manifest pins."""

    name: str
    version: str
    url: str
    manifest_sha256: str
    members: tuple[AdoptedFile, ...]

    @property
    def dataset(self) -> SourceReference:
        return SourceReference(self.url, self.manifest_sha256, self.url)

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(member.name for member in self.members)


def _release(source: RlDataSource[AdoptedRelease]) -> AdoptedRelease:
    if source.config is None:
        raise ValueError(f"adopted release {source.info.id} has no configuration")
    return source.config


def stream_sha256(fs, path: str) -> str:
    """Stream one pinned object's digest without buffering it; the execution region follows the caller's filesystem."""
    with fs.open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True)
class VerifiedFile:
    """One verified member's digest, row count, and per-source provenance."""

    sha256: str
    rows: int | None = None
    source_counts: Mapping[str, int] | None = None


def verify_release(config: AdoptedRelease) -> dict[str, VerifiedFile]:
    """Verify the pinned release against its pins and return the admitted composition.

    Every member's SHA-256 must match its pin. A member that records rows must
    carry that row count, and one that records provenance must carry exactly the
    recorded per-source counts, read from each task spec's ``source.dataset``
    (the trainer's ``data_source``). Batches stream; prompt payloads are counted,
    never stored.
    """
    fs, base = filesystem_for(config.url)
    composition: dict[str, VerifiedFile] = {}
    for member in config.members:
        path = prefix_join(base, member.name)
        observed_sha = stream_sha256(fs, path)
        if observed_sha != member.sha256:
            raise ValueError(f"{path}: sha256 {observed_sha} does not match the pinned {member.sha256}")
        verified = VerifiedFile(sha256=observed_sha)
        if member.rows is not None:
            with fs.open(path, "rb") as handle:
                rows = pq.ParquetFile(handle).metadata.num_rows
            if rows != member.rows:
                raise ValueError(f"{path}: {rows} rows, expected {member.rows}")
            verified = replace(verified, rows=rows)
        if member.source_counts is not None:
            counts = provenance_counts(fs, path)
            if counts != dict(member.source_counts):
                raise ValueError(f"{path}: provenance counts {dict(counts)} differ from the recorded manifest")
            verified = replace(verified, source_counts=dict(counts))
        composition[member.name] = verified
    return composition


def stream_task_specs(fs, path: str) -> Iterator[dict]:
    """Stream one pinned parquet's task specs, parsed, without storing payloads."""
    with fs.open(path, "rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(batch_size=64, columns=["task_spec"]):
            for value in batch.column("task_spec").to_pylist():
                yield json.loads(value)


def provenance_counts(fs, path: str) -> Counter:
    """Count each task spec's ``source.dataset`` across one pinned parquet, streaming."""
    counts: Counter = Counter()
    for task in stream_task_specs(fs, path):
        dataset = task.get("source", {}).get("dataset")
        if dataset is None:
            raise ValueError(f"{path}: a task spec carries no source.dataset")
        counts[dataset] += 1
    return counts


@dataclass(frozen=True)
class Adoption:
    """The release verification this stage runs and where its receipt lands.

    ``build_config`` returns this, so the artifact's fingerprint covers the pinned
    release: a changed URL, digest, row count, or provenance count is a different
    artifact, and an existing admission receipt cannot be reused for it.
    """

    output_path: str
    config: AdoptedRelease


def adopt_release(source: RlDataSource[AdoptedRelease], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    """Adopt the pinned release as the RL prompt pool: verify digests and provenance, convert nothing."""
    config = _release(source)
    if options.mode is SourceProcessingMode.QUICK:
        raise ValueError(f"an adopted release is admitted, not re-processed: {source.name}")

    def run(adoption: Adoption) -> CampaignArtifact:
        composition = verify_release(adoption.config)
        destination = StoragePath(adoption.output_path)
        destination.mkdirs(exist_ok=True)
        manifest = {
            "status": "completed",
            "release": adoption.config.url,
            "manifest_sha256": adoption.config.manifest_sha256,
            "composition": {name: asdict(verified) for name, verified in composition.items()},
        }
        receipt = destination / "manifest.json"
        receipt.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return CampaignArtifact(
            status="completed",
            result=PipelineResult(
                status=SourceStatus.COMPLETED,
                outputs={"release": adoption.config.url},
                evidence={"manifest": str(receipt)},
                stages=("verify",),
            ),
        )

    return ArtifactStep(
        name=config.name,
        version=config.version,
        artifact_type=CampaignArtifact,
        run=run,
        build_config=lambda context: Adoption(context.output_path, config),
        deps=(),
    )


def agentic_release() -> RlDataSource[AdoptedRelease]:
    """The round-3 agentic training pool: SWE-smith Harbor tasks with the Pi execution overlay."""
    return RlDataSource(
        info=SourceInfo(
            id="Long-Loop RL:agentic-train",
            origin=SourceOrigin.MARIN,
            upstream_dataset="long-loop-rl-agentic-train",
            upstream_name="Long-loop agentic RL train release",
            build_path="Adopted RL release",
            family="terminal-agent",
            tags=("agentic", "multi-turn"),
            count=AGENTIC_TRAIN_ROWS,
            notes=(
                "Adopted round-3 long-loop release: SWE-smith Harbor tasks under the Pi 0.87.0 "
                "non-root execution overlay, sealed by its own manifest. Every training row's "
                "data_source is long-loop-smartie-swesmith. The release also contains a 120-row "
                "validation.parquet that predates curriculum-owned holdouts; the launch no longer "
                "consumes it and this record does not adopt it. Qualification rests on the release "
                "seal and the long-loop launch receipts, not on curation in this repository."
            ),
            task_format=TaskFormat.HARBOR,
        ),
        config=AdoptedRelease(
            name=AGENTIC_RELEASE_NAME,
            version="1",
            url=AGENTIC_RELEASE_URL,
            manifest_sha256=AGENTIC_MANIFEST_SHA256,
            members=(
                AdoptedFile(
                    AGENTIC_TRAIN_FILE,
                    AGENTIC_TRAIN_SHA256,
                    AGENTIC_TRAIN_ROWS,
                    {AGENTIC_TRAIN_SOURCE: AGENTIC_TRAIN_ROWS},
                ),
            ),
        ),
        pipeline=adopt_release,
    )


def nonagentic_release() -> RlDataSource[AdoptedRelease]:
    """The round-3 nonagentic training pool: the six-source smartie blend."""
    return RlDataSource(
        info=SourceInfo(
            id="Long-Loop RL:nonagentic-train",
            origin=SourceOrigin.MARIN,
            upstream_dataset="long-loop-rl-nonagentic-train",
            upstream_name="Long-loop nonagentic RL train release",
            build_path="Adopted RL release",
            family="blend",
            tags=("rlvr", "single-turn"),
            count=NONAGENTIC_TRAIN_ROWS,
            notes=(
                "Adopted round-3 long-loop release: 100,000 prompt rows across six sources — "
                f"open-thoughts/TaskTrove {TASKTROVE_ROWS:,}, the reasoning-gym 0.1.25 wheel "
                f"{REASONING_GYM_ROWS:,}, nvidia/OpenScience {OPEN_SCIENCE_ROWS:,}, "
                f"nvidia/Nemotron-RL-Ultra-Training-Blends {NEMOTRON_BLENDS_ROWS:,}, "
                f"open-athena/Snowball-Smartie-RL-2026.10.15.32-Data {SNOWBALL_SMARTIE_ROWS:,}, "
                f"PRIME-RL/Eurus-2-RL-Data {EURUS_ROWS:,}. This release's own agentic parquet "
                "(10,210 rows) was superseded by the swe-verified release and is not part of the "
                "training pool. Qualification rests on the release seal and the long-loop launch receipts."
            ),
        ),
        config=AdoptedRelease(
            name=NONAGENTIC_RELEASE_NAME,
            version="1",
            url=NONAGENTIC_RELEASE_URL,
            manifest_sha256=NONAGENTIC_MANIFEST_SHA256,
            members=(
                AdoptedFile(
                    NONAGENTIC_TRAIN_FILE,
                    NONAGENTIC_TRAIN_SHA256,
                    NONAGENTIC_TRAIN_ROWS,
                    {
                        "PRIME-RL/Eurus-2-RL-Data": EURUS_ROWS,
                        "nvidia/Nemotron-RL-Ultra-Training-Blends": NEMOTRON_BLENDS_ROWS,
                        "nvidia/OpenScience": OPEN_SCIENCE_ROWS,
                        REASONING_GYM_SOURCE: REASONING_GYM_ROWS,
                        "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": SNOWBALL_SMARTIE_ROWS,
                        "open-thoughts/TaskTrove": TASKTROVE_ROWS,
                    },
                ),
            ),
        ),
        pipeline=adopt_release,
    )


def sources() -> list[RlDataSource[AdoptedRelease]]:
    return [agentic_release(), nonagentic_release()]


AGENTIC_RELEASE = agentic_release()
NONAGENTIC_RELEASE = nonagentic_release()
"""The two adopted releases as catalog sources; curricula bind entries against them."""


STAMPED_POOL_SCHEMA = pa.schema(
    [
        pa.field("task_spec", pa.string(), nullable=False),
        pa.field("teacher_route", pa.string(), nullable=False),
    ]
)
"""The stamped long-loop pool layout: task content plus the trainer's sampling key."""

STAMPED_POOLS_VERSION = "2026.11.18.71"
STAMPED_AGENTIC_POOL_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp4/long-loop-exp4/rl-pool-agentic/2026.11.18.71"
)
STAMPED_AGENTIC_POOL_SHA256 = "ad435c5bdd3ac7ef46be5359a8f5fa1b095e80fbc9bfbcb968f22b5fe7befae3"
STAMPED_NONAGENTIC_POOL_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp4/long-loop-exp4/rl-pool-nonagentic/2026.11.18.71"
)
STAMPED_NONAGENTIC_POOL_SHA256 = "4cb4fd8206a113509e6fcd9c0c4b1198c6d925eb964b1622c98d12ef36151402"
"""The regional Exp4 stamped pools the runs train on: route-labeled populations
over the adopted releases plus the admitted Nemotron instruction-following rows,
pinned by their train.parquet digests."""

STAMPED_POOLS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        AGENTIC_ROUTE: (STAMPED_AGENTIC_POOL_URI, STAMPED_AGENTIC_POOL_SHA256),
        NONAGENTIC_ROUTE: (STAMPED_NONAGENTIC_POOL_URI, STAMPED_NONAGENTIC_POOL_SHA256),
    }
)


def stream_stamped_pool(fs, path: str) -> Iterator[tuple[str, str]]:
    """Stream one stamped pool's ``(teacher_route, raw task_spec)`` pairs."""
    with fs.open(path, "rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(batch_size=64, columns=["task_spec", "teacher_route"]):
            routes = batch.column("teacher_route").to_pylist()
            tasks = batch.column("task_spec").to_pylist()
            yield from zip(routes, tasks)


def stamped_pool_rows(pools: Mapping[str, tuple[str, str]] = STAMPED_POOLS) -> dict[str, tuple[PoolRow, ...]]:
    """Every stamped pool's row identities, keyed by ``teacher_route``.

    Row identity is the task spec's ``id`` plus the canonical SHA-256 of its
    task document — the identity a curriculum's holdout records and extraction
    verifies."""
    rows_by_label: dict[str, list[PoolRow]] = {}
    for uri, sha256 in pools.values():
        filesystem, base = filesystem_for(uri)
        path = prefix_join(base, "train.parquet")
        observed = stream_sha256(filesystem, path)
        if observed != sha256:
            raise ValueError(f"{path}: sha256 {observed} does not match the pinned {sha256}")
        for route, raw in stream_stamped_pool(filesystem, path):
            task = json.loads(raw)
            rows_by_label.setdefault(route, []).append(PoolRow(id=task["id"], sha256=canonical_sha256(task)))
    return {label: tuple(rows) for label, rows in rows_by_label.items()}


def stamped_pool_step(route: str, version: str = STAMPED_POOLS_VERSION) -> ArtifactStep:
    """Adopt one pinned stamped pool as a graph artifact."""
    uri, sha256 = STAMPED_POOLS[route]
    return ArtifactStep.adopt(f"long-loop-exp4/stamped-{route}", version, source=uri, config={"sha256": sha256})


def _verified_pool_path(pool: dict) -> str:
    """Resolve and digest-verify one pinned stamped pool's train.parquet."""
    filesystem, base = filesystem_for(pool["uri"])
    path = prefix_join(base, "train.parquet")
    observed = stream_sha256(filesystem, path)
    if observed != pool["sha256"]:
        raise ValueError(f"{path}: sha256 {observed} does not match the pinned {pool['sha256']}")
    return path


def _write_pool_parquet(destination: StoragePath, name: str, rows: list[dict]) -> str:
    """Write one stamped-schema parquet through a local temp file and verify the readback."""
    destination.mkdirs(exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary) / name
        pq.write_table(pa.Table.from_pylist(rows, schema=STAMPED_POOL_SCHEMA), local)
        with local.open("rb") as reader:
            digest = hashlib.file_digest(reader, "sha256").hexdigest()
        target = destination / name
        with local.open("rb") as reader, target.open("wb") as writer:
            while chunk := reader.read(8 * 1024 * 1024):
                writer.write(chunk)
        with target.open("rb") as reader:
            if hashlib.file_digest(reader, "sha256").hexdigest() != digest:
                raise ValueError(f"{target}: readback differs from the written bytes")
        return digest


def extract_source_pool(*, output: str, pool: dict, label: str, excluded: Sequence[str], expected_rows: int) -> None:
    """Write one ``teacher_route`` population of a stamped pool, minus holdout rows.

    The pool's bytes must match their pinned digest. The kept rows carry the
    stamped schema unchanged, the holdout ids are excluded, and the remaining
    count must match ``expected_rows``. The manifest records the exclusion."""
    filesystem, _ = filesystem_for(pool["uri"])
    path = _verified_pool_path(pool)
    excluded_ids = set(excluded)
    kept: list[dict] = []
    for route, raw in stream_stamped_pool(filesystem, path):
        if route != label or json.loads(raw)["id"] in excluded_ids:
            continue
        kept.append({"task_spec": raw, "teacher_route": route})
    if len(kept) != expected_rows:
        raise ValueError(f"{path}: {label!r} holds {len(kept)} rows after exclusion, expected {expected_rows}")
    destination = StoragePath(output)
    _write_pool_parquet(destination, "train.parquet", kept)
    manifest = {"label": label, "rows": len(kept), "excluded_holdout_rows": len(excluded_ids)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def extract_validation_pool(*, output: str, pools: Sequence[dict], rows: Sequence[dict], seed: int) -> None:
    """Materialize a curriculum's holdout as validation.parquet from the stamped pools.

    Every recorded row must appear exactly once across the pools and its task
    document must hash to the recorded SHA-256 — the content check that makes
    the recorded identity load-bearing. The manifest records the full split
    payload and its per-label composition."""
    wanted = {(row["label"], row["id"]): row["sha256"] for row in rows}
    found: dict[tuple[str, str], str] = {}
    composition: Counter = Counter()
    for pool in pools:
        filesystem, _ = filesystem_for(pool["uri"])
        path = _verified_pool_path(pool)
        for route, raw in stream_stamped_pool(filesystem, path):
            task = json.loads(raw)
            key = (route, task["id"])
            if key not in wanted:
                continue
            if canonical_sha256(task) != wanted[key]:
                raise ValueError(f"{path}: holdout row {route!r}/{task['id']!r} differs from its recorded sha256")
            if key in found:
                raise ValueError(f"{path}: holdout row {route!r}/{task['id']!r} appears twice")
            found[key] = raw
            composition[route] += 1
    missing = sorted(wanted.keys() - found.keys())
    if missing:
        raise ValueError(f"holdout rows absent from the pools: {missing[:5]}")
    destination = StoragePath(output)
    ordered = sorted(found.items(), key=lambda item: item[0])
    _write_pool_parquet(
        destination, "validation.parquet", [{"task_spec": raw, "teacher_route": route} for (route, _), raw in ordered]
    )
    manifest = {
        "validation": {"size": len(rows), "seed": seed, "rows": list(rows)},
        "composition": dict(sorted(composition.items())),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

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
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

import pyarrow.parquet as pq
from marin.execution.lazy import ArtifactStep
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.pipeline.models import SourceStatus
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation.campaign import CampaignArtifact, PipelineResult
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.source import (
    RlDataSource,
    SourceInfo,
    SourceReference,
    TaskFormat,
)

AGENTIC_RELEASE_NAME = "long-loop-rl-agentic-train"
NONAGENTIC_RELEASE_NAME = "long-loop-rl-nonagentic-train"

AGENTIC_RELEASE_URL = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/rl-data/swe-verified-20261112-v4/release"
NONAGENTIC_RELEASE_URL = "s3://marin-us-east-02a/marin/users/benfeuer/long-loop/rl-data/smartie-mixed-20261110-v1"

AGENTIC_TRAIN_SOURCE = "long-loop-smartie-swesmith"
AGENTIC_TRAIN_SHA256 = "453552f9d4e1fa27710f9434a303f7ab5ff390c22deb54572f9be885a2a80f5d"
AGENTIC_VALIDATION_SHA256 = "8601f26d9b16467efdfb022166367a1c413adfa4118f5110a76b12258e254838"
AGENTIC_MANIFEST_SHA256 = "ec645c79b1d636df20dc0575db5bd7df608eaf0a06203cd63ead8e0c310860b3"
NONAGENTIC_TRAIN_SHA256 = "c8948b3c13957ee054b4b8b5553127f75c44a50577b5b2b798261e9f8a5abc45"
NONAGENTIC_MANIFEST_SHA256 = "94f21b06fedc6dce7184e783dab1349f1c14fed05fab3d47fcace61d2386701a"

AGENTIC_TRAIN_ROWS = 5608
AGENTIC_VALIDATION_ROWS = 120
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


def provenance_counts(fs, path: str) -> Counter:
    """Count each task spec's ``source.dataset`` across one pinned parquet, streaming."""
    counts: Counter = Counter()
    with fs.open(path, "rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(batch_size=64, columns=["task_spec"]):
            for value in batch.column("task_spec").to_pylist():
                dataset = json.loads(value).get("source", {}).get("dataset")
                if dataset is None:
                    raise ValueError(f"{path}: a task spec carries no source.dataset")
                counts[dataset] += 1
    return counts


def adopt_release(source: RlDataSource[AdoptedRelease], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]:
    """Adopt the pinned release as the RL prompt pool: verify digests and provenance, convert nothing."""
    config = _release(source)
    if options.mode is SourceProcessingMode.QUICK:
        raise ValueError(f"an adopted release is admitted, not re-processed: {source.name}")

    def run(output_path: str) -> CampaignArtifact:
        composition = verify_release(config)
        destination = StoragePath(output_path)
        destination.mkdirs(exist_ok=True)
        manifest = {
            "status": "completed",
            "release": config.url,
            "manifest_sha256": config.manifest_sha256,
            "composition": {name: asdict(verified) for name, verified in composition.items()},
        }
        receipt = destination / "manifest.json"
        receipt.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return CampaignArtifact(
            status="completed",
            result=PipelineResult(
                status=SourceStatus.COMPLETED,
                outputs={"release": config.url},
                evidence={"manifest": str(receipt)},
                stages=("verify",),
            ),
        )

    return ArtifactStep(
        name=config.name,
        version=config.version,
        artifact_type=CampaignArtifact,
        run=run,
        build_config=lambda context: context.output_path,
        deps=(),
    )


def agentic_release() -> RlDataSource[AdoptedRelease]:
    """The round-3 agentic training pool: SWE-smith Harbor tasks with the Pi execution overlay."""
    return RlDataSource(
        info=SourceInfo(
            id="Long-Loop RL:agentic-train",
            title="Long-loop agentic RL train release · swe-verified-20261112-v4",
            origin="Long-Loop RL",
            family="terminal-agent",
            tags=("agentic", "multi-turn"),
            count=AGENTIC_TRAIN_ROWS,
            notes=(
                "Adopted round-3 long-loop release: SWE-smith Harbor tasks under the Pi 0.87.0 "
                "non-root execution overlay, sealed by its own manifest. Every training row's "
                "data_source is long-loop-smartie-swesmith; the release also carries the 120-row "
                "held-out validation pool. Qualification rests on the release seal and the "
                "long-loop launch receipts, not on curation in this repository."
            ),
            task_format=TaskFormat.HARBOR,
            dataset=SourceReference(
                AGENTIC_RELEASE_URL,
                AGENTIC_MANIFEST_SHA256,
                AGENTIC_RELEASE_URL,
            ),
        ),
        config=AdoptedRelease(
            name=AGENTIC_RELEASE_NAME,
            version="1",
            url=AGENTIC_RELEASE_URL,
            manifest_sha256=AGENTIC_MANIFEST_SHA256,
            members=(
                AdoptedFile(
                    "agentic-train.parquet",
                    AGENTIC_TRAIN_SHA256,
                    AGENTIC_TRAIN_ROWS,
                    {AGENTIC_TRAIN_SOURCE: AGENTIC_TRAIN_ROWS},
                ),
                AdoptedFile("validation.parquet", AGENTIC_VALIDATION_SHA256, AGENTIC_VALIDATION_ROWS),
            ),
        ),
        pipeline=adopt_release,
    )


def nonagentic_release() -> RlDataSource[AdoptedRelease]:
    """The round-3 nonagentic training pool: the six-source smartie blend."""
    return RlDataSource(
        info=SourceInfo(
            id="Long-Loop RL:nonagentic-train",
            title="Long-loop nonagentic RL train release · smartie-mixed-20261110-v1",
            origin="Long-Loop RL",
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
            dataset=SourceReference(
                NONAGENTIC_RELEASE_URL,
                NONAGENTIC_MANIFEST_SHA256,
                NONAGENTIC_RELEASE_URL,
            ),
        ),
        config=AdoptedRelease(
            name=NONAGENTIC_RELEASE_NAME,
            version="1",
            url=NONAGENTIC_RELEASE_URL,
            manifest_sha256=NONAGENTIC_MANIFEST_SHA256,
            members=(
                AdoptedFile(
                    "nonagentic-train.parquet",
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

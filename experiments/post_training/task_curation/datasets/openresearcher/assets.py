# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage licensed OpenResearcher inputs directly to regional S3 from an Iris CPU job."""

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from functools import partial

from finestore.rl.licenses import LicenseReceipt, require_license
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact, result_type_name
from marin.execution.lazy import ArtifactStep, StepContext, run
from pydantic import TypeAdapter
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.pipeline.inputs import SourceFormat
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.local import artifact_prefix, local_client
from experiments.post_training.task_curation.pipeline import HfSource, download_step

REGIONAL_PREFIX = "s3://marin-us-east-02a/marin/"
DATASET = "OpenResearcher/OpenResearcher-Dataset"
DATASET_REVISION = "1edca18c5d7c9c105762da4f3c5c2b83855b3c6e"
CORPUS = "OpenResearcher/OpenResearcher-Corpus"
CORPUS_REVISION = "8367703babe6e080899a48782f365c284c32b668"
EXPECTED_PINS = {DATASET: DATASET_REVISION, CORPUS: CORPUS_REVISION}


@dataclass(frozen=True)
class FileDigest:
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class SourceAsset:
    source_id: str
    dataset_id: str
    revision: str
    files: tuple[FileDigest, ...]


class RegionalInputs(Artifact):
    source_roots: dict[str, str]
    source_manifest_sha256: str
    license_receipts_sha256: str
    verified_files: int
    verified_bytes: int
    iris_task_id: str


@dataclass(frozen=True)
class StageConfig:
    assets: tuple[SourceAsset, ...]
    receipts_json: str
    source_roots: dict[str, str]
    source_manifest_sha256: str
    license_receipts_sha256: str
    output_path: str


def checked_inputs(config: StageConfig) -> RegionalInputs:
    if not os.environ.get("IRIS_TASK_ID"):
        raise RuntimeError("Dataset preparation and validation must run inside the regional Iris CPU job")
    files, total_bytes = 0, 0
    receipts = json.loads(config.receipts_json)
    for asset in config.assets:
        require_license(LicenseReceipt.model_validate(receipts[asset.source_id]), asset.source_id, "train", ban=None)
        for expected in asset.files:
            path = StoragePath(config.source_roots[asset.dataset_id]) / expected.path
            if file_sha256(path) != expected.sha256:
                raise ValueError(f"Pinned source LFS SHA256 mismatch: {path}")
            files += 1
            total_bytes += expected.bytes
    output = StoragePath(config.output_path)
    (output / "source-manifest.json").write_text(json.dumps([asdict(asset) for asset in config.assets], indent=2) + "\n")
    (output / "license-receipts.json").write_text(config.receipts_json)
    return RegionalInputs(
        path=config.output_path,
        source_roots=config.source_roots,
        source_manifest_sha256=config.source_manifest_sha256,
        license_receipts_sha256=config.license_receipts_sha256,
        verified_files=files,
        verified_bytes=total_bytes,
        iris_task_id=os.environ["IRIS_TASK_ID"],
    )


def stage_config(
    assets: tuple[SourceAsset, ...],
    receipts_json: str,
    downloads: dict[str, ArtifactStep],
    manifest_sha256: str,
    receipts_sha256: str,
    context: StepContext,
) -> StageConfig:
    return StageConfig(
        assets,
        receipts_json,
        {name: context.artifact_path(step) for name, step in downloads.items()},
        manifest_sha256,
        receipts_sha256,
        context.output_path,
    )


def pinned_document(uri: str, sha256: str) -> bytes:
    data = StoragePath(uri).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"Metadata SHA256 mismatch: {uri}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--license-receipts", required=True)
    parser.add_argument("--license-receipts-sha256", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    StoragePath(args.artifact_root).relative_to(StoragePath(REGIONAL_PREFIX))
    if args.run and not os.environ.get("IRIS_TASK_ID"):
        raise RuntimeError("Launch this preparation through Iris in the same region as the S3 artifact root")
    configure_coreweave_s3()
    assets = TypeAdapter(tuple[SourceAsset, ...]).validate_json(
        pinned_document(args.source_manifest, args.source_manifest_sha256)
    )
    if {asset.dataset_id: asset.revision for asset in assets} != EXPECTED_PINS or len(assets) != 2:
        raise ValueError("Source manifest differs from the two approved immutable OpenResearcher releases")
    receipts_json = pinned_document(args.license_receipts, args.license_receipts_sha256).decode()
    receipts = {key: LicenseReceipt.model_validate(value) for key, value in json.loads(receipts_json).items()}
    for asset in assets:
        require_license(receipts[asset.source_id], asset.source_id, "train", ban=None)
    runtime = CampaignRuntime()
    with artifact_prefix(args.artifact_root):
        downloads = {
            asset.dataset_id: download_step(
                HfSource(
                    asset.dataset_id, asset.revision, tuple(file.path for file in asset.files), SourceFormat.PARQUET
                ),
                runtime,
            )
            for asset in assets
        }
        step = ArtifactStep(
            name="rl-data/openresearcher/regional-inputs",
            version=args.artifact_version,
            artifact_type=RegionalInputs,
            deps=list(downloads.values()),
            run=checked_inputs,
            build_config=partial(
                stage_config, assets, receipts_json, downloads, args.source_manifest_sha256, args.license_receipts_sha256
            ),
        )
        if not args.run:
            print(
                json.dumps(
                    {"path": step.path(), "fingerprint": step.fingerprint(), "type": result_type_name(RegionalInputs)}
                )
            )
            return
        with local_client() as client:
            with ZephyrContext(
                client=client,
                max_workers=args.max_workers,
                resources=ResourceConfig(cpu=1, ram="2g"),
                chunk_storage_prefix=str(StoragePath(args.artifact_root) / ".zephyr/openresearcher-assets"),
            ) as context:
                with runtime.activate(context):
                    result = run(step)[0]
        StoragePath(args.output).write_text(result.model_dump_json(indent=2))

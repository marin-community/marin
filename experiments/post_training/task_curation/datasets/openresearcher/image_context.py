# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a regional index context for the approved unprivileged image builder."""

import argparse
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from finestore.rl.licenses import LicenseReceipt, require_license
from iris.cluster.client.job_info import get_job_info
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.openresearcher import OpenResearcherState

from experiments.post_training.task_curation.datasets.openresearcher.assets import (
    CORPUS,
    CORPUS_REVISION,
    DATASET,
    REGIONAL_PREFIX,
    RegionalInputs,
    pinned_document,
)
from experiments.post_training.task_curation.datasets.openresearcher.grounding_pipeline import local_index
from experiments.post_training.task_curation.datasets.openresearcher.index import BM25Index
from experiments.post_training.task_curation.datasets.openresearcher.runtime import (
    IMPLEMENTATION_REVISION,
    TOKENIZER_REVISION,
)
from experiments.post_training.task_curation.environment import PINNED_IMAGE
from experiments.post_training.task_curation.local import artifact_prefix, local_client

MAX_LAYER_BYTES = 6 * 1024**3


class ActorImageContext(Artifact):
    base_image: str
    index_receipt_sha256: str
    index_manifest_sha256: str
    state_sha256: str
    context_sha256: str
    context_bytes: int
    layer_bytes: tuple[int, ...]
    dockerfile_sha256: str


@dataclass(frozen=True)
class ContextConfig:
    base_image: str
    index_receipt: str
    index_receipt_sha256: str
    stage_receipt: str
    stage_receipt_sha256: str
    workdir: str
    output_path: str


def index_layers(directory: Path, destination: Path) -> tuple[int, ...]:
    """Bound each index COPY layer below the registry's ten-gigabyte layer limit."""
    sizes = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_LAYER_BYTES:
            raise ValueError(f"Index file exceeds the layer budget: {path.name}")
        if not sizes or sizes[-1] + size > MAX_LAYER_BYTES:
            sizes.append(0)
        target = destination / f"index-{len(sizes) - 1:04d}" / path.relative_to(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        sizes[-1] += size
    return tuple(sizes)


def prepare_context(config: ContextConfig) -> ActorImageContext:
    info = get_job_info()
    if info is None:
        raise RuntimeError("Stage this large index only inside the approved regional CPU job")
    staged = RegionalInputs.model_validate_json(pinned_document(config.stage_receipt, config.stage_receipt_sha256))
    licenses = json.loads(
        pinned_document(str(StoragePath(staged.path) / "license-receipts.json"), staged.license_receipts_sha256)
    )
    for dataset in (DATASET, CORPUS):
        source_id = "OpenResearcher:" + dataset.split("/")[1]
        require_license(LicenseReceipt.model_validate(licenses[source_id]), source_id, "train", ban=None)
    index = BM25Index.model_validate_json(pinned_document(config.index_receipt, config.index_receipt_sha256))
    if index.stage_receipt_sha256 != config.stage_receipt_sha256 or index.content_mismatches:
        raise ValueError("Actor index is not bound to the approved verified inputs")
    work = Path(config.workdir)
    work.mkdir(parents=True, exist_ok=False)
    context = work / "context"
    context.mkdir()
    local_index(index, work / "index")
    layers = index_layers(work / "index", context)
    state = OpenResearcherState(
        implementation_revision=IMPLEMENTATION_REVISION,
        corpus_revision=CORPUS_REVISION,
        index_manifest_sha256=index.index_manifest_sha256,
        tokenizer_revision=TOKENIZER_REVISION,
        top_k=5,
        snippet_max_tokens=512,
        include_get_document=True,
    )
    (context / "state.json").write_text(state.model_dump_json(indent=2))
    lines = [f"FROM {config.base_image}", "COPY state.json /opt/openresearcher/state.json"]
    lines.extend(f"COPY index-{position:04d}/ /opt/openresearcher/index/bm25/" for position in range(len(layers)))
    (context / "Dockerfile").write_text("\n".join(lines) + "\n")
    archive = work / "context.tar.gz"
    subprocess.run(
        [
            "tar",
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "-I",
            "gzip -1",
            "-cf",
            str(archive),
            "-C",
            str(context),
            ".",
        ],
        check=True,
    )
    sha = file_sha256(StoragePath(str(archive)))
    output = StoragePath(config.output_path)
    target = output / "context.tar.gz"
    with archive.open("rb") as reader, target.open("wb", auto_mkdir=True) as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
    if file_sha256(target) != sha:
        raise ValueError("Regional image context upload differs from its sealed archive")
    (output / "Dockerfile").write_bytes((context / "Dockerfile").read_bytes())
    (output / "state.json").write_bytes((context / "state.json").read_bytes())
    return ActorImageContext(
        path=config.output_path,
        base_image=config.base_image,
        index_receipt_sha256=config.index_receipt_sha256,
        index_manifest_sha256=index.index_manifest_sha256,
        state_sha256=canonical_sha256(state.model_dump(mode="json")),
        context_sha256=sha,
        context_bytes=archive.stat().st_size,
        layer_bytes=layers,
        dockerfile_sha256=file_sha256(StoragePath(str(context / "Dockerfile"))),
    )


def context_config(config: ContextConfig, context: StepContext) -> ContextConfig:
    return replace(config, output_path=context.output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "base-image",
        "index-receipt",
        "index-receipt-sha256",
        "stage-receipt",
        "stage-receipt-sha256",
        "workdir",
        "artifact-root",
        "artifact-version",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if PINNED_IMAGE.fullmatch(args.base_image) is None:
        raise ValueError("Actor runtime image requires an immutable digest")
    StoragePath(args.artifact_root).relative_to(StoragePath(REGIONAL_PREFIX))
    configure_coreweave_s3()
    logging.basicConfig(level=logging.INFO)
    config = ContextConfig(
        args.base_image,
        args.index_receipt,
        args.index_receipt_sha256,
        args.stage_receipt,
        args.stage_receipt_sha256,
        args.workdir,
        "",
    )
    with artifact_prefix(args.artifact_root):
        step = ArtifactStep(
            name="rl-data/openresearcher/actor-image-context",
            version=args.artifact_version,
            artifact_type=ActorImageContext,
            run=prepare_context,
            build_config=partial(context_config, config),
        )
        if not args.run:
            print(json.dumps({"path": step.path(), "fingerprint": step.fingerprint()}))
            return
        with local_client():
            result = run(step)[0]
        StoragePath(args.output).write_text(result.model_dump_json(indent=2))

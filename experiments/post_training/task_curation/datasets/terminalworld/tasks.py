# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import TerminalWorld's full native population under its explicit TRAIN ruling."""

import gzip
import hashlib
import io
import json
import tarfile
from collections.abc import Iterator
from functools import cache
from pathlib import Path, PurePosixPath

from finestore.rl.status_taxonomy import DataRestriction
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.native_harbor import native_task, unpack_native
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

DATASET = "terminalworld/TerminalWorld"
REVISION = "364549435c0fc130eaf0ccfc047a8a698c72643c"
VERSION = "2026.11.22.1"
MANIFEST = "data/full.jsonl.gz"
MANIFEST_SHA256 = "380180fd4b2635776a69e929f4539f25df3399632f215dee3992bed1a3236e51"
USE_RULING = Path(__file__).with_name("intended-use.json")


@cache
def native_manifest(path: str) -> dict[str, dict]:
    """Read and verify the full native selection manifest."""
    raw = StoragePath(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("TerminalWorld full manifest differs from its frozen pin")
    rows = [json.loads(line) for line in gzip.decompress(raw).splitlines() if line.strip()]
    tasks = {row["task_id"]: row for row in rows}
    if len(rows) != 1530 or len(tasks) != len(rows):
        raise ValueError("TerminalWorld full manifest count or task identities changed")
    return tasks


def read_archive(path: StoragePath, context: ConversionContext) -> Iterator[dict]:
    """Join one pinned task archive with its original manifest metadata."""
    task_id = path.name.removesuffix(".tar.gz")
    metadata = native_manifest(str(context.inputs["manifest"] / MANIFEST))[task_id]
    if metadata["artifact_path"] != "artifacts/" + path.name:
        raise ValueError("TerminalWorld archive differs from its manifest reference")
    yield {**metadata, "path": task_id, "task_binary": path.read_bytes()}


def decode_archive(row: dict, context: ConversionContext) -> dict:
    """Remove only the archive's task-directory prefix, preserving member bytes and metadata."""
    prefix = row["task_id"] + "/"
    normalized = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:*") as source,
        tarfile.open(fileobj=normalized, mode="w", format=tarfile.PAX_FORMAT) as target,
    ):
        seen = set()
        for member in source:
            name = member.name.removeprefix("./")
            if name.rstrip("/") == row["task_id"] and member.isdir():
                continue
            if not name.startswith(prefix):
                raise ValueError(f"TerminalWorld member is outside its task root: {name}")
            relative = name.removeprefix(prefix)
            path = PurePosixPath(relative)
            if not relative or path.is_absolute() or ".." in path.parts or relative in seen:
                raise ValueError(f"Unsafe or duplicate TerminalWorld archive member: {name}")
            seen.add(relative)
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError(f"Unsupported TerminalWorld archive member: {name}")
            entry = member.replace(name=relative, deep=True)
            if "path" in entry.pax_headers:
                entry.pax_headers["path"] = relative
            target.addfile(entry, source.extractfile(member) if member.isfile() else None)
    data = unpack_native({**row, "task_binary": normalized.getvalue()}, context)
    if data["instruction"] != row["instruction"]:
        raise ValueError("TerminalWorld native instruction differs from its manifest")
    return {**data, "archive_sha256": hashlib.sha256(row["task_binary"]).hexdigest()}


def sources() -> list[RlDataSource[CurationRecipe]]:
    ruling = json.loads(USE_RULING.read_text())
    return [
        RlDataSource(
            info=SourceInfo(
                origin=SourceOrigin.TERMINALWORLD,
                upstream_dataset="TerminalWorld",
                upstream_name="TerminalWorld full",
                upstream_url=f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
                build_path="Native Harbor task packages; full config",
                count=1530,
                family="terminal",
                task_format=TaskFormat.HARBOR,
                data_restriction=DataRestriction.OPEN,
                tags=("license:cc-by-4.0",),
                notes=(
                    f"TRAIN under {ruling['authority']}: {ruling['reason']} "
                    "The upstream config is full and its named split remains test. Verified (200) and sample (20) "
                    "are exact subsets and are not appended as separate training populations. Native prompts, "
                    "canary comments, task.toml budgets, environment, tests and original solutions remain unchanged. "
                    f"Use ruling: {USE_RULING.relative_to(Path(__file__).parents[5])}."
                ),
            ),
            config=CurationRecipe(
                name="terminalworld-full-train",
                version=VERSION,
                source=HfSource(
                    DATASET,
                    REVISION,
                    ("artifacts/*.tar.gz",),
                    SourceFormat.GENERATED,
                    read=read_archive,
                    decode=decode_archive,
                ),
                inputs={"manifest": HfSource(DATASET, REVISION, (MANIFEST,), SourceFormat.JSONL)},
                convert=native_task,
                intended_use=IntendedUse.TRAIN,
                oracle_solutions=True,
            ),
            pipeline=process_rows,
        )
    ]

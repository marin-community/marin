# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare Exp5 additions on CPUs without admitting the blocked research slot."""

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
from iris.cluster.client.job_info import get_job_info
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.experiment.cli import build_options
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow

from experiments.post_training.long_loop.exp4_data import POOL_SCHEMA
from experiments.post_training.long_loop.rl_dataset_admission import file_digest
from experiments.post_training.task_curation.datasets.skyrl.math import convert_hardmath
from experiments.post_training.task_curation.exp5_curriculum import (
    HARDMATH_FILE,
    HARDMATH_INPUT_ROWS,
    HARDMATH_LABEL,
    HARDMATH_REPO,
    HARDMATH_REVISION,
    HARDMATH_ROWS,
    HARDMATH_SHA256,
    IFBENCH_LABEL,
    IFBENCH_REPO,
    IFBENCH_REVISION,
    IFBENCH_ROWS,
    IFBENCH_SHA256,
    IFBENCH_SOURCE,
    IFBENCH_SOURCE_REVISION,
    PENDING_SOURCES,
    TARGET_SHARES,
    exp5_rl_curriculum,
)

OUTPUT_PREFIX = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp5"
BASE_PREFIX = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp4/long-loop-exp4"
BASE_VERSION = "2026.11.18.71"
AGENTIC_MANIFEST_SHA256 = "ea12c6d6956079b32f5e638f103ae5c6866ab2fccc8ed9611dd2ed9dcd7f98ea"
NONAGENTIC_MANIFEST_SHA256 = "6fbdb302de1d4827fd1181959e4b42ba2c400086819aac86c52e18bc93aba841"
NONAGENTIC_SHA256 = "4cb4fd8206a113509e6fcd9c0c4b1198c6d925eb964b1622c98d12ef36151402"
IFBENCH_URI = f"hf://datasets/{IFBENCH_REPO}@{IFBENCH_REVISION}/tasks.parquet"
HARDMATH_URI = f"hf://datasets/{HARDMATH_REPO}@{HARDMATH_REVISION}/{HARDMATH_FILE}"


def base_pool(route: str) -> str:
    return f"{BASE_PREFIX}/rl-pool-{route}/{BASE_VERSION}"


def ifbench_row(row: dict) -> tuple[dict, Counter]:
    """Keep the admitted IF contract; reject any nonempty runtime or allocation."""
    raw = json.loads(row["task_spec"])
    source = raw["source"]
    if (source["dataset"], source["revision"]) != (IFBENCH_SOURCE, IFBENCH_SOURCE_REVISION):
        raise ValueError("IFBench row differs from the admitted upstream pin")
    if json.loads(row.get("runtime") or "{}"):
        raise ValueError("IFBench requires an adapter for its nonempty runtime")
    removed = Counter()
    for field in ("integrity_contract", "execution_requirements"):
        if field in raw:
            if raw.pop(field) is not None:
                raise ValueError(f"Cannot discard IFBench {field}")
            removed[field] += 1
    serialized = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    TaskSpec.model_validate_json(serialized)
    return {"task_spec": serialized, "teacher_route": IFBENCH_LABEL}, removed


def hardmath_task(data: dict, index: int) -> TaskSpec | ImportRejection:
    """Use the base converter on every input row, without a difficulty predicate."""
    return convert_hardmath(
        RawRow(
            id=f"hardmath:{index}",
            source=Source(
                dataset=HARDMATH_REPO,
                revision=HARDMATH_REVISION,
                row=f"{HARDMATH_FILE}:{index}",
                importer_revision="1",
            ),
            data=data,
        ),
        ConversionContext(inputs={}, grader_environment=None),
    )


def local_input(uri: str, expected_sha256: str, destination: Path) -> Path:
    with StoragePath(uri).open("rb") as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    if file_digest(destination) != expected_sha256:
        raise ValueError(f"Exp5 input digest changed: {uri}")
    return destination


def prepare_pool(*, output: str, baseline: str, ifbench: str, hardmath: str) -> None:
    """Publish a verified pool and preserve every base-HARDMath conversion rejection."""
    if get_job_info() is None:
        raise RuntimeError("Prepare Exp5 data inside the regional Iris CPU job")
    base_manifest_bytes = (StoragePath(base_pool("nonagentic")) / "manifest.json").read_bytes()
    if hashlib.sha256(base_manifest_bytes).hexdigest() != NONAGENTIC_MANIFEST_SHA256:
        raise ValueError("Exp4 nonagentic source manifest changed")
    base_manifest = json.loads(base_manifest_bytes)
    counts = Counter()
    null_fields = Counter()
    rejects = []
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        base_file = local_input(baseline, NONAGENTIC_SHA256, directory / "base.parquet")
        if_file = local_input(ifbench, IFBENCH_SHA256, directory / "ifbench.parquet")
        hard_file = local_input(hardmath, HARDMATH_SHA256, directory / "hardmath.parquet")
        destination = directory / "train.parquet"
        with pq.ParquetWriter(destination, POOL_SCHEMA, compression="zstd") as writer:
            for batch in pq.ParquetFile(base_file).iter_batches(batch_size=128):
                rows = batch.to_pylist()
                for row in rows:
                    TaskSpec.model_validate_json(row["task_spec"])
                    counts[row["teacher_route"]] += 1
                writer.write_table(pa.Table.from_pylist(rows, schema=POOL_SCHEMA))
            if dict(counts) != base_manifest["source_counts"]:
                raise ValueError("Exp4 population changed while copying into Exp5")
            for batch in pq.ParquetFile(if_file).iter_batches(batch_size=128):
                rows = []
                for row in batch.to_pylist():
                    stamped, removed = ifbench_row(row)
                    rows.append(stamped)
                    null_fields.update(removed)
                counts[IFBENCH_LABEL] += len(rows)
                writer.write_table(pa.Table.from_pylist(rows, schema=POOL_SCHEMA))
            raw_math = pq.read_table(hard_file).to_pylist()
            if len(raw_math) != HARDMATH_INPUT_ROWS:
                raise ValueError("HARDMath base input population changed")
            math_rows = []
            for index, data in enumerate(raw_math):
                task = hardmath_task(data, index)
                if isinstance(task, ImportRejection):
                    if task.kind != ImportFailureKind.SOURCE_DEFECT or task.reason != "invalid_reference":
                        raise ValueError(f"Unexpected HARDMath conversion failure: {task}")
                    rejects.append({"source_row": index, **task.model_dump(mode="json")})
                    continue
                math_rows.append({"task_spec": task.model_dump_json(), "teacher_route": HARDMATH_LABEL})
            counts[HARDMATH_LABEL] = len(math_rows)
            writer.write_table(pa.Table.from_pylist(math_rows, schema=POOL_SCHEMA))
        if dict(counts) != {
            **base_manifest["source_counts"],
            IFBENCH_LABEL: IFBENCH_ROWS,
            HARDMATH_LABEL: HARDMATH_ROWS,
        }:
            raise ValueError("Exp5 prepared source counts differ from the pinned populations")
        root = StoragePath(output)
        with destination.open("rb") as source, (root / "train.parquet").open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        digest = file_digest(destination)
        with (root / "train.parquet").open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                raise ValueError("Exp5 pool readback failed")
    (root / "hardmath-rejections.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rejects))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "curriculum": exp5_rl_curriculum().identity,
                "target_shares": dict(TARGET_SHARES),
                "pending_sources": dict(PENDING_SOURCES),
                "training_ready": False,
                "rows": sum(counts.values()),
                "source_counts": dict(counts),
                "sha256": digest,
                "baseline_manifest_sha256": NONAGENTIC_MANIFEST_SHA256,
                "inputs": {baseline: NONAGENTIC_SHA256, ifbench: IFBENCH_SHA256, hardmath: HARDMATH_SHA256},
                "ifbench_removed_null_fields": dict(null_fields),
                "hardmath": {
                    "variant": "base",
                    "difficulty_filter": None,
                    "input_rows": HARDMATH_INPUT_ROWS,
                    "converted_rows": HARDMATH_ROWS,
                    "source_defects": len(rejects),
                    "rejections_file": "hardmath-rejections.jsonl",
                },
            },
            indent=2,
        )
        + "\n"
    )


def pool_step(version: str) -> ArtifactStep:
    baseline = ArtifactStep.adopt(
        "long-loop-exp5/inputs/exp4-nonagentic",
        BASE_VERSION,
        source=base_pool("nonagentic") + "/train.parquet",
        config={"sha256": NONAGENTIC_SHA256, "manifest_sha256": NONAGENTIC_MANIFEST_SHA256},
    )
    ifbench = ArtifactStep.adopt(
        "long-loop-exp5/inputs/ifbench",
        version,
        source=IFBENCH_URI,
        config={"revision": IFBENCH_REVISION, "sha256": IFBENCH_SHA256, "rows": IFBENCH_ROWS},
    )
    hardmath = ArtifactStep.adopt(
        "long-loop-exp5/inputs/hardmath-base",
        version,
        source=HARDMATH_URI,
        config={"revision": HARDMATH_REVISION, "sha256": HARDMATH_SHA256, "rows": HARDMATH_INPUT_ROWS},
    )
    return apply(
        "long-loop-exp5/rl-pool-nonagentic",
        prepare_pool,
        version=version,
        output=OUT,
        baseline=baseline,
        ifbench=ifbench,
        hardmath=hardmath,
    )


@click.command(help=__doc__)
@build_options
def main() -> ArtifactStep:
    configure_coreweave_s3()
    os.environ["MARIN_PREFIX"] = OUTPUT_PREFIX
    return pool_step(resolve_version("exp5-rl-data", None))


if __name__ == "__main__":
    main()

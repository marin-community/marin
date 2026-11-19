# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stamp admitted Exp4 task populations for the curriculum, in regional CPU workers."""

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.experiment.cli import build_options
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_RELEASE_URL,
    AGENTIC_TRAIN_SHA256,
    NONAGENTIC_RELEASE_URL,
    NONAGENTIC_TRAIN_SHA256,
)
from experiments.post_training.task_curation.exp4_curriculum import CARRY_OVER_SHARES, exp4_rl_curriculum

NEMOTRON_URI = (
    "hf://datasets/open-athena/marinskyrl-nemotron-if-bfb072d2-retained-20261113-v1"
    "@07de9679afd9afd0f46d30fd4ddccce3ef7ec98e/tasks.parquet"
)
NEMOTRON_SHA256 = "e6ee366e4ab4e181713c053c88d6819c1408b6399e4bb6c6bbd2dfb2af724ece"
NEMOTRON_SOURCE = "nvidia/Llama-Nemotron-Post-Training-Dataset"
NEMOTRON_REVISION = "ab2a40d258a6a4d9d4c277d702aeea445081766c"
POOL_SCHEMA = pa.schema(
    [
        pa.field("task_spec", pa.string(), nullable=False),
        pa.field("teacher_route", pa.string(), nullable=False),
    ]
)


def stamped_rows(rows: list[dict], *, nemotron: bool) -> tuple[list[dict], int]:
    """Preserve task content and label its curriculum population; reject unknown contracts."""
    output = []
    removed_null_contracts = 0
    for row in rows:
        serialized = row["task_spec"]
        task = json.loads(serialized)
        if nemotron:
            if task["source"]["dataset"] != NEMOTRON_SOURCE or task["source"]["revision"] != NEMOTRON_REVISION:
                raise ValueError("Admitted Nemotron row differs from the curriculum's upstream pin")
            # Atlas records an optional delivery contract. The consumer has no such
            # field: only an explicitly null contract has no semantics to migrate.
            if "integrity_contract" in task:
                if task.pop("integrity_contract") is not None:
                    raise ValueError("Exp4 cannot discard an Atlas delivery contract")
                removed_null_contracts += 1
                serialized = json.dumps(task, ensure_ascii=False, separators=(",", ":"))
            if json.loads(row.get("runtime") or "{}"):
                raise ValueError("Exp4 requires an explicit adapter for a nonempty Atlas runtime")
            route = "nemotron_if"
        else:
            route = task["source"]["dataset"]
        TaskSpec.model_validate_json(serialized)
        output.append({"task_spec": serialized, "teacher_route": route})
    return output, removed_null_contracts


def prepare_pool(*, output: str, sources: list[dict]) -> None:
    """Verify complete source hashes before writing a count-checked, labeled pool."""
    curriculum = exp4_rl_curriculum()
    counts = Counter()
    inputs = []
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        destination = directory / "train.parquet"
        with pq.ParquetWriter(destination, POOL_SCHEMA, compression="zstd") as writer:
            for index, source in enumerate(sources):
                local = directory / f"source-{index}.parquet"
                digest = hashlib.sha256()
                with StoragePath(source["uri"]).open("rb") as reader, local.open("wb") as target:
                    while chunk := reader.read(8 * 1024 * 1024):
                        digest.update(chunk)
                        target.write(chunk)
                if digest.hexdigest() != source["sha256"]:
                    raise ValueError("Exp4 pool input hash differs from its admitted release")
                count = 0
                null_contracts = 0
                for batch in pq.ParquetFile(local).iter_batches(batch_size=128):
                    rows, normalized = stamped_rows(batch.to_pylist(), nemotron=source["nemotron"])
                    counts.update(row["teacher_route"] for row in rows)
                    count += len(rows)
                    null_contracts += normalized
                    writer.write_table(pa.Table.from_pylist(rows, schema=POOL_SCHEMA))
                if count != source["rows"]:
                    raise ValueError(f"Exp4 input count changed: {count} != {source['rows']}")
                inputs.append({**source, "verified_rows": count, "removed_null_contracts": null_contracts})
                local.unlink()
        expected = {source["label"]: source["rows"] for source in sources if source["nemotron"]}
        if any(source["rows"] == 100000 for source in sources):
            expected.update({label: round(weight * 200000) for label, weight in CARRY_OVER_SHARES.items()})
        if any(source["rows"] == 5608 for source in sources):
            expected["long-loop-smartie-swesmith"] = 5608
        if dict(counts) != expected:
            raise ValueError(f"Exp4 pool composition differs: {dict(counts)} != {expected}")
        with destination.open("rb") as reader:
            raw_hash = hashlib.file_digest(reader, "sha256").hexdigest()
        with destination.open("rb") as reader, (StoragePath(output) / "train.parquet").open("wb") as target:
            while chunk := reader.read(8 * 1024 * 1024):
                target.write(chunk)
        # Read back the published file; the receipt certifies bytes at the destination.
        with (StoragePath(output) / "train.parquet").open("rb") as reader:
            if hashlib.file_digest(reader, "sha256").hexdigest() != raw_hash:
                raise ValueError("Exp4 pool readback failed")
        (StoragePath(output) / "manifest.json").write_text(
            json.dumps(
                {
                    "curriculum": curriculum.identity,
                    "inputs": inputs,
                    "rows": sum(counts.values()),
                    "source_counts": dict(counts),
                    "sha256": raw_hash,
                    "task_content_changes": (
                        "Only absent-consumer Atlas integrity_contract=null removed; all originals pinned"
                    ),
                },
                indent=2,
            )
        )


def pool_steps(version: str) -> dict[str, ArtifactStep]:
    sources = {
        "agentic": [
            {
                "uri": AGENTIC_RELEASE_URL + "/agentic-train.parquet",
                "sha256": AGENTIC_TRAIN_SHA256,
                "rows": 5608,
                "nemotron": False,
                "label": "long-loop-smartie-swesmith",
            }
        ],
        "nonagentic": [
            {
                "uri": NONAGENTIC_RELEASE_URL + "/nonagentic-train.parquet",
                "sha256": NONAGENTIC_TRAIN_SHA256,
                "rows": 100000,
                "nemotron": False,
                "label": "carry-over",
            },
            {"uri": NEMOTRON_URI, "sha256": NEMOTRON_SHA256, "rows": 54953, "nemotron": True, "label": "nemotron_if"},
        ],
    }
    steps = {}
    for route, members in sources.items():
        deps = tuple(
            ArtifactStep.adopt(
                f"long-loop-exp4/admitted/{route}-{index}",
                version,
                source=member["uri"],
                config=member,
            )
            for index, member in enumerate(members)
        )
        step = apply(
            f"long-loop-exp4/rl-pool-{route}",
            prepare_pool,
            version=version,
            output=OUT,
            sources=[{**member, "uri": dep} for member, dep in zip(members, deps, strict=True)],
        )
        steps[route] = step
    return steps


@click.command(help=__doc__)
@build_options
def main() -> dict[str, ArtifactStep]:
    configure_coreweave_s3()
    os.environ["MARIN_PREFIX"] = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp4"
    return pool_steps(resolve_version("exp4-rl-data", None))


if __name__ == "__main__":
    main()

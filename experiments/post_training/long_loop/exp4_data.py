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
from marin.rl.curriculum import PoolRow, validation_payload
from taskcompendium.identity import canonical_sha256
from marin.experiment.cli import build_options
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_RELEASE_URL,
    AGENTIC_TRAIN_SHA256,
    AGENTIC_TRAIN_SOURCE,
    NONAGENTIC_RELEASE_URL,
    NONAGENTIC_TRAIN_SHA256,
    TEACHER_ROUTE_LABELS,
)
from experiments.post_training.task_curation.exp4_curriculum import (
    CARRY_OVER_SHARES,
    NEMOTRON_IF_LABEL,
    SOURCE_ROUTES,
    exp4_rl_curriculum,
    exp4_validation_split,
)

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
            route = NEMOTRON_IF_LABEL
        else:
            dataset = task["source"]["dataset"]
            if dataset not in TEACHER_ROUTE_LABELS:
                raise ValueError(f"Exp4 pool row carries an unadmitted data_source: {dataset!r}")
            route = TEACHER_ROUTE_LABELS[dataset]
        TaskSpec.model_validate_json(serialized)
        output.append({"task_spec": serialized, "teacher_route": route})
    return output, removed_null_contracts


def _write_pool_file(destination: StoragePath, name: str, rows: list[dict]) -> str:
    """Write one stamped-schema parquet under the pool artifact and return its verified digest."""
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary) / "file.parquet"
        with pq.ParquetWriter(local, POOL_SCHEMA, compression="zstd") as writer:
            writer.write_table(pa.Table.from_pylist(rows, schema=POOL_SCHEMA))
        with local.open("rb") as reader:
            digest = hashlib.file_digest(reader, "sha256").hexdigest()
        target = destination / name
        target.parent.mkdirs(exist_ok=True)
        with local.open("rb") as reader, target.open("wb") as writer:
            while chunk := reader.read(8 * 1024 * 1024):
                writer.write(chunk)
        with target.open("rb") as reader:
            if hashlib.file_digest(reader, "sha256").hexdigest() != digest:
                raise ValueError(f"{target}: readback differs from the written bytes")
        return digest


def prepare_pool(*, output: str, sources: list[dict]) -> None:
    """Stamp the admitted Exp4 populations, draw the curriculum's holdout, and write the pools.

    One step over every source, so the holdout draw spans all eight labels: the
    stamped rows (route-labeled by ``teacher_route``, the reasoning-gym wheel URL
    renamed to its source identity) split into per-source training pools that
    exclude the holdout, the holdout's ``validation.parquet``, and a manifest
    recording the split, the per-file digests, and the fixed curriculum's
    identity. The launch reads the holdout back from the manifest; the training
    pools carry no validation prompts."""
    counts = Counter()
    stamped: list[tuple[str, str]] = []
    inputs = []
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
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
                stamped.extend((row["teacher_route"], row["task_spec"]) for row in rows)
                count += len(rows)
                null_contracts += normalized
            if count != source["rows"]:
                raise ValueError(f"Exp4 input count changed: {count} != {source['rows']}")
            inputs.append({**source, "verified_rows": count, "removed_null_contracts": null_contracts})
            local.unlink()
    expected = {source["label"]: source["rows"] for source in sources if source["nemotron"]}
    if any(source["rows"] == 100000 for source in sources):
        expected.update({label: round(weight * 200000) for label, weight in CARRY_OVER_SHARES.items()})
    if any(source["rows"] == 5608 for source in sources):
        expected[AGENTIC_TRAIN_SOURCE] = 5608
    if dict(counts) != expected:
        raise ValueError(f"Exp4 pool composition differs: {dict(counts)} != {expected}")
    pool_rows = {
        label: tuple(
            PoolRow(id=json.loads(raw)["id"], sha256=canonical_sha256(json.loads(raw)))
            for route, raw in stamped
            if route == label
        )
        for label in sorted(counts)
    }
    split = exp4_validation_split(pool_rows)
    curriculum = exp4_rl_curriculum(split.validation)
    destination = StoragePath(output)
    destination.mkdirs(exist_ok=True)
    wanted = {(row.label, row.id): row.sha256 for row in split.validation.rows}
    train_ids = {label: {row.id for row in rows} for label, rows in split.training.items()}
    train_rows: dict[str, list[dict]] = {label: [] for label in sorted(train_ids)}
    held: list[dict] = []
    for route, raw in stamped:
        task = json.loads(raw)
        key = (route, task["id"])
        if key in wanted:
            if canonical_sha256(task) != wanted[key]:
                raise ValueError(f"holdout row {route!r}/{task['id']!r} differs from its recorded sha256")
            held.append({"task_spec": raw, "teacher_route": route})
        elif task["id"] in train_ids[route]:
            train_rows[route].append({"task_spec": raw, "teacher_route": route})
        else:
            raise ValueError(f"Exp4 pool row {route!r}/{task['id']!r} landed on neither side of the holdout")
    files = {f"{label}/train.parquet": _write_pool_file(destination, f"{label}/train.parquet", rows) for label, rows in train_rows.items()}
    files["validation.parquet"] = _write_pool_file(destination, "validation.parquet", held)
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "curriculum": curriculum.identity,
                "validation": validation_payload(split.validation),
                "composition": dict(sorted(counts.items())),
                "train_composition": {label: len(rows) for label, rows in sorted(train_rows.items())},
                "routes": dict(sorted(SOURCE_ROUTES.items())),
                "files": files,
                "inputs": inputs,
                "rows": sum(counts.values()),
                "task_content_changes": (
                    "Only absent-consumer Atlas integrity_contract=null removed; all originals pinned"
                ),
            },
            indent=2,
        )
    )


def pool_steps(version: str) -> dict[str, ArtifactStep]:
    """The Exp4 pool artifact: per-source holdout-excluding pools plus the holdout's validation data.

    One step stamps every admitted source, draws the curriculum's holdout, and
    writes ``{label}/train.parquet`` for each source, ``validation.parquet`` for
    the holdout, and the manifest that records the split and the fixed
    curriculum's identity."""
    sources = [
        {
            "uri": prefix_join(AGENTIC_RELEASE_URL, "agentic-train.parquet"),
            "sha256": AGENTIC_TRAIN_SHA256,
            "rows": 5608,
            "nemotron": False,
            "label": AGENTIC_TRAIN_SOURCE,
        },
        {
            "uri": prefix_join(NONAGENTIC_RELEASE_URL, "nonagentic-train.parquet"),
            "sha256": NONAGENTIC_TRAIN_SHA256,
            "rows": 100000,
            "nemotron": False,
            "label": "carry-over",
        },
        {"uri": NEMOTRON_URI, "sha256": NEMOTRON_SHA256, "rows": 54953, "nemotron": True, "label": NEMOTRON_IF_LABEL},
    ]
    deps = tuple(
        ArtifactStep.adopt(f"long-loop-exp4/admitted/{index}", version, source=member["uri"], config=member)
        for index, member in enumerate(sources)
    )
    step = apply(
        "long-loop-exp4/rl-pools",
        prepare_pool,
        version=version,
        output=OUT,
        sources=[{**member, "uri": dep} for member, dep in zip(sources, deps, strict=True)],
    )
    return {"pools": step}


@click.command(help=__doc__)
@build_options
def main() -> dict[str, ArtifactStep]:
    configure_coreweave_s3()
    os.environ["MARIN_PREFIX"] = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp4"
    return pool_steps(resolve_version("exp4-rl-data", None))


if __name__ == "__main__":
    main()

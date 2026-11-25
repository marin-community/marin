# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stamp the Exp5 pools: the eleven-source mix, drawn holdout, and the training-ready manifest.

Exp5 extends the Exp4 population with IFBench, the thin OpenResearcher service variant, and the
unfiltered HARDMath base conversion. The adopted agentic and blend rows are copied byte for byte —
including the regenerated Reasoning Gym label the Exp4 stamp produces — and every new source is
converted with its admitted recipe. The manifest records, per source, whether the pool carries the
full admitted version or a subsample with the upstream admitted row count, the grader-version
provenance, and ``training_ready``.
"""

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import click
import pyarrow.parquet as pq
from iris.cluster.client.job_info import get_job_info
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.experiment.cli import build_options
from marin.rl.curriculum import PoolRow, validation_payload
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, Source, TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow

from experiments.post_training.long_loop.exp4_data import (
    GENERATED_POOL_ROWS,
    GENERATED_V3_FULL_RUN_ROWS,
    NEMOTRON_IF_LABEL,
    NEMOTRON_IF_RECIPE_VERSION,
    NEMOTRON_JSONL_URL,
    NEMOTRON_KEPT_ROWS,
    NEMOTRON_REJECTED_ROWS,
    NEMOTRON_REVISION,
    NEMOTRON_SOURCE,
    NEMOTRON_UPSTREAM_ROWS,
    _regenerate_generated_label,
    _write_pool_file,
    row_identity,
)
from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.long_loop_rl import AGENTIC_TRAIN_SOURCE, TEACHER_ROUTE_LABELS
from experiments.post_training.task_curation.datasets.reasoning_gym.tasks import GENERATOR_URL
from experiments.post_training.task_curation.datasets.skyrl.ifeval import convert_nemotron_if
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
    OPENRESEARCHER_LABEL,
    OPENRESEARCHER_SOURCE,
    exp5_rl_curriculum,
    exp5_validation_split,
)
from experiments.post_training.task_curation.images.build import environment_artifact

OUTPUT_PREFIX = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp5"
AGENTIC_RELEASE_URL = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/rl-data/swe-verified-20261112-v4/release"
AGENTIC_TRAIN_FILE = "agentic-train.parquet"
AGENTIC_TRAIN_SHA256 = "453552f9d4e1fa27710f9434a303f7ab5ff390c22deb54572f9be885a2a80f5d"
AGENTIC_ROWS = 5608
NONAGENTIC_RELEASE_URL = "s3://marin-us-east-02a/marin/users/benfeuer/long-loop/rl-data/smartie-mixed-20261110-v1"
NONAGENTIC_TRAIN_FILE = "nonagentic-train.parquet"
NONAGENTIC_TRAIN_SHA256 = "c8948b3c13957ee054b4b8b5553127f75c44a50577b5b2b798261e9f8a5abc45"
BLEND_ROWS = 100_000
IFBENCH_URI = f"hf://datasets/{IFBENCH_REPO}@{IFBENCH_REVISION}/tasks.parquet"
HARDMATH_URI = f"hf://datasets/{HARDMATH_REPO}@{HARDMATH_REVISION}/{HARDMATH_FILE}"
OPENRESEARCHER_SERVICE_REPO = "open-athena/openresearcher-openresearcher-dataset-service-20261121-v1"
OPENRESEARCHER_SERVICE_REVISION = "0c4f39584d7f0121e97f6ad4f4761749add56fef"
OPENRESEARCHER_SERVICE_PARQUET_SHA256 = "346e892a7052d1f93ee31ae208100adccee06ff1918344efe2974fabc331226c"
OPENRESEARCHER_SERVICE_URI = (
    f"hf://datasets/{OPENRESEARCHER_SERVICE_REPO}@{OPENRESEARCHER_SERVICE_REVISION}/train.parquet"
)
OPENRESEARCHER_DATASET = OPENRESEARCHER_SOURCE
OPENRESEARCHER_SERVICE_IMPORTER_REVISION = "2026.11.21.service.1"
OPENRESEARCHER_ROWS = 283
HARDMATH_REJECTIONS = HARDMATH_INPUT_ROWS - HARDMATH_ROWS
GENERATED_LABEL = TEACHER_ROUTE_LABELS[GENERATOR_URL]
NONAGENTIC_ROUTE = "nonagentic"
ROUTES = {
    **{label: NONAGENTIC_ROUTE for label in TEACHER_ROUTE_LABELS.values()},
    AGENTIC_TRAIN_SOURCE: "agentic",
    NEMOTRON_IF_LABEL: NONAGENTIC_ROUTE,
    IFBENCH_LABEL: NONAGENTIC_ROUTE,
    OPENRESEARCHER_LABEL: NONAGENTIC_ROUTE,
    HARDMATH_LABEL: NONAGENTIC_ROUTE,
}
"""Each pool label and the rollout route the buffer admits it through."""
EXPECTED_COMPOSITION = {
    AGENTIC_TRAIN_SOURCE: AGENTIC_ROWS,
    "PRIME-RL/Eurus-2-RL-Data": 10_317,
    "nvidia/Nemotron-RL-Ultra-Training-Blends": 12_426,
    "nvidia/OpenScience": 20_790,
    GENERATED_LABEL: GENERATED_POOL_ROWS,
    "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": 11_634,
    "open-thoughts/TaskTrove": 24_033,
    NEMOTRON_IF_LABEL: NEMOTRON_KEPT_ROWS,
    IFBENCH_LABEL: IFBENCH_ROWS,
    HARDMATH_LABEL: HARDMATH_ROWS,
    OPENRESEARCHER_LABEL: OPENRESEARCHER_ROWS,
}


def ifbench_row(row: dict) -> tuple[dict, Counter]:
    """Keep the admitted IF contract; reject any nonempty runtime or allocation."""
    raw = json.loads(row["task_spec"])
    source = raw["source"]
    if (source["dataset"], source["revision"]) != (IFBENCH_SOURCE, IFBENCH_SOURCE_REVISION):
        raise ValueError("IFBench row differs from the admitted upstream pin")
    if json.loads(row.get("runtime") or "{}"):
        raise ValueError("IFBench requires an adapter for its nonempty runtime")
    removed: Counter = Counter()
    for field in ("integrity_contract", "execution_requirements"):
        if field in raw:
            if raw.pop(field) is not None:
                raise ValueError(f"Cannot discard IFBench {field}")
            removed[field] += 1
    serialized = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    TaskSpec.model_validate_json(serialized)
    return {"task_spec": serialized, "teacher_route": IFBENCH_LABEL}, removed


def openresearcher_row(row: dict) -> dict:
    """Keep the thin service contract; drop only the null Atlas delivery contract."""
    raw = json.loads(row["task_spec"])
    source = raw["source"]
    if source["dataset"] != OPENRESEARCHER_DATASET or source["importer_revision"] != (
        OPENRESEARCHER_SERVICE_IMPORTER_REVISION
    ):
        raise ValueError("OpenResearcher row differs from the admitted thin service pin")
    if json.loads(row.get("runtime") or "{}"):
        raise ValueError("OpenResearcher requires an adapter for its nonempty runtime")
    if raw.pop("integrity_contract", None) is not None:
        raise ValueError("Cannot discard an OpenResearcher delivery contract")
    serialized = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    TaskSpec.model_validate_json(serialized)
    return {"task_spec": serialized, "teacher_route": OPENRESEARCHER_LABEL}


def hardmath_task(data: dict, index: int) -> TaskSpec | ImportRejection:
    """Convert every base HARDMath row without selecting by difficulty."""
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


def _regenerated_rows_file(parquet_path: Path, packages_lock: str) -> tuple[list[dict], dict]:
    """Re-stamp the blend's generated label through the current pipeline, exactly as the Exp4 stamp does."""
    generated_old: dict[int, dict] = {}
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=128):
        for row in batch.to_pylist():
            task = json.loads(row["task_spec"])
            TaskSpec.model_validate_json(row["task_spec"])
            if TEACHER_ROUTE_LABELS[task["source"]["dataset"]] == GENERATED_LABEL:
                generated_old[int(task["source"]["row"].rsplit(":", 1)[-1])] = row
    rows, provenance = _regenerate_generated_label(generated_old, packages_lock)
    return rows, provenance


def prepare_pool(
    *,
    output: str,
    agentic: str,
    blend: str,
    nemotron: str,
    ifbench: str,
    openresearcher: str,
    hardmath: str,
    grader_environment: str,
) -> None:
    """Stamp the eleven-label pools and record the drawn holdout and the training-ready manifest."""
    if get_job_info() is None:
        raise RuntimeError("Prepare Exp5 pools inside the regional Iris CPU job")
    packages_lock = prefix_join(grader_environment, "requirements.lock")
    counts: Counter = Counter()
    stamped: list[tuple[str, str]] = []
    nemotron_rejections: list[dict] = []
    hardmath_rejections: list[dict] = []
    generated_provenance: dict = {}
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for index, (uri, digest, name) in enumerate(
            (
                (agentic, AGENTIC_TRAIN_SHA256, AGENTIC_TRAIN_FILE),
                (blend, NONAGENTIC_TRAIN_SHA256, NONAGENTIC_TRAIN_FILE),
                (nemotron, None, "instruction_following.jsonl"),
                (ifbench, IFBENCH_SHA256, "tasks.parquet"),
                (openresearcher, OPENRESEARCHER_SERVICE_PARQUET_SHA256, "train.parquet"),
                (hardmath, HARDMATH_SHA256, HARDMATH_FILE),
            )
        ):
            local = directory / f"input-{index}"
            with StoragePath(uri).open("rb") as source, local.open("wb") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    target.write(chunk)
            if digest is not None:
                with local.open("rb") as reader:
                    observed = hashlib.file_digest(reader, "sha256").hexdigest()
                if observed != digest:
                    raise ValueError(f"{name} differs from its admitted digest: {observed}")
            if index == 0:
                for batch in pq.ParquetFile(local).iter_batches(batch_size=128):
                    for row in batch.to_pylist():
                        task = json.loads(row["task_spec"])
                        TaskSpec.model_validate_json(row["task_spec"])
                        route = TEACHER_ROUTE_LABELS[task["source"]["dataset"]]
                        counts[route] += 1
                        stamped.append((route, row["task_spec"]))
            elif index == 1:
                rows, generated_provenance = _regenerated_rows_file(local, packages_lock)
                for batch in pq.ParquetFile(local).iter_batches(batch_size=128):
                    for row in batch.to_pylist():
                        task = json.loads(row["task_spec"])
                        TaskSpec.model_validate_json(row["task_spec"])
                        route = TEACHER_ROUTE_LABELS[task["source"]["dataset"]]
                        if route == GENERATED_LABEL:
                            continue
                        counts[route] += 1
                        stamped.append((route, row["task_spec"]))
                for row in rows:
                    counts[row["teacher_route"]] += 1
                    stamped.append((row["teacher_route"], row["task_spec"]))
            elif index == 2:
                upstream = 0
                # Byte-level line splitting: the Nemotron text embeds Unicode separators that
                # str.splitlines would treat as line breaks inside JSON strings.
                for raw_line in local.read_bytes().split(b"\n"):
                    if not raw_line.strip():
                        continue
                    upstream += 1
                    source = Source(
                        dataset=NEMOTRON_SOURCE,
                        revision=NEMOTRON_REVISION,
                        row=f"RL/instruction_following/instruction_following.jsonl:{upstream - 1}",
                        importer_revision=NEMOTRON_IF_RECIPE_VERSION,
                    )
                    raw = RawRow(id=row_identity("nemotron_if", source), source=source, data=json.loads(raw_line))
                    task = convert_nemotron_if(
                        raw,
                        ConversionContext(
                            inputs={},
                            grader_environment=EnvironmentRequirements(
                                command_semantics=CommandSemantics.LINUX_PROCESS,
                                packages_lock=packages_lock,
                            ),
                        ),
                    )
                    if isinstance(task, ImportRejection):
                        nemotron_rejections.append({"row": upstream - 1, **task.model_dump(mode="json")})
                        continue
                    counts[NEMOTRON_IF_LABEL] += 1
                    stamped.append((NEMOTRON_IF_LABEL, task.model_dump_json()))
                if upstream != NEMOTRON_UPSTREAM_ROWS:
                    raise ValueError(f"The nemotron upstream holds {upstream} rows, expected {NEMOTRON_UPSTREAM_ROWS}")
            elif index == 3:
                for batch in pq.ParquetFile(local).iter_batches(batch_size=128):
                    for row in batch.to_pylist():
                        stamped_row, _ = ifbench_row(row)
                        counts[IFBENCH_LABEL] += 1
                        stamped.append((IFBENCH_LABEL, stamped_row["task_spec"]))
            elif index == 4:
                for batch in pq.ParquetFile(local).iter_batches(batch_size=128):
                    for row in batch.to_pylist():
                        stamped_row = openresearcher_row(row)
                        counts[OPENRESEARCHER_LABEL] += 1
                        stamped.append((OPENRESEARCHER_LABEL, stamped_row["task_spec"]))
            else:
                raw_math = pq.read_table(local).to_pylist()
                if len(raw_math) != HARDMATH_INPUT_ROWS:
                    raise ValueError("HARDMath base input population changed")
                converted = 0
                for row_index, data in enumerate(raw_math):
                    task = hardmath_task(data, row_index)
                    if isinstance(task, ImportRejection):
                        if task.kind != ImportFailureKind.SOURCE_DEFECT or task.reason != "invalid_reference":
                            raise ValueError(f"Unexpected HARDMath conversion failure: {task}")
                        hardmath_rejections.append({"source_row": row_index, **task.model_dump(mode="json")})
                        continue
                    converted += 1
                    counts[HARDMATH_LABEL] += 1
                    stamped.append((HARDMATH_LABEL, task.model_dump_json()))
                if converted != HARDMATH_ROWS or len(hardmath_rejections) != HARDMATH_REJECTIONS:
                    raise ValueError(
                        f"HARDMath converted {converted} rows with {len(hardmath_rejections)} rejections; "
                        f"expected {HARDMATH_ROWS} and {HARDMATH_REJECTIONS}"
                    )
    if dict(counts) != EXPECTED_COMPOSITION:
        raise ValueError(f"The pool composition differs: {dict(counts)} != {EXPECTED_COMPOSITION}")
    if len(nemotron_rejections) != NEMOTRON_REJECTED_ROWS:
        raise ValueError(f"nemotron_if rejected {len(nemotron_rejections)} rows, expected {NEMOTRON_REJECTED_ROWS}")
    pool_rows = {
        label: tuple(
            PoolRow(id=json.loads(raw)["id"], sha256=canonical_sha256(json.loads(raw)))
            for route, raw in stamped
            if route == label
        )
        for label in sorted(counts)
    }
    split = exp5_validation_split(pool_rows)
    curriculum = exp5_rl_curriculum(split.validation)
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
            raise ValueError(f"pool row {route!r}/{task['id']!r} landed on neither side of the holdout")
    files = {
        f"{label}/train.parquet": _write_pool_file(destination, f"{label}/train.parquet", rows)
        for label, rows in train_rows.items()
    }
    files["validation.parquet"] = _write_pool_file(destination, "validation.parquet", held)
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "curriculum": curriculum.identity,
                "validation": validation_payload(split.validation),
                "composition": dict(sorted(counts.items())),
                "train_composition": {label: len(rows) for label, rows in sorted(train_rows.items())},
                "routes": ROUTES,
                "files": files,
                "rows": sum(counts.values()),
                "training_ready": True,
                "provisioned": {
                    AGENTIC_TRAIN_SOURCE: {
                        "provisioned": "full",
                        "upstream_admitted_count": AGENTIC_ROWS,
                        "version": "swe-verified-20261112-v4 release (the admitted agentic training release)",
                    },
                    "PRIME-RL/Eurus-2-RL-Data": {
                        "provisioned": "subsample",
                        "upstream_admitted_count": None,
                        "version": "smartie-mixed-20261110-v1 adopted composition",
                    },
                    "nvidia/Nemotron-RL-Ultra-Training-Blends": {
                        "provisioned": "subsample",
                        "upstream_admitted_count": None,
                        "version": "smartie-mixed-20261110-v1 adopted composition",
                    },
                    "nvidia/OpenScience": {
                        "provisioned": "subsample",
                        "upstream_admitted_count": None,
                        "version": "smartie-mixed-20261110-v1 adopted composition",
                    },
                    GENERATED_LABEL: {
                        "provisioned": "subsample",
                        "upstream_admitted_count": GENERATED_V3_FULL_RUN_ROWS,
                        "version": (
                            "reasoning_gym-0.1.25 generated v3 at the adopted 20,800 cycle indices (the full "
                            "v3 run yields 104,000; the pools keep the adopted composition minus the recorded "
                            "grader-defective exclusion)"
                        ),
                    },
                    "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": {
                        "provisioned": "subsample",
                        "upstream_admitted_count": None,
                        "version": "smartie-mixed-20261110-v1 adopted composition",
                    },
                    "open-thoughts/TaskTrove": {
                        "provisioned": "subsample",
                        "upstream_admitted_count": None,
                        "version": "smartie-mixed-20261110-v1 adopted composition",
                    },
                    NEMOTRON_IF_LABEL: {
                        "provisioned": "full",
                        "upstream_admitted_count": NEMOTRON_UPSTREAM_ROWS,
                        "version": (
                            f"recipe v{NEMOTRON_IF_RECIPE_VERSION} over the pinned upstream file; the "
                            "conversion's filtration is part of the version"
                        ),
                    },
                    IFBENCH_LABEL: {
                        "provisioned": "full",
                        "upstream_admitted_count": IFBENCH_ROWS,
                        "version": f"{IFBENCH_REPO}@{IFBENCH_REVISION} retained tasks export",
                    },
                    OPENRESEARCHER_LABEL: {
                        "provisioned": "full",
                        "upstream_admitted_count": OPENRESEARCHER_ROWS,
                        "version": f"{OPENRESEARCHER_SERVICE_REPO}@{OPENRESEARCHER_SERVICE_REVISION} thin service pin",
                    },
                    HARDMATH_LABEL: {
                        "provisioned": "full",
                        "upstream_admitted_count": HARDMATH_INPUT_ROWS,
                        "version": (
                            f"{HARDMATH_REPO}@{HARDMATH_REVISION} base conversion; the recorded source defects "
                            "are part of the version"
                        ),
                    },
                },
                "grader_provenance": {
                    GENERATED_LABEL: generated_provenance,
                    "ifeval_family_embedded_graders": (
                        "known-stale: the adopted rows keep their embedded #11978/#11985/#11991-era IFEval "
                        "graders, which erred toward false negatives; a future full re-admission replaces them"
                    ),
                    NEMOTRON_IF_LABEL: (
                        f"recipe v{NEMOTRON_IF_RECIPE_VERSION}: verifyit's IFEval checks with each row's "
                        "non-null arguments"
                    ),
                },
                "openresearcher": {
                    "track": "thin-image + hosted retrieval service",
                    "dataset_id": OPENRESEARCHER_SERVICE_REPO,
                    "dataset_revision": OPENRESEARCHER_SERVICE_REVISION,
                    "parquet_sha256": OPENRESEARCHER_SERVICE_PARQUET_SHA256,
                    "rows": OPENRESEARCHER_ROWS,
                    "dropped_null_integrity_contracts": OPENRESEARCHER_ROWS,
                },
                "hardmath": {
                    "variant": "base",
                    "difficulty_filter": None,
                    "input_rows": HARDMATH_INPUT_ROWS,
                    "converted_rows": HARDMATH_ROWS,
                    "source_defects": len(hardmath_rejections),
                },
                "task_content_changes": (
                    "The adopted agentic and blend rows are byte-identical (the generated label regenerated "
                    "at v3 exactly as the Exp4 stamp); nemotron_if recipe v2; IFBench retained contract with "
                    "null Atlas fields dropped; the OpenResearcher thin service pin with null delivery "
                    "contracts dropped; HARDMath base conversion unfiltered"
                ),
            },
            indent=2,
        )
    )


def pool_steps(version: str, *, grader_environment: ArtifactStep) -> dict[str, ArtifactStep]:
    """The Exp5 pool artifact: eleven per-source holdout-excluding pools plus the holdout's validation data."""
    sources = [
        {"uri": prefix_join(AGENTIC_RELEASE_URL, AGENTIC_TRAIN_FILE), "sha256": AGENTIC_TRAIN_SHA256},
        {"uri": prefix_join(NONAGENTIC_RELEASE_URL, NONAGENTIC_TRAIN_FILE), "sha256": NONAGENTIC_TRAIN_SHA256},
        {"uri": NEMOTRON_JSONL_URL},
        {"uri": IFBENCH_URI, "sha256": IFBENCH_SHA256},
        {"uri": OPENRESEARCHER_SERVICE_URI, "sha256": OPENRESEARCHER_SERVICE_PARQUET_SHA256},
        {"uri": HARDMATH_URI, "sha256": HARDMATH_SHA256},
    ]
    deps = tuple(
        ArtifactStep.adopt(f"long-loop-exp5/admitted/{index}", version, source=member["uri"], config=member)
        for index, member in enumerate(sources)
    )
    step = apply(
        "long-loop-exp5/rl-pools",
        prepare_pool,
        version=version,
        output=OUT,
        agentic=deps[0],
        blend=deps[1],
        nemotron=deps[2],
        ifbench=deps[3],
        openresearcher=deps[4],
        hardmath=deps[5],
        grader_environment=grader_environment,
    )
    return {"pools": step}


@click.command(help=__doc__)
@build_options
def main() -> dict[str, ArtifactStep]:
    configure_coreweave_s3()
    os.environ["MARIN_PREFIX"] = OUTPUT_PREFIX
    return pool_steps(resolve_version("exp5-rl-data", None), grader_environment=environment_artifact(GRADER_PACKAGES))


if __name__ == "__main__":
    main()

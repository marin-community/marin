# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stamp the Exp4 pools: adopted releases, the regenerated Reasoning Gym label, and recipe-v2 nemotron_if.

The adopted smartie-mixed blend and the swe-verified agentic release are copied byte for byte; their
rows embed the graders they were admitted with. Two sources bind versions today's grader fixes created:
the generated Reasoning Gym label is regenerated with the current pipeline (generated version 3 — the
same prompts, the repaired verifyit scorers and controls, and the cross-day reproducibility comparison),
and nemotron_if is converted from the pinned upstream file with the verifyit-IFEval recipe (version 2).
The manifest records, per source, whether the pool carries the full admitted version or a subsample
together with the upstream admitted row count and the grader-version provenance.
"""

import base64
import hashlib
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import pyarrow as pa
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
from taskcompendium.pipeline.models import ImportRejection, RawRow

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_RELEASE_URL,
    AGENTIC_TRAIN_SHA256,
    AGENTIC_TRAIN_SOURCE,
    NONAGENTIC_RELEASE_URL,
    NONAGENTIC_TRAIN_SHA256,
    TEACHER_ROUTE_LABELS,
)
from experiments.post_training.task_curation.datasets.reasoning_gym.tasks import (
    EXCLUDED_GENERATORS,
    GENERATOR_ARCHIVE,
    GENERATOR_PARTS,
    GENERATOR_SHA256,
    GENERATOR_URL,
    GENERATOR_VERSION,
    PYTHON_HASH_SEED,
    convert_generated,
    generated_rows,
)
from experiments.post_training.task_curation.datasets.skyrl.ifeval import convert_nemotron_if
from experiments.post_training.task_curation.exp4_curriculum import (
    NEMOTRON_SOURCE,
    SOURCE_ROUTES,
    exp4_rl_curriculum,
    exp4_validation_split,
)
from experiments.post_training.task_curation.images.build import environment_artifact

NEMOTRON_JSONL_URL = (
    "hf://datasets/nvidia/Llama-Nemotron-Post-Training-Dataset"
    "@ab2a40d258a6a4d9d4c277d702aeea445081766c/RL/instruction_following/instruction_following.jsonl"
)
NEMOTRON_REVISION = "ab2a40d258a6a4d9d4c277d702aeea445081766c"
NEMOTRON_UPSTREAM_ROWS = 56_339
NEMOTRON_KEPT_ROWS = 56_083
NEMOTRON_REJECTED_ROWS = 256
NEMOTRON_IF_RECIPE_VERSION = "2"
NEMOTRON_IF_LABEL = "nemotron_if"
GENERATOR_RECIPE_NAME = "reasoning_gym_generated"
GENERATED_ROW_COUNT = 20_800
GENERATED_V3_FULL_RUN_ROWS = 104_000
EXCLUDED_GENERATED_FAMILIES = {"codeio": 191}
"""Adopted generated families left out of the rebuilt pools: codeio's entries are floating-point
dependent (the polynomial root's Newton path converges differently across platforms and libm
builds), so their fresh-dataset comparison fails outside the original build environment and their
grader cannot re-derive them. The census verified every other family reproduces exactly."""
GENERATED_POOL_ROWS = GENERATED_ROW_COUNT - sum(EXCLUDED_GENERATED_FAMILIES.values())
EXPECTED_COMPOSITION = {
    AGENTIC_TRAIN_SOURCE: 5608,
    "PRIME-RL/Eurus-2-RL-Data": 10_317,
    "nvidia/Nemotron-RL-Ultra-Training-Blends": 12_426,
    "nvidia/OpenScience": 20_790,
    "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": 11_634,
    "open-thoughts/TaskTrove": 24_033,
    NEMOTRON_IF_LABEL: NEMOTRON_KEPT_ROWS,
}
EXPECTED_COMPOSITION[TEACHER_ROUTE_LABELS[GENERATOR_URL]] = GENERATED_POOL_ROWS
FIVE_FAMILIES = ("complex_arithmetic", "coin_flip", "game_of_life_halting", "propositional_logic", "puzzle24")
FIVE_FAMILY_COUNTS = {
    "coin_flip": 215,
    "complex_arithmetic": 216,
    "game_of_life_halting": 206,
    "propositional_logic": 186,
    "puzzle24": 192,
}
TIME_INTERVALS_ROWS = 201
POOL_SCHEMA = pa.schema(
    [
        pa.field("task_spec", pa.string(), nullable=False),
        pa.field("teacher_route", pa.string(), nullable=False),
    ]
)


def _grader_lock(artifact_path: str) -> str:
    return prefix_join(artifact_path, "requirements.lock")


def row_identity(recipe_name: str, source: Source) -> str:
    return f"{recipe_name}-{canonical_sha256(source.model_dump())}"


def _row_family(task: dict) -> str | None:
    """The Reasoning Gym family a stamped row's verifier config records, when it carries one."""
    for resource in (task.get("resources") or {}).get("verifier") or []:
        if resource.get("path", "").endswith("config.json"):
            content = base64.b64decode((resource.get("source") or {}).get("content_base64", "")).decode(errors="replace")
            config = json.loads(content)
            return (config.get("contract") or {}).get("entry", {}).get("metadata", {}).get("source_dataset")
    return None


def _cycle_index(task: dict) -> int:
    return int(task["source"]["row"].rsplit(":", 1)[-1])


def _old_question(task: dict) -> str:
    for resource in (task.get("resources") or {}).get("verifier") or []:
        if resource.get("path", "").endswith("config.json"):
            content = base64.b64decode((resource.get("source") or {}).get("content_base64", "")).decode(errors="replace")
            config = json.loads(content)
            return config["contract"]["entry"]["question"]
    raise ValueError(f"Generated row {task.get('id')!r} carries no verifier config")


def write_pool_file(destination: StoragePath, name: str, rows: list[dict]) -> str:
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


def _regenerate_generated_label(old_rows: dict[int, dict], packages_lock: str) -> tuple[list[dict], dict]:
    """Regenerate the generated label with the current pipeline and guard it against the adopted rows.

    Returns the new stamped rows and the grader-provenance record: the per-family counts, the five
    fail-open families' rows, and the prompt-equality receipt over every regenerated row.
    """
    if get_job_info() is None:
        raise RuntimeError("Regenerate the Reasoning Gym label inside the regional Iris CPU job")
    wheel = StoragePath(GENERATOR_URL)
    rejections: list[dict] = []
    regenerated: dict[int, str] = {}

    def regenerate_part(part: int) -> list[tuple[int, str]]:
        """The part's full row set, regenerated in the original generation context.

        ``generate.py`` entries can depend on their process-local generation order, so a row only
        reproduces when its part generates exactly what the original run generated; the stamp takes
        the adopted rows from that output and applies the policy exclusion afterwards.
        """
        context = ConversionContext(
            inputs={},
            grader_environment=EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                packages_lock=packages_lock,
            ),
        )
        converted = []
        for index, data in generated_rows(
            wheel, GENERATOR_VERSION, EXCLUDED_GENERATORS, PYTHON_HASH_SEED, part, GENERATOR_PARTS, None
        ):
            if index not in old_rows:
                # The current cycle is larger than the adopted one; only the adopted rows are stamped.
                continue
            source = Source(
                dataset=GENERATOR_URL,
                revision=GENERATOR_SHA256,
                row=f"{GENERATOR_ARCHIVE}:{index}",
                importer_revision="1",
            )
            raw = RawRow(id=row_identity(GENERATOR_RECIPE_NAME, source), source=source, data=data)
            task = convert_generated(raw, context)
            if isinstance(task, ImportRejection):
                rejections.append({"index": index, **task.model_dump(mode="json")})
                continue
            converted.append((index, task.model_dump_json()))
        return converted

    with ThreadPoolExecutor(max_workers=8) as pool:
        for part_result in pool.map(regenerate_part, range(GENERATOR_PARTS)):
            for index, raw_task in part_result:
                regenerated[index] = raw_task
    excluded_families = tuple(EXCLUDED_GENERATED_FAMILIES)
    excluded_indices = {
        index for index, row in old_rows.items() if _row_family(json.loads(row["task_spec"])) in excluded_families
    }
    if len(excluded_indices) != sum(EXCLUDED_GENERATED_FAMILIES.values()):
        raise ValueError(
            f"The adopted label holds {len(excluded_indices)} rows in the excluded families "
            f"{excluded_families}; expected {sum(EXCLUDED_GENERATED_FAMILIES.values())}"
        )
    unexpected = [rejection for rejection in rejections if rejection["index"] not in excluded_indices]
    if unexpected:
        raise ValueError(f"The regeneration rejected rows outside the excluded families: {unexpected[:3]}")
    for index in excluded_indices:
        regenerated.pop(index, None)
    if len(regenerated) != GENERATED_POOL_ROWS:
        raise ValueError(
            f"The regenerated label holds {len(regenerated)} rows; expected {GENERATED_POOL_ROWS} "
            f"({GENERATED_ROW_COUNT} adopted minus the recorded exclusions)"
        )
    for index, old in sorted(old_rows.items()):
        if index in excluded_indices:
            continue
        new = json.loads(regenerated[index])
        old_task = json.loads(old["task_spec"])
        if new["context"]["events"][0]["content"] != _old_question(old_task):
            raise ValueError(f"Regenerated row {index} changed the prompt")
        if new["id"] != old_task["id"]:
            raise ValueError(f"Regenerated row {index} changed the task identity")
    tasks = [json.loads(raw) for raw in regenerated.values()]
    counts = Counter(_row_family(task) for task in tasks)
    five_family = {family: counts[family] for family in FIVE_FAMILIES}
    if five_family != FIVE_FAMILY_COUNTS:
        raise ValueError(f"The five repaired families' counts changed: {five_family} != {FIVE_FAMILY_COUNTS}")
    if counts["time_intervals"] != TIME_INTERVALS_ROWS:
        raise ValueError(f"time_intervals rows changed: {counts['time_intervals']} != {TIME_INTERVALS_ROWS}")
    five_ids = sorted(task["id"] for task in tasks if _row_family(task) in FIVE_FAMILIES)
    provenance = {
        "generator_version": GENERATOR_VERSION,
        "generated_version": "3",
        "row_count": len(regenerated),
        "regenerated_indices": (
            "the adopted label's exact cycle indices (the full v3 run yields "
            f"{GENERATED_V3_FULL_RUN_ROWS}; the pools keep the adopted composition minus the "
            "excluded families)"
        ),
        "prompts_unchanged": True,
        "excluded_families": [
            {
                "family": family,
                "rows": count,
                "note": (
                    "codeio's entries are floating-point dependent: the polynomial root's Newton path "
                    "converges differently across platforms and libm builds, so the fresh-dataset "
                    "comparison fails outside the original build environment. The full prompt census "
                    "verified every other family reproduces exactly."
                ),
            }
            for family, count in sorted(EXCLUDED_GENERATED_FAMILIES.items())
        ],
        "shared_grade_script": (
            "every row's embedded reasoning_gym_grade.py is regenerated: the cross-day entry comparison "
            "(#11951) and the repaired verifyit scorers (#11987)"
        ),
        "five_family_rows": five_family,
        "five_family_ids": five_ids,
        "time_intervals_rows": counts["time_intervals"],
    }
    rows = [
        {"task_spec": regenerated[index], "teacher_route": TEACHER_ROUTE_LABELS[GENERATOR_URL]}
        for index in sorted(regenerated)
    ]
    return rows, provenance


def prepare_pool(*, output: str, agentic: str, blend: str, nemotron: str, grader_environment: str) -> None:
    """Stamp the pools: adopted rows byte for byte, the regenerated label, and recipe-v2 nemotron_if."""
    if get_job_info() is None:
        raise RuntimeError("Prepare Exp4 pools inside the regional Iris CPU job")
    packages_lock = _grader_lock(grader_environment)
    counts: Counter = Counter()
    stamped: list[tuple[str, str]] = []
    nemotron_rejections: list[dict] = []
    generated_old: dict[int, dict] = {}
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for index, (uri, digest) in enumerate(
            (
                (agentic, AGENTIC_TRAIN_SHA256),
                (blend, NONAGENTIC_TRAIN_SHA256),
                (nemotron, None),
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
                    raise ValueError(f"Input {index} differs from its admitted digest: {observed}")
            if index == 0:
                for batch in pq.ParquetFile(local).iter_batches(batch_size=128):
                    for row in batch.to_pylist():
                        task = json.loads(row["task_spec"])
                        TaskSpec.model_validate_json(row["task_spec"])
                        route = TEACHER_ROUTE_LABELS[task["source"]["dataset"]]
                        counts[route] += 1
                        stamped.append((route, row["task_spec"]))
            elif index == 1:
                for batch in pq.ParquetFile(local).iter_batches(batch_size=128):
                    for row in batch.to_pylist():
                        task = json.loads(row["task_spec"])
                        TaskSpec.model_validate_json(row["task_spec"])
                        route = TEACHER_ROUTE_LABELS[task["source"]["dataset"]]
                        if route == TEACHER_ROUTE_LABELS[GENERATOR_URL]:
                            generated_old[_cycle_index(task)] = row
                            continue
                        counts[route] += 1
                        stamped.append((route, row["task_spec"]))
                rows, generated_provenance = _regenerate_generated_label(generated_old, packages_lock)
                for row in rows:
                    counts[row["teacher_route"]] += 1
                    stamped.append((row["teacher_route"], row["task_spec"]))
            else:
                context = ConversionContext(
                    inputs={},
                    grader_environment=EnvironmentRequirements(
                        command_semantics=CommandSemantics.LINUX_PROCESS,
                        packages_lock=packages_lock,
                    ),
                )
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
                    raw = RawRow(
                        id=row_identity("nemotron_if", source),
                        source=source,
                        data=json.loads(raw_line),
                    )
                    task = convert_nemotron_if(raw, context)
                    if isinstance(task, ImportRejection):
                        nemotron_rejections.append({"row": upstream - 1, **task.model_dump(mode="json")})
                        continue
                    counts[NEMOTRON_IF_LABEL] += 1
                    stamped.append((NEMOTRON_IF_LABEL, task.model_dump_json()))
                if upstream != NEMOTRON_UPSTREAM_ROWS:
                    raise ValueError(
                        f"The nemotron upstream file holds {upstream} rows, expected {NEMOTRON_UPSTREAM_ROWS}"
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
            raise ValueError(f"pool row {route!r}/{task['id']!r} landed on neither side of the holdout")
    files = {
        f"{label}/train.parquet": write_pool_file(destination, f"{label}/train.parquet", rows)
        for label, rows in train_rows.items()
    }
    files["validation.parquet"] = write_pool_file(destination, "validation.parquet", held)
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "curriculum": curriculum.identity,
                "validation": validation_payload(split.validation),
                "composition": dict(sorted(counts.items())),
                "train_composition": {label: len(rows) for label, rows in sorted(train_rows.items())},
                "routes": dict(sorted(SOURCE_ROUTES.items())),
                "files": files,
                "rows": sum(counts.values()),
                "provisioned": {
                    AGENTIC_TRAIN_SOURCE: {
                        "provisioned": "full",
                        "upstream_admitted_count": 5608,
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
                    TEACHER_ROUTE_LABELS[GENERATOR_URL]: {
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
                },
                "grader_provenance": {
                    TEACHER_ROUTE_LABELS[GENERATOR_URL]: generated_provenance,
                    "ifeval_family_embedded_graders": (
                        "known-stale: the adopted rows keep their embedded #11978/#11985/#11991-era IFEval "
                        "graders, which erred toward false negatives; a future full re-admission replaces them"
                    ),
                    NEMOTRON_IF_LABEL: (
                        f"recipe v{NEMOTRON_IF_RECIPE_VERSION}: verifyit's IFEval checks with each row's "
                        "non-null arguments"
                    ),
                },
                "nemotron_rejections": nemotron_rejections,
                "task_content_changes": (
                    "The adopted agentic and blend rows are byte-identical; the generated label is "
                    "regenerated at v3 (same prompts, repaired scorers and controls) and nemotron_if is "
                    "re-converted at recipe v2"
                ),
            },
            indent=2,
        )
    )


def pool_steps(version: str, *, grader_environment: ArtifactStep) -> dict[str, ArtifactStep]:
    """The Exp4 pool artifact: per-source holdout-excluding pools plus the holdout's validation data."""
    sources = [
        {
            "uri": prefix_join(AGENTIC_RELEASE_URL, "agentic-train.parquet"),
            "sha256": AGENTIC_TRAIN_SHA256,
            "rows": 5608,
        },
        {
            "uri": prefix_join(NONAGENTIC_RELEASE_URL, "nonagentic-train.parquet"),
            "sha256": NONAGENTIC_TRAIN_SHA256,
            "rows": 100_000,
        },
        {"uri": NEMOTRON_JSONL_URL, "rows": NEMOTRON_UPSTREAM_ROWS},
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
        agentic=deps[0],
        blend=deps[1],
        nemotron=deps[2],
        grader_environment=grader_environment,
    )
    return {"pools": step}


@click.command(help=__doc__)
@build_options
def main() -> dict[str, ArtifactStep]:
    configure_coreweave_s3()
    os.environ["MARIN_PREFIX"] = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp4"
    return pool_steps(resolve_version("exp4-rl-data", None), grader_environment=environment_artifact(GRADER_PACKAGES))


if __name__ == "__main__":
    main()

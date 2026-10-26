# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select a training inventory from admitted curation artifacts."""

import json
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from pydantic import BaseModel, ConfigDict, PositiveInt
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.structure_closure import structure_closure_task
from taskcompendium.models import (
    SCHEMA_VERSION,
    EnvironmentRequirements,
    ScriptGrader,
    Source,
    TaskSpec,
    VerifyitGrader,
    grades_with_judge,
)
from taskcompendium.pipeline.fingerprints import deduplication_key, semantic_digest
from taskcompendium.pipeline.models import IntendedUse, NormalizedTask, RawRow
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample
from taskcompendium.runtime.resources import resource_bytes
from verifyit.json_objects import unique_object
from zephyr.writers import write_parquet_file

TRAIN_FILENAME = "train.parquet"
VALIDATION_FILENAME = "validation.parquet"
FULL_VALIDATION_FILENAME = "full_validation.parquet"
TASK_MANIFEST_FILENAME = "task_manifest.parquet"
TASK_SPEC_COLUMN = "task_spec"
TASK_JSON_COLUMN = "task_json"
TASK_SCHEMA = pa.schema([pa.field(TASK_SPEC_COLUMN, pa.string(), nullable=False)])
HELDOUT_SCHEMA_VERSION = "0.27"
STRUCTURE_CLOSURE_DATASET = "structure-closure"
HELDOUT_FAMILIES = MappingProxyType(
    {
        "ultra_sft_step3200_calendar_v2": "calendar",
        "ultra_sft_step3200_comp_coding": "competitive_coding",
        "ultra_sft_step3200_reasoning_gym": "reasoning_gym",
        "ultra_sft_step3200_stem_mcqa": "stem",
    }
)
AAII_CATEGORIES = MappingProxyType(
    {
        "calendar": "format",
        "competitive_coding": "coding",
        "reasoning_gym": "scientific_reasoning",
        "stem": "scientific_reasoning",
        "structural": "format",
    }
)
TASK_MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("family", pa.string(), nullable=False),
        pa.field("aaii_category", pa.string(), nullable=False),
        pa.field("agentic", pa.bool_(), nullable=False),
        pa.field("grader_kind", pa.string(), nullable=False),
        pa.field("upstream_pass_rate", pa.float64()),
    ]
)


class HeldoutSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str
    version: str
    rows: PositiveInt


class SelectionRecipe(BaseModel):
    """An explicit source allowlist, family quotas and fixed evaluation input."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    sources: dict[str, str]
    family_limits: dict[str, PositiveInt]
    source_limits: dict[str, PositiveInt] | None = None
    heldout: HeldoutSource
    validation_per_family: PositiveInt
    seed: int


@dataclass(frozen=True)
class SelectionInput:
    source_id: str
    family: str
    root: str


@dataclass(frozen=True)
class SelectionConfig:
    sources: tuple[SelectionInput, ...]
    recipe: SelectionRecipe
    heldout: str
    output_path: str


@dataclass(frozen=True)
class HeldoutSelection:
    full: tuple[TaskSpec, ...]
    sample: tuple[TaskSpec, ...]
    family_counts: dict[str, int]


@dataclass(frozen=True)
class Candidate:
    task_id: str
    task_json: str
    source_id: str
    # Semantic deduplication key, including private grader inputs for opaque contracts.
    contract: str


@dataclass
class SourceCounts:
    input: int = 0
    heldout_overlap: int = 0
    duplicate: int = 0
    eligible: int = 0
    selected: int = 0


def _grader_contract(task: TaskSpec) -> dict[str, Any]:
    config = next(resource for resource in task.resources.verifier if resource.path == "config.json")
    return json.loads(resource_bytes(config), object_pairs_hook=unique_object)["contract"]


def validation_task(row: dict[str, Any], grader_environments: Mapping[str, EnvironmentRequirements]) -> TaskSpec:
    """Import a pinned original Snowball development task into the current schema."""
    raw = row["lowered_task_spec"]
    document = json.loads(raw, object_pairs_hook=unique_object)
    original = document["task"]
    if original["schema_version"] != HELDOUT_SCHEMA_VERSION:
        raise ValueError("The pinned heldout TaskSpec version changed")
    if original["source"]["dataset"] == STRUCTURE_CLOSURE_DATASET:
        converted = structure_closure_task(
            RawRow(original["id"], Source.model_validate(original["source"]), {"lowered_task_spec": raw})
        )
        if not isinstance(converted, NormalizedTask):
            raise ValueError(f"Native structure-closure import failed for {original['id']}: {converted}")
        return converted.task
    task = TaskSpec.model_validate({**original, "schema_version": SCHEMA_VERSION})
    if document["runtime"]["task_machine"] is not None:
        raise ValueError("Heldout source machine contract changed")
    if document["session"]["task_session"] != "shellbox" or document["session"]["max_turns"] != 1:
        raise ValueError("The heldout source must use the one-turn Shellbox contract")
    if isinstance(task.grader, ScriptGrader) and task.grader.environment.docker_image is not None:
        dataset = _grader_contract(task)["dataset"]
        component = f"MarinSkyRL:nemotron_ultra_rlvr1/{dataset}"
        if component not in grader_environments:
            raise ValueError(f"No admitted grader environment for {component}")
        task = task.model_copy(
            update={"grader": task.grader.model_copy(update={"environment": grader_environments[component]})}
        )
    return task


def validation_family(task: TaskSpec) -> str:
    if task.source.dataset == STRUCTURE_CLOSURE_DATASET:
        return "structural"
    return HELDOUT_FAMILIES[_grader_contract(task)["dataset"]]


def select_heldout(
    rows: list[dict[str, Any]], recipe: SelectionRecipe, grader_environments: Mapping[str, EnvironmentRequirements]
) -> HeldoutSelection:
    """Retain the full exclusion set and choose a seeded panel within each family."""
    full = tuple(validation_task(row, grader_environments) for row in rows)
    if len({task.id for task in full}) != len(full):
        raise ValueError("Heldout task IDs are not unique")
    families = {task.id: validation_family(task) for task in full}
    selected: set[str] = set()
    for family in sorted(set(families.values())):
        count, sample = seeded_sample(
            (task for task in full if families[task.id] == family),
            size=recipe.validation_per_family,
            key=lambda task: seeded_order(task.id, recipe.seed),
        )
        if count < recipe.validation_per_family:
            raise ValueError(f"Heldout source lacks enough {family} tasks")
        selected.update(task.id for task in sample)
    return HeldoutSelection(
        full=full,
        sample=tuple(task for task in full if task.id in selected),
        family_counts=dict(Counter(families[task.id] for task in full if task.id in selected)),
    )


def task_records(tasks: Iterator[TaskSpec]) -> Iterator[dict[str, str]]:
    for task in tasks:
        yield {TASK_SPEC_COLUMN: task.model_dump_json()}


def _admitted_records(source: SelectionInput) -> Iterator[dict[str, str]]:
    root = StoragePath(source.root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["mode"] != "full":
        raise ValueError(f"Selection requires FULL curation: {source.source_id}")
    if manifest["intended_use"] != IntendedUse.TRAIN:
        raise ValueError(f"Selection source is not a training population: {source.source_id}")
    for path in sorted((root / "final" / "*.parquet").glob(), key=str):
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=256, columns=[TASK_JSON_COLUMN]):
                yield from batch.to_pylist()


def _grader_environments(sources: tuple[SelectionInput, ...]) -> dict[str, EnvironmentRequirements]:
    components = {f"MarinSkyRL:nemotron_ultra_rlvr1/{dataset}" for dataset in HELDOUT_FAMILIES}
    environments = {}
    for source in sources:
        if source.source_id not in components:
            continue
        record = next(_admitted_records(source), None)
        if record is None:
            raise ValueError(f"No admitted grader environment for {source.source_id}")
        task = TaskSpec.model_validate_json(record[TASK_JSON_COLUMN])
        if not isinstance(task.grader, ScriptGrader) or task.grader.environment.docker_image is not None:
            raise ValueError(f"Admitted component needs a local script grader: {source.source_id}")
        environments[source.source_id] = task.grader.environment
    return environments


def _grader_kind(task: TaskSpec) -> str:
    if grades_with_judge(task.grader):
        return "remote_judge"
    if not isinstance(task.grader, ScriptGrader | VerifyitGrader):
        raise ValueError(f"Task has no runnable grader: {task.id}")
    environment = task.grader.environment
    if environment is None:
        return "in_process"
    return "image" if environment.docker_image is not None else "package"


def _manifest_records(tasks: tuple[TaskSpec, ...]) -> Iterator[dict[str, Any]]:
    for task in tasks:
        family = validation_family(task)
        passed = None
        if task.source.dataset != STRUCTURE_CLOSURE_DATASET:
            passed = _grader_contract(task).get("pass_rate_passed")
        yield {
            "task_id": task.id,
            "family": family,
            "aaii_category": AAII_CATEGORIES[family],
            "agentic": "agentic" in task.tags,
            "grader_kind": _grader_kind(task),
            "upstream_pass_rate": None if passed is None else passed / 8,
        }


def _grader_counts(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=[TASK_SPEC_COLUMN]):
        counts.update(_grader_kind(TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN])) for row in batch.to_pylist())
    return {kind: counts[kind] for kind in ("package", "image", "remote_judge", "in_process")}


def write_training(config: SelectionConfig, full: tuple[TaskSpec, ...], destination: Path) -> dict[str, SourceCounts]:
    """Filter admitted tasks before seeded sampling without replacement."""
    heldout_ids = {task.id for task in full}
    heldout_public = {semantic_digest(task, False) for task in full}
    heldout_contexts = {task.context.model_dump_json() for task in full}
    selected_ids: set[str] = set()
    selected_contracts: set[str] = set()
    counts = {source.source_id: SourceCounts() for source in config.sources}

    def candidates(sources: tuple[SelectionInput, ...]) -> Iterator[Candidate]:
        seen_ids: set[str] = set()
        seen_contracts: set[str] = set()
        for source in sources:
            for row in _admitted_records(source):
                task = TaskSpec.model_validate_json(row[TASK_JSON_COLUMN])
                source_counts = counts[source.source_id]
                source_counts.input += 1
                if (
                    task.id in heldout_ids
                    or semantic_digest(task, False) in heldout_public
                    or task.context.model_dump_json() in heldout_contexts
                ):
                    source_counts.heldout_overlap += 1
                    continue
                contract = deduplication_key(task)
                if (
                    task.id in selected_ids
                    or contract in selected_contracts
                    or task.id in seen_ids
                    or contract in seen_contracts
                ):
                    source_counts.duplicate += 1
                    continue
                seen_ids.add(task.id)
                seen_contracts.add(contract)
                source_counts.eligible += 1
                yield Candidate(task.id, row[TASK_JSON_COLUMN], source.source_id, contract)

    def selected_records() -> Iterator[dict[str, str]]:
        groups = (
            [(source.source_id, config.recipe.source_limits[source.source_id], (source,)) for source in config.sources]
            if config.recipe.source_limits is not None
            else [
                (family, limit, tuple(source for source in config.sources if source.family == family))
                for family, limit in sorted(config.recipe.family_limits.items())
            ]
        )
        for group, limit, sources in groups:
            count, sample = seeded_sample(
                candidates(sources), size=limit, key=lambda row: seeded_order(row.task_id, config.recipe.seed)
            )
            if count < limit:
                raise ValueError(f"Selection shortfall for {group}: requested {limit}, eligible {count}")
            for row in sample:
                selected_ids.add(row.task_id)
                selected_contracts.add(row.contract)
                counts[row.source_id].selected += 1
                yield {TASK_SPEC_COLUMN: row.task_json}

    write_parquet_file(selected_records(), str(destination), schema=TASK_SCHEMA)
    return counts


def prepare(config: SelectionConfig) -> None:
    """Publish selected training TaskSpecs and the fixed and complete heldout panels."""
    with StoragePath(config.heldout).open("rb") as source:
        heldout = pq.read_table(source).to_pylist()
    if len(heldout) != config.recipe.heldout.rows:
        raise ValueError("Heldout source row count changed")
    selection = select_heldout(heldout, config.recipe, _grader_environments(config.sources))
    with TemporaryDirectory(prefix="task-selection-") as temporary:
        directory = Path(temporary)
        counts = write_training(config, selection.full, directory / TRAIN_FILENAME)
        write_parquet_file(
            task_records(iter(selection.sample)), str(directory / VALIDATION_FILENAME), schema=TASK_SCHEMA
        )
        write_parquet_file(
            task_records(iter(selection.full)), str(directory / FULL_VALIDATION_FILENAME), schema=TASK_SCHEMA
        )
        write_parquet_file(
            _manifest_records(selection.full), str(directory / TASK_MANIFEST_FILENAME), schema=TASK_MANIFEST_SCHEMA
        )
        summary = {
            "recipe": config.recipe.model_dump(mode="json"),
            "sources": [asdict(source) for source in config.sources],
            "source_counts": {source: asdict(count) for source, count in counts.items()},
            "training_rows": sum(count.selected for count in counts.values()),
            "training_family_counts": {
                family: sum(counts[source.source_id].selected for source in config.sources if source.family == family)
                for family in sorted(config.recipe.family_limits)
            },
            "validation_rows": len(selection.sample),
            "full_validation_rows": len(selection.full),
            "validation_family_counts": selection.family_counts,
            "validation_task_ids": [task.id for task in selection.sample],
            "grader_kind_counts": {
                name.removesuffix(".parquet"): _grader_counts(directory / name)
                for name in (TRAIN_FILENAME, VALIDATION_FILENAME, FULL_VALIDATION_FILENAME)
            },
        }
        destination = StoragePath(config.output_path)
        destination.mkdirs()
        for name in (TRAIN_FILENAME, VALIDATION_FILENAME, FULL_VALIDATION_FILENAME, TASK_MANIFEST_FILENAME):
            (destination / name).upload_from(str(directory / name))
        (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def task_selection_step(curated: dict[str, ArtifactStep], recipe: SelectionRecipe) -> ArtifactStep[Artifact]:
    """Bind a terminal selection to admitted source artifacts and a pinned heldout input."""
    missing = recipe.sources.keys() - curated.keys()
    if missing:
        raise ValueError(f"Selection requires curation sources: {', '.join(sorted(missing))}")
    if not recipe.sources or set(recipe.sources.values()) != recipe.family_limits.keys():
        raise ValueError("Selection sources must cover exactly the families with quotas")
    if recipe.source_limits is not None:
        if recipe.source_limits.keys() != recipe.sources.keys():
            raise ValueError("Source quotas must cover exactly the selection sources")
        family_totals: Counter[str] = Counter()
        for source, limit in recipe.source_limits.items():
            family_totals[recipe.sources[source]] += limit
        if family_totals != recipe.family_limits:
            raise ValueError("Source quotas must sum to each family quota")
    heldout = ArtifactStep.adopt(
        f"rl-data/selection-inputs/{recipe.name}-heldout", recipe.heldout.version, recipe.heldout.uri
    )
    inputs = {source_id: curated[source_id] for source_id in sorted(recipe.sources)}

    def build_config(ctx: StepContext) -> SelectionConfig:
        return SelectionConfig(
            tuple(
                SelectionInput(source_id, recipe.sources[source_id], ctx.artifact_path(step))
                for source_id, step in inputs.items()
            ),
            recipe,
            ctx.artifact_path(heldout),
            ctx.output_path,
        )

    return ArtifactStep(
        name=f"rl-data/task-selection/{recipe.name}",
        version=recipe.version,
        artifact_type=Artifact,
        deps=(*inputs.values(), heldout),
        run=prepare,
        build_config=build_config,
    )

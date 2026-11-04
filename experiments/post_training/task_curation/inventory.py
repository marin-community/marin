# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select a training inventory from admitted curation artifacts."""

import json
import math
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
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
from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.structure_closure import structure_closure_task
from taskcompendium.evaluation.corpus import EvaluationCase, evaluation_cases
from taskcompendium.evaluation.spec import ChatGeneration, TextGeneration
from taskcompendium.grader import grader_config
from taskcompendium.importers.evalchemy.assets.code_protocol import mbpp_example
from taskcompendium.models import (
    SCHEMA_VERSION,
    ConversationInput,
    EnvironmentRequirements,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    grades_in_process,
    grades_with_judge,
)
from taskcompendium.pipeline.fingerprints import deduplication_key, semantic_digest
from taskcompendium.pipeline.models import IntendedUse, NormalizedTask, RawRow
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample
from verifyit.json_objects import unique_object
from zephyr.writers import write_parquet_file

from experiments.post_training.task_curation.datasets.nemotron_ultra.components import BLENDS

SELECTION_PREFIX = "rl-data/task-selection"
HELDOUT_BLEND = "rlvr1"
STRUCTURAL_FAMILY = "structural"
PARQUET_BATCH_ROWS = 256
TRAIN_FILENAME = "train.parquet"
VALIDATION_FILENAME = "validation.parquet"
SUMMARY_FILENAME = "summary.json"
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
        STRUCTURAL_FAMILY: "format",
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


class GraderKind(StrEnum):
    PACKAGE = "package"
    IMAGE = "image"
    REMOTE_JUDGE = "remote_judge"
    IN_PROCESS = "in_process"


class PanelFormat(StrEnum):
    HISTORICAL = "historical"
    TASK_SPEC = "task_spec"
    NORMALIZED = "normalized"
    EVALUATION = "evaluation"


class HeldoutSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str
    version: str
    rows: PositiveInt
    format: PanelFormat


class SelectionRecipe(BaseModel):
    """An explicit source allowlist, family quotas and fixed evaluation input."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    sources: dict[str, str]
    family_limits: dict[str, PositiveInt]
    source_limits: dict[str, PositiveInt] | None = None
    heldout: tuple[HeldoutSource, ...]
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
    heldout: tuple[HeldoutSource, ...]
    output_path: str


@dataclass(frozen=True)
class HeldoutSelection:
    full: tuple[TaskSpec, ...]
    sample: tuple[TaskSpec, ...]
    family_counts: dict[str, int]


class PanelSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input: HeldoutSource
    family: str
    aaii_category: str
    license: str
    count: PositiveInt


class PanelMeasurement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str
    version: str
    rows: PositiveInt
    model_revision: str
    run_output: str


class TestPanelRecipe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    sources: tuple[PanelSource, ...]
    training: HeldoutSource
    development: HeldoutSource
    fast_counts: dict[str, PositiveInt]
    seed: int
    pass_rates: PanelMeasurement | None = None
    provisional: HeldoutSource | None = None


@dataclass(frozen=True)
class TestPanelConfig:
    recipe: TestPanelRecipe
    output_path: str


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
    return grader_config(task)["contract"]


def validation_task(row: dict[str, Any], grader_environments: Mapping[str, EnvironmentRequirements]) -> TaskSpec:
    """Import a historical task; an image-based script grader takes its admitted component's environment."""
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
        if dataset not in grader_environments:
            raise ValueError(f"No admitted grader environment for {BLENDS[HELDOUT_BLEND][dataset].id}")
        task = task.model_copy(
            update={"grader": task.grader.model_copy(update={"environment": grader_environments[dataset]})}
        )
    return task


def validation_family(task: TaskSpec) -> str:
    if task.source.dataset == STRUCTURE_CLOSURE_DATASET:
        return STRUCTURAL_FAMILY
    return HELDOUT_FAMILIES[_grader_contract(task)["dataset"]]


def select_heldout(full: tuple[TaskSpec, ...], recipe: SelectionRecipe) -> HeldoutSelection:
    """Retain the full exclusion set and choose a seeded panel within each family."""
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


def _panel_source_tasks(
    source: HeldoutSource, grader_environments: Mapping[str, EnvironmentRequirements]
) -> Iterator[TaskSpec]:
    count = 0
    if source.format == PanelFormat.EVALUATION:
        for case in evaluation_cases(source.uri):
            count += 1
            yield _chat_panel_task(case)
    else:
        root = StoragePath(source.uri)
        paths = [root]
        column = TASK_SPEC_COLUMN
        if source.format == PanelFormat.HISTORICAL:
            column = "lowered_task_spec"
        elif source.format == PanelFormat.NORMALIZED:
            manifest = json.loads((root / "manifest.json").read_text())
            if manifest["intended_use"] != IntendedUse.EVAL:
                raise ValueError(f"Panel source is not reserved for evaluation: {source.uri}")
            paths = sorted((root / "normalize/*.parquet").glob(), key=str)
            column = TASK_JSON_COLUMN
        for path in paths:
            with path.open("rb") as stream:
                for batch in pq.ParquetFile(stream).iter_batches(batch_size=PARQUET_BATCH_ROWS, columns=[column]):
                    for row in batch.to_pylist():
                        if source.format == PanelFormat.NORMALIZED and row[column] is None:
                            continue
                        count += 1
                        yield (
                            validation_task(row, grader_environments)
                            if source.format == PanelFormat.HISTORICAL
                            else TaskSpec.model_validate_json(row[column])
                        )
    if count != source.rows:
        raise ValueError(f"Panel source row count changed: {source.uri}: expected {source.rows}, read {count}")


def _chat_panel_task(case: EvaluationCase) -> TaskSpec:
    task, operation = case.task, case.item.operation
    if not isinstance(operation, ChatGeneration | TextGeneration):
        raise ValueError(f"Test panels require generated answers: {task.id}")
    if isinstance(operation, ChatGeneration) and operation.variants:
        zero_shot = [variant.context for variant in operation.variants if variant.fewshot == 0]
        if len(zero_shot) != 1:
            raise ValueError(f"Test panel requires one zero-shot prompt: {task.id}")
        task = task.model_copy(update={"context": zero_shot[0]})
    if "benchmark:mbppplus" in task.tags:
        problem = source_evidence(task)
        prompt = "Write the Python function described below. Return the code in a Python code block.\n\n" + mbpp_example(
            problem["prompt"], problem["test_list"]
        )
        task = task.model_copy(update={"context": ConversationInput(events=(TextMessage(role="user", content=prompt),))})
    return task


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
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=PARQUET_BATCH_ROWS, columns=[TASK_JSON_COLUMN]):
                yield from batch.to_pylist()


def _grader_environments(sources: tuple[SelectionInput, ...]) -> dict[str, EnvironmentRequirements]:
    components = {BLENDS[HELDOUT_BLEND][dataset].id: dataset for dataset in HELDOUT_FAMILIES}
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
        environments[components[source.source_id]] = task.grader.environment
    return environments


def _grader_kind(task: TaskSpec) -> GraderKind:
    if grades_with_judge(task.grader):
        return GraderKind.REMOTE_JUDGE
    if grades_in_process(task.grader):
        return GraderKind.IN_PROCESS
    if not isinstance(task.grader, ScriptGrader | VerifyitGrader) or task.grader.environment is None:
        raise ValueError(f"Task has no runnable grader: {task.id}")
    environment = task.grader.environment
    return GraderKind.IMAGE if environment.docker_image is not None else GraderKind.PACKAGE


def _manifest_records(tasks: tuple[TaskSpec, ...]) -> Iterator[dict[str, Any]]:
    for task in tasks:
        family = validation_family(task)
        contract = None if family == STRUCTURAL_FAMILY else _grader_contract(task)
        yield {
            "task_id": task.id,
            "family": family,
            "aaii_category": AAII_CATEGORIES[family],
            "agentic": "agentic" in task.tags,
            "grader_kind": _grader_kind(task),
            "upstream_pass_rate": None if contract is None else contract.get("pass_rate"),
        }


def _grader_counts(path: Path) -> dict[str, int]:
    counts: Counter[GraderKind] = Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=PARQUET_BATCH_ROWS, columns=[TASK_SPEC_COLUMN]):
        counts.update(_grader_kind(TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN])) for row in batch.to_pylist())
    return {kind.value: counts[kind] for kind in GraderKind}


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
    environments = _grader_environments(config.sources)
    heldout = tuple(task for source in config.heldout for task in _panel_source_tasks(source, environments))
    selection = select_heldout(heldout, config.recipe)
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
        (destination / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


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
    heldout = tuple(
        ArtifactStep.adopt(f"rl-data/selection-inputs/{recipe.name}-heldout-{index}", source.version, source.uri)
        for index, source in enumerate(recipe.heldout)
    )
    inputs = {source_id: curated[source_id] for source_id in sorted(recipe.sources)}

    def build_config(ctx: StepContext) -> SelectionConfig:
        return SelectionConfig(
            tuple(
                SelectionInput(source_id, recipe.sources[source_id], ctx.artifact_path(step))
                for source_id, step in inputs.items()
            ),
            recipe,
            tuple(
                source.model_copy(update={"uri": ctx.artifact_path(step)})
                for source, step in zip(recipe.heldout, heldout, strict=True)
            ),
            ctx.output_path,
        )

    return ArtifactStep(
        name=f"{SELECTION_PREFIX}/{recipe.name}",
        version=recipe.version,
        artifact_type=Artifact,
        deps=(*inputs.values(), *heldout),
        run=prepare,
        build_config=build_config,
    )


def _panel_measurements(measurement: PanelMeasurement | None) -> dict[str, dict[str, Any]]:
    if measurement is None:
        return {}
    with StoragePath(measurement.uri).open("rb") as stream:
        rows = pq.read_table(stream).to_pylist()
    by_id = {row["task_id"]: row for row in rows}
    if len(by_id) != len(rows) or len(rows) != measurement.rows:
        raise ValueError("Panel pass-rate rows must have unique task IDs and the declared population")
    for row in rows:
        if (
            row["attempted"] != 8
            or not 0 < row["samples"] <= row["attempted"]
            or not 0 <= row["passed"] <= row["samples"]
            or not math.isfinite(row["mean_score"])
            or not 0 <= row["mean_score"] <= 1
            or row["model_revision"] != measurement.model_revision
        ):
            raise ValueError(f"Panel pass-rate evidence is incomplete: {row['task_id']}")
    return by_id


def prepare_test_panel(config: TestPanelConfig) -> None:
    """Publish seeded locally graded evaluation tasks after checking training and development exclusions."""
    recipe = config.recipe
    excluded_ids, excluded_digests, excluded_contexts = set(), set(), set()
    for source in (recipe.training, recipe.development):
        for task in _panel_source_tasks(source, {}):
            excluded_ids.add(task.id)
            excluded_digests.add(semantic_digest(task, False))
            excluded_contexts.add(task.context.model_dump_json())
    selected: list[tuple[TaskSpec, PanelSource]] = []
    selected_ids, selected_digests, selected_contexts = set(), set(), set()
    source_counts = []

    def candidates(source: PanelSource, counts: Counter[str]) -> Iterator[TaskSpec]:
        seen_ids, seen_digests, seen_contexts = set(), set(), set()
        for task in _panel_source_tasks(source.input, {}):
            digest, context = semantic_digest(task, False), task.context.model_dump_json()
            counts["input"] += 1
            if task.id in excluded_ids or digest in excluded_digests or context in excluded_contexts:
                raise ValueError(f"Test panel overlaps training or development: {task.id}")
            if "eval" not in task.tags:
                raise ValueError(f"Test panel source must reserve its tasks for evaluation: {task.id}")
            if _grader_kind(task) not in (GraderKind.IN_PROCESS, GraderKind.PACKAGE):
                raise ValueError(f"Test panel requires a local grader: {task.id}")
            if (
                task.id in selected_ids
                or task.id in seen_ids
                or digest in selected_digests
                or digest in seen_digests
                or context in selected_contexts
                or context in seen_contexts
            ):
                counts["duplicate"] += 1
                continue
            seen_ids.add(task.id)
            seen_digests.add(digest)
            seen_contexts.add(context)
            yield task

    for source in recipe.sources:
        counts: Counter[str] = Counter()
        count, tasks = seeded_sample(
            candidates(source, counts), size=source.count, key=lambda task: seeded_order(task.id, recipe.seed)
        )
        if count < source.count:
            raise ValueError(
                f"Test panel source shortfall: {source.input.uri}: requested {source.count}, eligible {count}"
            )
        for task in tasks:
            selected.append((task, source))
            selected_ids.add(task.id)
            selected_digests.add(semantic_digest(task, False))
            selected_contexts.add(task.context.model_dump_json())
        source_counts.append({"source": source.input.uri, **counts, "eligible": count, "selected": len(tasks)})

    rates = _panel_measurements(recipe.pass_rates)
    if rates and rates.keys() != selected_ids:
        raise ValueError("Pass-rate evidence must join every provisional panel task exactly once")
    agents = [rates[task.id] for task, source in selected if source.aaii_category == "agents" and rates]
    agent_pass_rate = sum(row["passed"] for row in agents) / sum(row["samples"] for row in agents) if agents else None
    if agent_pass_rate is not None and not 0.05 <= agent_pass_rate <= 0.95:
        selected = [(task, source) for task, source in selected if source.aaii_category != "agents"]
    selected.sort(key=lambda entry: entry[0].id)
    full = tuple(task for task, _ in selected)
    family_counts = Counter(source.family for _, source in selected)
    requested_fast = {family: count for family, count in recipe.fast_counts.items() if family in family_counts}
    if set(requested_fast) != family_counts.keys():
        raise ValueError("Fast-panel quotas must cover every selected family")
    total_fast = sum(recipe.fast_counts.values())
    total_weights = sum(requested_fast.values())
    fast_counts = {family: total_fast * count // total_weights for family, count in requested_fast.items()}
    remainder = total_fast - sum(fast_counts.values())
    for family in sorted(
        requested_fast, key=lambda family: (-(total_fast * requested_fast[family] % total_weights), family)
    )[:remainder]:
        fast_counts[family] += 1
    fast_ids = set()
    for family, size in fast_counts.items():
        count, sample = seeded_sample(
            (task for task, source in selected if source.family == family),
            size=size,
            key=lambda task: seeded_order(task.id, recipe.seed),
        )
        if count < size:
            raise ValueError(f"Fast-panel source shortfall for {family}")
        fast_ids.update(task.id for task in sample)
    fast = tuple(task for task in full if task.id in fast_ids)
    if recipe.provisional is not None:
        previous = tuple(_panel_source_tasks(recipe.provisional, {}))
        previous_by_id = {task.id: task for task in previous}
        if set(previous_by_id) != rates.keys() or any(previous_by_id[task.id] != task for task in full):
            raise ValueError("Final panel membership or task content differs from its measured provisional panel")
        if len(full) == len(previous):
            with (StoragePath(recipe.provisional.uri).parent / VALIDATION_FILENAME).open("rb") as stream:
                previous_fast = tuple(
                    TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN]) for row in pq.read_table(stream).to_pylist()
                )
            if previous_fast != fast:
                raise ValueError("Final fast-panel membership differs from its measured provisional panel")
    manifest = []
    for task, source in selected:
        rate = rates.get(task.id)
        manifest.append(
            {
                "task_id": task.id,
                "family": source.family,
                "aaii_category": source.aaii_category,
                "agentic": "agentic" in task.tags,
                "grader_kind": _grader_kind(task),
                "source": task.source.model_dump_json(),
                "license": source.license,
                "difficulty": None if rate is None else 1 - rate["mean_score"],
                "initializer_mean_score": None if rate is None else rate["mean_score"],
                "initializer_pass_rate": None if rate is None else rate["passed"] / rate["samples"],
                "attempted": None if rate is None else rate["attempted"],
                "scored": None if rate is None else rate["samples"],
            }
        )
    schema = pa.schema(
        [
            *TASK_MANIFEST_SCHEMA,
            ("source", pa.string()),
            ("license", pa.string()),
            ("difficulty", pa.float64()),
            ("initializer_mean_score", pa.float64()),
            ("initializer_pass_rate", pa.float64()),
            ("attempted", pa.int32()),
            ("scored", pa.int32()),
        ]
    )
    with TemporaryDirectory(prefix="test-panel-") as temporary:
        directory = Path(temporary)
        for name, tasks in ((FULL_VALIDATION_FILENAME, full), (VALIDATION_FILENAME, fast)):
            write_parquet_file(task_records(iter(tasks)), str(directory / name), schema=TASK_SCHEMA)
        write_parquet_file(iter(manifest), str(directory / TASK_MANIFEST_FILENAME), schema=schema)
        summary = {
            "recipe": recipe.model_dump(mode="json"),
            "full_validation_rows": len(full),
            "validation_rows": len(fast),
            "family_counts": dict(family_counts),
            "fast_family_counts": fast_counts,
            "category_counts": dict(Counter(source.aaii_category for _, source in selected)),
            "source_counts": source_counts,
            "agents_pass_rate": agent_pass_rate,
            "pass_rate_measured": recipe.pass_rates is not None,
            "grader_kind_counts": {
                name.removesuffix(".parquet"): _grader_counts(directory / name)
                for name in (FULL_VALIDATION_FILENAME, VALIDATION_FILENAME)
            },
        }
        destination = StoragePath(config.output_path)
        destination.mkdirs()
        for name in (FULL_VALIDATION_FILENAME, VALIDATION_FILENAME, TASK_MANIFEST_FILENAME):
            (destination / name).upload_from(str(directory / name))
        (destination / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def test_panel_step(recipe: TestPanelRecipe) -> ArtifactStep[Artifact]:
    """Bind a locally graded panel to pinned sources, exclusions and optional initializer measurements."""
    sources = (*[source.input for source in recipe.sources], recipe.training, recipe.development)
    if recipe.provisional is not None:
        sources = (*sources, recipe.provisional)
    dependencies = tuple(
        ArtifactStep.adopt(f"rl-data/selection-inputs/{recipe.name}/panel-{index}", source.version, source.uri)
        for index, source in enumerate(sources)
    )
    if recipe.pass_rates is not None:
        dependencies = (
            *dependencies,
            ArtifactStep.adopt(
                f"rl-data/selection-inputs/{recipe.name}/pass-rates", recipe.pass_rates.version, recipe.pass_rates.uri
            ),
        )

    def build_config(context: StepContext) -> TestPanelConfig:
        return TestPanelConfig(recipe, context.output_path)

    return ArtifactStep(
        name=f"{SELECTION_PREFIX}/{recipe.name}",
        version=recipe.version,
        artifact_type=Artifact,
        deps=dependencies,
        run=prepare_test_panel,
        build_config=build_config,
    )

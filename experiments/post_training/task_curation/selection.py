# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select training tasks and evaluation panels from admitted curation artifacts."""

import hashlib
import json
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
from taskcompendium.identity import canonical_sha256
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
from taskcompendium.pipeline.fingerprints import deduplication_key
from taskcompendium.pipeline.models import IntendedUse, NormalizedTask, RawRow
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample
from verifyit.json_objects import unique_object
from zephyr.writers import write_parquet_file

from experiments.post_training.task_curation.curation_catalog import MANIFEST_FILENAME
from experiments.post_training.task_curation.datasets.nemotron_ultra.components import BLENDS

SELECTION_PREFIX = "rl-data/task-selection"
SELECTION_INPUT_PREFIX = "rl-data/selection-inputs"
HELDOUT_BLEND = "rlvr1"
STRUCTURAL_FAMILY = "structural"
PARQUET_BATCH_ROWS = 256
# Answers per task in the initializer run that measures panel pass rates.
PASS_RATE_ATTEMPTS = 8
TRAIN_FILENAME = "train.parquet"
VALIDATION_FILENAME = "validation.parquet"
SUMMARY_FILENAME = "summary.json"
FULL_VALIDATION_FILENAME = "full_validation.parquet"
TASK_MANIFEST_FILENAME = "task_manifest.parquet"
TASK_SPEC_COLUMN = "task_spec"
TASK_JSON_COLUMN = "task_json"
LOWERED_TASK_SPEC_COLUMN = "lowered_task_spec"
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
EVAL_LITE_MANIFEST_SCHEMA = pa.schema(
    [
        *(field for field in TASK_MANIFEST_SCHEMA if field.name != "upstream_pass_rate"),
        *((name, pa.string()) for name in ("part", "source", "license")),
        ("difficulty", pa.float64()),
    ]
)


class GraderKind(StrEnum):
    PACKAGE = "package"
    IMAGE = "image"
    REMOTE_JUDGE = "remote_judge"
    IN_PROCESS = "in_process"


LOCAL_GRADERS = (GraderKind.IN_PROCESS, GraderKind.PACKAGE)


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
    exclusions: tuple[HeldoutSource, ...] = ()
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
    exclusions: tuple[HeldoutSource, ...]


@dataclass(frozen=True)
class HeldoutSelection:
    full: tuple[TaskSpec, ...]
    sample: tuple[TaskSpec, ...]
    family_counts: dict[str, int]


class EvalLiteSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    rows: PositiveInt
    format: PanelFormat
    family: str
    aaii_category: str
    license: str
    count: PositiveInt


class EvalLiteMeasurement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str
    version: str
    rows: PositiveInt
    model_revision: str


class EvalLiteRecipe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    sources: tuple[EvalLiteSource, ...]
    training: tuple[HeldoutSource, ...]
    development: HeldoutSource
    fast_counts: dict[str, PositiveInt]
    seed: int
    pass_rates: EvalLiteMeasurement | None = None


@dataclass(frozen=True)
class EvalLiteConfig:
    recipe: EvalLiteRecipe
    sources: dict[str, HeldoutSource]
    output_path: str


@dataclass(frozen=True)
class EvalLiteRate:
    task_id: str
    task_digest: str
    model_revision: str
    attempted: int
    samples: int
    passed: int
    mean_score: float


@dataclass(frozen=True)
class EvalLiteSelection:
    tasks: tuple[tuple[TaskSpec, EvalLiteSource], ...]
    fast: tuple[TaskSpec, ...]
    counts: dict[str, Counter[str]]


@dataclass(frozen=True)
class PanelKeys:
    """Tagged task keys: duplicates share an exact key; exclusions also match the whitespace-normalized statement."""

    exact: tuple[tuple[str, str], ...]
    exclusion: tuple[tuple[str, str], ...]


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
    raw = row[LOWERED_TASK_SPEC_COLUMN]
    document = json.loads(raw, object_pairs_hook=unique_object)
    original = document["task"]
    if original["schema_version"] != HELDOUT_SCHEMA_VERSION:
        raise ValueError("The pinned heldout TaskSpec version changed")
    if original["source"]["dataset"] == STRUCTURE_CLOSURE_DATASET:
        converted = structure_closure_task(
            RawRow(original["id"], Source.model_validate(original["source"]), {LOWERED_TASK_SPEC_COLUMN: raw})
        )
        if not isinstance(converted, NormalizedTask):
            raise ValueError(f"Native structure-closure import failed for {original['id']}: {converted}")
        return converted.task
    task = TaskSpec.model_validate({**original, "schema_version": SCHEMA_VERSION})
    if document["runtime"]["task_machine"] is not None:
        raise ValueError("Heldout source machine contract changed")
    if document["session"]["task_session"] != "shellbox" or document["session"]["max_turns"] != 1:
        raise ValueError("The heldout source must use the one-turn Shellbox contract")
    if isinstance(task.grader, ScriptGrader) and _grader_kind(task) == GraderKind.IMAGE:
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


def _stratified_sample(
    tasks: tuple[TaskSpec, ...], families: Mapping[str, str], counts: Mapping[str, int], seed: int
) -> tuple[TaskSpec, ...]:
    selected: set[str] = set()
    for family, size in counts.items():
        count, sample = seeded_sample(
            (task for task in tasks if families[task.id] == family),
            size=size,
            key=lambda task: seeded_order(task.id, seed),
        )
        if count < size:
            raise ValueError(f"Panel source lacks enough {family} tasks")
        selected.update(task.id for task in sample)
    return tuple(task for task in tasks if task.id in selected)


def select_heldout(full: tuple[TaskSpec, ...], recipe: SelectionRecipe) -> HeldoutSelection:
    """Retain the full exclusion set and choose a seeded panel within each family."""
    if len({task.id for task in full}) != len(full):
        raise ValueError("Heldout task IDs are not unique")
    families = {task.id: validation_family(task) for task in full}
    sample = _stratified_sample(
        full, families, dict.fromkeys(sorted(set(families.values())), recipe.validation_per_family), recipe.seed
    )
    return HeldoutSelection(
        full=full,
        sample=sample,
        family_counts=dict(Counter(families[task.id] for task in sample)),
    )


def _panel_source_tasks(
    source: HeldoutSource, grader_environments: Mapping[str, EnvironmentRequirements]
) -> Iterator[TaskSpec]:
    root, count = StoragePath(source.uri), 0
    tasks: Iterator[TaskSpec]
    if source.format == PanelFormat.EVALUATION:
        tasks = map(_chat_panel_task, evaluation_cases(source.uri))
    elif source.format == PanelFormat.NORMALIZED:
        if json.loads((root / MANIFEST_FILENAME).read_text())["intended_use"] != IntendedUse.EVAL:
            raise ValueError(f"Panel source is not reserved for evaluation: {source.uri}")
        paths = sorted((root / "normalize/*.parquet").glob(), key=str)
        raw = (row[TASK_JSON_COLUMN] for path in paths for row in _parquet_rows(path, TASK_JSON_COLUMN))
        tasks = (TaskSpec.model_validate_json(value) for value in raw if value is not None)
    elif source.format == PanelFormat.HISTORICAL:
        tasks = (validation_task(row, grader_environments) for row in _parquet_rows(root, LOWERED_TASK_SPEC_COLUMN))
    else:
        tasks = (TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN]) for row in _parquet_rows(root, TASK_SPEC_COLUMN))
    for task in tasks:
        count += 1
        yield task
    if count != source.rows:
        raise ValueError(f"Panel source row count changed: {source.uri}: expected {source.rows}, read {count}")


def _parquet_rows(path: StoragePath, column: str) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=PARQUET_BATCH_ROWS, columns=[column]):
            yield from batch.to_pylist()


def _chat_panel_task(case: EvaluationCase) -> TaskSpec:
    task, operation = case.task, case.item.operation
    if not isinstance(operation, ChatGeneration | TextGeneration):
        raise ValueError(f"EvalLite requires generated answers: {task.id}")
    if isinstance(operation, ChatGeneration) and operation.variants:
        zero_shot = [variant.context for variant in operation.variants if variant.fewshot == 0]
        if len(zero_shot) != 1:
            raise ValueError(f"EvalLite requires one zero-shot prompt: {task.id}")
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
    manifest = json.loads((root / MANIFEST_FILENAME).read_text())
    if manifest["mode"] != "full":
        raise ValueError(f"Selection requires FULL curation: {source.source_id}")
    if manifest["intended_use"] != IntendedUse.TRAIN:
        raise ValueError(f"Selection source is not a training population: {source.source_id}")
    for path in sorted((root / "final" / "*.parquet").glob(), key=str):
        yield from _parquet_rows(path, TASK_JSON_COLUMN)


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
        if not isinstance(task.grader, ScriptGrader) or _grader_kind(task) != GraderKind.PACKAGE:
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
    return (
        GraderKind.PACKAGE
        if environment.packages_lock is not None or environment.package_spec is not None
        else GraderKind.IMAGE
    )


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
    rows = _parquet_rows(StoragePath(str(path)), TASK_SPEC_COLUMN)
    counts = Counter(_grader_kind(TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN])) for row in rows)
    return {kind.value: counts[kind] for kind in GraderKind}


def write_training(
    config: SelectionConfig, heldout: tuple[TaskSpec, ...], exclusions: tuple[TaskSpec, ...], destination: Path
) -> dict[str, SourceCounts]:
    """Sample admitted local tasks after excluding panel overlaps and duplicates."""
    excluded_keys = {key for task in (*heldout, *exclusions) for key in _panel_keys(task).exclusion}
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
                if _grader_kind(task) not in LOCAL_GRADERS:
                    raise ValueError(f"Training selection requires local graders: {task.id}")
                if not excluded_keys.isdisjoint(_panel_keys(task).exclusion):
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
    if any(_grader_kind(task) not in LOCAL_GRADERS for task in heldout):
        raise ValueError("Development panel requires local graders")
    selection = select_heldout(heldout, config.recipe)
    with TemporaryDirectory(prefix="task-selection-") as temporary:
        directory = Path(temporary)
        excluded = tuple(task for source in config.exclusions for task in _panel_source_tasks(source, environments))
        counts = write_training(config, selection.full, excluded, directory / TRAIN_FILENAME)
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
        }
        task_files = (TRAIN_FILENAME, VALIDATION_FILENAME, FULL_VALIDATION_FILENAME)
        _publish_selection(directory, config.output_path, task_files, summary)


def _publish_selection(directory: Path, output_path: str, task_files: tuple[str, ...], summary: dict[str, Any]) -> None:
    """Upload the task files and manifest, then the summary with each task file's grader-kind counts."""
    grader_kinds = {name.removesuffix(".parquet"): _grader_counts(directory / name) for name in task_files}
    destination = StoragePath(output_path)
    destination.mkdirs()
    for name in (*task_files, TASK_MANIFEST_FILENAME):
        (destination / name).upload_from(str(directory / name))
    summary = {**summary, "grader_kind_counts": grader_kinds}
    (destination / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _pinned_inputs(name: str, sources: tuple[HeldoutSource, ...]) -> tuple[ArtifactStep, ...]:
    return tuple(
        ArtifactStep.adopt(f"{SELECTION_INPUT_PREFIX}/{name}-{index}", source.version, source.uri)
        for index, source in enumerate(sources)
    )


def _resolved(
    context: StepContext, sources: tuple[HeldoutSource, ...], steps: tuple[ArtifactStep, ...]
) -> tuple[HeldoutSource, ...]:
    return tuple(
        source.model_copy(update={"uri": context.artifact_path(step)})
        for source, step in zip(sources, steps, strict=True)
    )


def task_selection_step(curated: dict[str, ArtifactStep], recipe: SelectionRecipe) -> ArtifactStep[Artifact]:
    """Bind admitted sources and fixed evaluation exclusions to a versioned training selection."""
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
    heldout = _pinned_inputs(f"{recipe.name}-heldout", recipe.heldout)
    exclusions = _pinned_inputs(f"{recipe.name}-exclude", recipe.exclusions)
    inputs = {source_id: curated[source_id] for source_id in sorted(recipe.sources)}

    def build_config(ctx: StepContext) -> SelectionConfig:
        return SelectionConfig(
            tuple(
                SelectionInput(source_id, recipe.sources[source_id], ctx.artifact_path(step))
                for source_id, step in inputs.items()
            ),
            recipe,
            _resolved(ctx, recipe.heldout, heldout),
            ctx.output_path,
            _resolved(ctx, recipe.exclusions, exclusions),
        )

    return ArtifactStep(
        name=f"{SELECTION_PREFIX}/{recipe.name}",
        version=recipe.version,
        artifact_type=Artifact,
        deps=(*inputs.values(), *heldout, *exclusions),
        run=prepare,
        build_config=build_config,
    )


def _eval_lite_measurements(measurement: EvalLiteMeasurement | None) -> dict[str, EvalLiteRate]:
    if measurement is None:
        return {}
    with StoragePath(measurement.uri).open("rb") as stream:
        rows = [EvalLiteRate(**row) for row in pq.read_table(stream).to_pylist()]
    by_id = {row.task_id: row for row in rows}
    if len(by_id) != len(rows) or len(rows) != measurement.rows:
        raise ValueError("Panel pass-rate rows must have unique task IDs and the declared population")
    for row in rows:
        # The range comparison also refuses NaN and infinite scores.
        valid = (
            0 < row.samples
            and 0 <= row.passed <= row.samples <= row.attempted == PASS_RATE_ATTEMPTS
            and 0 <= row.mean_score <= 1
        )
        if not valid or row.model_revision != measurement.model_revision:
            raise ValueError(f"Panel pass-rate evidence is incomplete: {row.task_id}")
    return by_id


def _panel_keys(task: TaskSpec) -> PanelKeys:
    """Exact ID and conversation keys; exclusion adds the normalized final user statement when there is one."""
    users = [event.content for event in task.context.events if isinstance(event, TextMessage) and event.role == "user"]
    statement = (("statement", hashlib.sha256(" ".join(users[-1].split()).encode()).hexdigest()),) if users else ()
    exact = (("id", task.id), ("context", canonical_sha256(task.context.model_dump(mode="json"))))
    return PanelKeys(exact, (*exact, *statement))


def _check_pass_rates(tasks: tuple[TaskSpec, ...], rates: Mapping[str, EvalLiteRate]) -> None:
    if not {task.id for task in tasks} <= rates.keys():
        raise ValueError("Pass-rate evidence must cover every selected panel task")
    for task in tasks:
        if rates[task.id].task_digest != canonical_sha256(task.model_dump(mode="json")):
            raise ValueError(f"Pass-rate evidence measures different task content: {task.id}")


def _fast_subset(recipe: EvalLiteRecipe, selected: list[tuple[TaskSpec, EvalLiteSource]]) -> tuple[TaskSpec, ...]:
    """The seeded per-family fast quotas, in full-panel order."""
    if recipe.fast_counts.keys() != {part.family for _, part in selected}:
        raise ValueError("Fast-panel quotas must cover every selected family")
    return _stratified_sample(
        tuple(task for task, _ in selected),
        {task.id: part.family for task, part in selected},
        recipe.fast_counts,
        recipe.seed,
    )


def select_eval_lite(
    recipe: EvalLiteRecipe,
    sources: Mapping[str, Iterator[TaskSpec]],
    excluded: Iterator[TaskSpec],
    rates: Mapping[str, EvalLiteRate],
) -> EvalLiteSelection:
    """Select canonical tasks from source iterators, assigning duplicates to the first source."""
    excluded_keys = {key for task in excluded for key in _panel_keys(task).exclusion}
    selected: list[tuple[TaskSpec, EvalLiteSource]] = []
    taken: set[tuple[str, str]] = set()
    audit: dict[str, Counter[str]] = {}

    def candidates(part: EvalLiteSource, counts: Counter[str]) -> Iterator[TaskSpec]:
        seen: set[tuple[str, str]] = set()
        for task in sources[part.source_id]:
            counts["input"] += 1
            keys = _panel_keys(task)
            if not excluded_keys.isdisjoint(keys.exclusion):
                counts["overlap"] += 1
            elif "eval" not in task.tags:
                raise ValueError(f"EvalLite source must reserve its tasks for evaluation: {task.id}")
            elif _grader_kind(task) not in LOCAL_GRADERS:
                raise ValueError(f"EvalLite requires a local grader: {task.id}")
            elif not taken.isdisjoint(keys.exact) or not seen.isdisjoint(keys.exact):
                counts["duplicate"] += 1
            else:
                seen.update(keys.exact)
                yield task

    for part in recipe.sources:
        counts: Counter[str] = Counter()
        eligible, tasks = seeded_sample(
            candidates(part, counts), size=part.count, key=lambda task: seeded_order(task.id, recipe.seed)
        )
        if eligible < part.count:
            raise ValueError(f"EvalLite part shortfall: {part.source_id}: requested {part.count}, eligible {eligible}")
        selected.extend((task, part) for task in tasks)
        taken.update(key for task in tasks for key in _panel_keys(task).exact)
        counts.update(eligible=eligible, selected=len(tasks))
        audit[part.source_id] = counts
    selected.sort(key=lambda entry: entry[0].id)
    if recipe.pass_rates is not None:
        _check_pass_rates(tuple(task for task, _ in selected), rates)
    return EvalLiteSelection(tuple(selected), _fast_subset(recipe, selected), audit)


def prepare_eval_lite(config: EvalLiteConfig) -> None:
    """Read pinned sources and publish the selected full panel, fast subset, manifest and audit summary."""
    recipe = config.recipe
    rates = _eval_lite_measurements(recipe.pass_rates)
    selection = select_eval_lite(
        recipe,
        {name: _panel_source_tasks(source, {}) for name, source in config.sources.items()},
        (task for source in (*recipe.training, recipe.development) for task in _panel_source_tasks(source, {})),
        rates,
    )
    manifest = (
        {
            "task_id": task.id,
            "family": part.family,
            "aaii_category": part.aaii_category,
            "agentic": "agentic" in task.tags,
            "grader_kind": _grader_kind(task),
            "part": part.source_id.lower().replace(":", "-"),
            "source": task.source.model_dump_json(),
            "license": part.license,
            "difficulty": 1 - rates[task.id].mean_score if rates else None,
        }
        for task, part in selection.tasks
    )
    with TemporaryDirectory(prefix="eval-lite-") as temporary:
        directory = Path(temporary)
        full = (task for task, _ in selection.tasks)
        write_parquet_file(task_records(full), str(directory / FULL_VALIDATION_FILENAME), schema=TASK_SCHEMA)
        write_parquet_file(task_records(iter(selection.fast)), str(directory / VALIDATION_FILENAME), schema=TASK_SCHEMA)
        write_parquet_file(manifest, str(directory / TASK_MANIFEST_FILENAME), schema=EVAL_LITE_MANIFEST_SCHEMA)
        summary = {
            "recipe": recipe.model_dump(mode="json"),
            "full_validation_rows": len(selection.tasks),
            "validation_rows": len(selection.fast),
            "family_counts": dict(Counter(part.family for _, part in selection.tasks)),
            "category_counts": dict(Counter(part.aaii_category for _, part in selection.tasks)),
            "source_counts": {part.source_id: dict(selection.counts[part.source_id]) for part in recipe.sources},
        }
        _publish_selection(directory, config.output_path, (FULL_VALIDATION_FILENAME, VALIDATION_FILENAME), summary)


def eval_lite_step(curated: dict[str, ArtifactStep], recipe: EvalLiteRecipe) -> ArtifactStep[Artifact]:
    """Bind panel sources and training/development exclusions to their versioned artifacts."""
    missing = {part.source_id for part in recipe.sources} - curated.keys()
    if missing:
        raise ValueError(f"Panel requires corpus sources: {', '.join(sorted(missing))}")
    inputs = {part.source_id: curated[part.source_id] for part in recipe.sources}
    excluded = (*recipe.training, recipe.development)
    exclusions = _pinned_inputs(f"{recipe.name}/exclude", excluded)
    measurement = recipe.pass_rates
    measurement_step = (
        None
        if measurement is None
        else ArtifactStep.adopt(
            f"{SELECTION_INPUT_PREFIX}/{recipe.name}/pass-rates", measurement.version, measurement.uri
        )
    )

    def build_config(context: StepContext) -> EvalLiteConfig:
        *training, development = _resolved(context, excluded, exclusions)
        update = {
            "training": tuple(training),
            "development": development,
            "pass_rates": (
                measurement.model_copy(update={"uri": context.artifact_path(measurement_step)})
                if measurement is not None and measurement_step is not None
                else None
            ),
        }
        sources = {
            part.source_id: HeldoutSource(
                uri=context.artifact_path(inputs[part.source_id]),
                version=inputs[part.source_id].version,
                rows=part.rows,
                format=part.format,
            )
            for part in recipe.sources
        }
        return EvalLiteConfig(recipe.model_copy(update=update), sources, context.output_path)

    return ArtifactStep(
        name=f"{SELECTION_PREFIX}/{recipe.name}",
        version=recipe.version,
        artifact_type=Artifact,
        deps=(*inputs.values(), *exclusions, *((measurement_step,) if measurement_step is not None else ())),
        run=prepare_eval_lite,
        build_config=build_config,
    )

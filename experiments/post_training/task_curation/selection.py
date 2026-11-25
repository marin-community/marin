# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select training and evaluation task sets from curated and released task files."""

import hashlib
import io
import json
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import CurationMode
from finestore.rl.source_identity import canonical_source_id
from finestore.rl.strata import (
    DifficultyBands,
    MembershipState,
    ReferenceMeasurement,
    SelectionDefinition,
    SelectionParent,
    SelectionTaskRow,
    TaskStrata,
    read_selection_manifest,
)
from finestore.schema import arrow_schema
from marin.execution.artifact import Artifact, is_mutable_version, validate_path_segment, validate_version
from marin.execution.fingerprint import register_fingerprint
from marin.execution.lazy import ArtifactStep, StepContext
from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.structure_closure import structure_closure_task
from taskcompendium.evaluation.corpus import EvaluationCase, evaluation_cases
from taskcompendium.evaluation.spec import ChatGeneration, TextGeneration
from taskcompendium.grader import grader_config
from taskcompendium.grading_result import Outcome
from taskcompendium.identity import canonical_sha256
from taskcompendium.importers.evalchemy.assets.code_protocol import mbpp_example
from taskcompendium.models import (
    UPGRADABLE_SCHEMA_VERSION,
    ConversationInput,
    EnvironmentRequirements,
    GraderMachine,
    NoGrader,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
    grader_environment,
    grades_in_process,
    grades_with_judge,
)
from taskcompendium.pipeline.fingerprints import deduplication_key
from taskcompendium.pipeline.models import IntendedUse, NormalizedTask, RawRow
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample
from verifyit.json_objects import unique_object
from zephyr.writers import write_parquet_file

from experiments.post_training.task_curation import CANONICAL_ARTIFACT_ROOT
from experiments.post_training.task_curation.curation_catalog import MANIFEST_FILENAME

SELECTION_PREFIX = "rl-data/task-selection"
SELECTION_INPUT_PREFIX = "rl-data/selection-inputs"
HELDOUT_BLEND = "rlvr1"
STRUCTURAL_FAMILY = "structural"
PARQUET_BATCH_ROWS = 256
# Answers per task in the initializer run that measures panel pass rates.
PASS_RATE_ATTEMPTS = 8
MIN_PROBED_TASKS = 50
MIN_GRADED_SHARE = 0.9
TRAIN_FILENAME = "train.parquet"
VALIDATION_FILENAME = "validation.parquet"
SUMMARY_FILENAME = "summary.json"
FULL_VALIDATION_FILENAME = "full_validation.parquet"
TASK_MANIFEST_FILENAME = "task_manifest.parquet"
TASK_SPEC_COLUMN = "task_spec"
TASK_JSON_COLUMN = "task_json"
LOWERED_TASK_SPEC_COLUMN = "lowered_task_spec"
TASK_SCHEMA = pa.schema([pa.field(TASK_SPEC_COLUMN, pa.string(), nullable=False), *arrow_schema(TaskStrata)])
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
        "agentic": "coding",
        "calendar": "format",
        "competitive_coding": "coding",
        "reasoning_gym": "scientific_reasoning",
        "stem": "scientific_reasoning",
        STRUCTURAL_FAMILY: "format",
    }
)
TASK_MANIFEST_SCHEMA = arrow_schema(SelectionTaskRow)


class GraderKind(StrEnum):
    PACKAGE = "package"
    IMAGE = "image"
    REMOTE_JUDGE = "remote_judge"
    IN_PROCESS = "in_process"
    HARNESS = "harness"


LOCAL_GRADERS = frozenset((GraderKind.IN_PROCESS, GraderKind.PACKAGE))


class PanelFormat(StrEnum):
    HISTORICAL = "historical"
    TASK_SPEC = "task_spec"
    TASK_DIGEST = "task_digest"
    NORMALIZED = "normalized"
    EVALUATION = "evaluation"


class HeldoutSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str
    version: str
    rows: PositiveInt
    format: PanelFormat


class ReleaseAdmission(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str
    sha256: str


class ReleasedSource(HeldoutSource):
    format: Literal[PanelFormat.TASK_SPEC]
    family: str
    sha256: str
    agentic: bool
    license: str
    admission: ReleaseAdmission | None = Field(default=None, exclude_if=lambda admission: admission is None)


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
    admitted_graders: frozenset[GraderKind] = LOCAL_GRADERS
    released: tuple[ReleasedSource, ...] = Field(default=(), exclude_if=lambda sources: not sources)


def _selection_recipe_fingerprint(recipe: SelectionRecipe) -> dict[str, Any]:
    """Keep the serialized recipe shape of published local-grader selections."""
    fields = vars(recipe).copy()
    if recipe.admitted_graders == LOCAL_GRADERS:
        fields.pop("admitted_graders")
    if not recipe.released:
        fields.pop("released")
    recipe_type = type(recipe)
    return {"__repr__": f"{recipe_type.__module__}.{recipe_type.__qualname__}", "vars": fields}


register_fingerprint(SelectionRecipe, _selection_recipe_fingerprint)


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
    topic: str = "untagged"
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
    parent: SelectionParent | None = None
    difficulty_bands: DifficultyBands | None = None


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
    excluded: int = 0
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
    # The pinned heldout graders keep the 0.30 shape, so they enter through the 0.30 upgrade.
    task = TaskSpec.model_validate({**original, "schema_version": UPGRADABLE_SCHEMA_VERSION})
    if document["runtime"]["task_machine"] is not None:
        raise ValueError("Heldout source machine contract changed")
    if document["session"]["task_session"] != "shellbox" or document["session"]["max_turns"] != 1:
        raise ValueError("The heldout source must use the one-turn Shellbox contract")
    if isinstance(task.grader, ScriptGrader) and _grader_kind(task) == GraderKind.IMAGE:
        dataset = _grader_contract(task)["dataset"]
        if dataset not in grader_environments:
            from experiments.post_training.task_curation.datasets.nemotron_ultra.components import (  # noqa: PLC0415 - curation-only dependency
                BLENDS,
            )

            raise ValueError(f"No admitted grader environment for {BLENDS[HELDOUT_BLEND][dataset].id}")
        placement = task.grader.placement
        assert isinstance(placement, GraderMachine)
        placement = placement.model_copy(update={"environment": grader_environments[dataset]})
        task = task.model_copy(update={"grader": task.grader.model_copy(update={"placement": placement})})
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


def task_records(tasks: Iterator[TaskSpec], strata: Mapping[str, TaskStrata]) -> Iterator[dict[str, Any]]:
    for task in tasks:
        tags = strata[task.id]
        yield {TASK_SPEC_COLUMN: task.model_dump_json(), **tags.model_dump()}


def _selection_row(task: TaskSpec, strata: TaskStrata, *, state: MembershipState) -> SelectionTaskRow:
    return SelectionTaskRow(
        task_id=task.id,
        task_digest=canonical_sha256(task.model_dump(mode="json")),
        **strata.model_copy(update={"agentic": "agentic" in task.tags}).model_dump(),
        grader_kind=_grader_kind(task),
        raw_source=task.source.model_dump_json(),
        state=state,
    )


def _admitted_records(source: SelectionInput) -> Iterator[tuple[TaskSpec, str]]:
    root = StoragePath(source.root)
    manifest = json.loads((root / MANIFEST_FILENAME).read_text())
    if manifest["mode"] not in {CurationMode.SAMPLE, CurationMode.FULL}:
        raise ValueError(f"Selection requires SAMPLE/FULL curation: {source.source_id}")
    if manifest["intended_use"] != IntendedUse.TRAIN:
        raise ValueError(f"Selection source is not a training population: {source.source_id}")
    for path in sorted((root / "final" / "*.parquet").glob(), key=str):
        for row in _parquet_rows(path, TASK_JSON_COLUMN):
            task_json = row[TASK_JSON_COLUMN]
            yield TaskSpec.model_validate_json(task_json), task_json


def _release_admission_attempts(source: ReleasedSource) -> list[dict[str, Any]]:
    admission = source.admission
    if admission is None:
        raise ValueError(f"Released source requires admission evidence: {source.uri}")
    contents = StoragePath(admission.uri).read_bytes()
    if hashlib.sha256(contents).hexdigest() != admission.sha256:
        raise ValueError(f"Admission record SHA-256 mismatch: {admission.uri}")
    attempts = pq.read_table(
        io.BytesIO(contents), columns=["selection", "task_id", "task_digest", "outcome"]
    ).to_pylist()
    selections = {attempt["selection"] for attempt in attempts}
    if len(selections) != 1 or None in selections:
        raise ValueError("Admission attempts must name one versioned selection")
    (identity,) = selections
    name, _, version = identity.rpartition("@")
    validate_path_segment("selection name", name)
    validate_version(version)
    if is_mutable_version(version):
        raise ValueError("Admission requires an immutable probe selection")
    summary = StoragePath(CANONICAL_ARTIFACT_ROOT) / SELECTION_PREFIX / name / version / SUMMARY_FILENAME
    recipe = json.loads(summary.read_text())["recipe"]
    # The digest identifies the released Parquet independently of where it is stored.
    if not any(pin.get("sha256") == source.sha256 for pin in recipe.get("released", ())):
        raise ValueError(f"Probe selection {identity} does not pin released source {source.uri}")
    return attempts


def _check_release_admission(
    source: ReleasedSource, released: set[tuple[str, str]], attempts: list[dict[str, Any]]
) -> None:
    observed = set()
    graded = set()
    for attempt in attempts:
        key = (attempt["task_id"], attempt["task_digest"])
        if key not in released:
            continue
        observed.add(key)
        if attempt["outcome"] == Outcome.GRADED:
            graded.add(key)
    if len(observed) < MIN_PROBED_TASKS or len(graded) < MIN_GRADED_SHARE * len(observed):
        raise ValueError(
            f"Released source requires at least {MIN_PROBED_TASKS} probed tasks with "
            f"{MIN_GRADED_SHARE:.0%} graded. Found {len(graded)} of {len(observed)} graded: {source.uri}"
        )


def with_agentic_tag(task: TaskSpec) -> TaskSpec:
    """Use the agentic tag that selection adds to published rows from an agentic source."""
    return task if "agentic" in task.tags else task.model_copy(update={"tags": (*task.tags, "agentic")})


def _released_records(source: ReleasedSource, attempts: list[dict[str, Any]]) -> Iterator[tuple[TaskSpec, str]]:
    contents = StoragePath(source.uri).read_bytes()
    if hashlib.sha256(contents).hexdigest() != source.sha256:
        raise ValueError(f"Released source SHA-256 mismatch: {source.uri}")
    parquet = pq.ParquetFile(io.BytesIO(contents))
    count = parquet.metadata.num_rows
    if count != source.rows:
        raise ValueError(f"Panel source row count changed: {source.uri}: expected {source.rows}, read {count}")
    released = set()
    for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_ROWS, columns=[TASK_SPEC_COLUMN]):
        for raw in batch.column(0).to_pylist():
            task = TaskSpec.model_validate_json(raw)
            if "agentic" in task.tags and not source.agentic:
                raise ValueError(f"Released task {task.id} contradicts the source's agentic declaration")
            published = with_agentic_tag(task) if source.agentic else task
            released.add((published.id, canonical_sha256(published.model_dump(mode="json"))))
            yield task, published.model_dump_json()
    _check_release_admission(source, released, attempts)


def _grader_environments(sources: tuple[SelectionInput, ...]) -> dict[str, EnvironmentRequirements]:
    from experiments.post_training.task_curation.datasets.nemotron_ultra.components import (  # noqa: PLC0415 - curation-only dependency
        BLENDS,
    )

    components = {BLENDS[HELDOUT_BLEND][dataset].id: dataset for dataset in HELDOUT_FAMILIES}
    environments = {}
    for source in sources:
        if source.source_id not in components:
            continue
        record = next(_admitted_records(source), None)
        if record is None:
            raise ValueError(f"No admitted grader environment for {source.source_id}")
        task, _ = record
        if not isinstance(task.grader, ScriptGrader) or _grader_kind(task) != GraderKind.PACKAGE:
            raise ValueError(f"Admitted component needs a local script grader: {source.source_id}")
        environment = grader_environment(task.grader)
        assert environment is not None
        environments[components[source.source_id]] = environment
    return environments


def _grader_kind(task: TaskSpec) -> GraderKind:
    if isinstance(task.grader, NoGrader):
        return GraderKind.HARNESS
    if grades_with_judge(task.grader):
        return GraderKind.REMOTE_JUDGE
    if grades_in_process(task.grader):
        return GraderKind.IN_PROCESS
    environment = grader_environment(task.grader)
    if environment is None:
        raise ValueError(f"Task has no runnable grader: {task.id}")
    return (
        GraderKind.PACKAGE
        if environment.packages_lock is not None or environment.package_spec is not None
        else GraderKind.IMAGE
    )


def _manifest_records(tasks: tuple[TaskSpec, ...], active: set[str]) -> Iterator[dict[str, Any]]:
    for task in tasks:
        family = validation_family(task)
        row = _selection_row(
            task,
            TaskStrata(family=family, aai_category=AAII_CATEGORIES[family], source=task.source.dataset),
            state=MembershipState.ACTIVE if task.id in active else MembershipState.CANDIDATE,
        )
        upstream = None if family == STRUCTURAL_FAMILY else _grader_contract(task).get("pass_rate")
        yield row.model_copy(update={"measured_difficulty": 1 - upstream if upstream is not None else None}).model_dump(
            mode="json"
        )


def _grader_counts(path: Path) -> dict[str, int]:
    rows = _parquet_rows(StoragePath(str(path)), TASK_SPEC_COLUMN)
    counts = Counter(_grader_kind(TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN])) for row in rows)
    return {kind.value: counts[kind] for kind in GraderKind}


def write_training(
    config: SelectionConfig,
    heldout: tuple[TaskSpec, ...],
    exclusions: tuple[TaskSpec, ...],
    destination: Path,
    *,
    excluded_digests: frozenset[str] = frozenset(),
) -> dict[str, SourceCounts]:
    """Sample admitted training tasks after excluding panel overlaps and duplicates."""
    heldout_keys = {key for task in heldout for key in _panel_keys(task).exclusion}
    excluded_keys = {key for task in exclusions for key in _panel_keys(task).exclusion}
    selected_ids: set[str] = set()
    selected_contracts: set[str] = set()
    releases = {source.uri: source for source in config.recipe.released}
    attempts = {uri: _release_admission_attempts(source) for uri, source in releases.items()}
    inputs = (*config.sources, *(SelectionInput(uri, source.family, uri) for uri, source in releases.items()))
    counts = {source.source_id: SourceCounts() for source in inputs}
    families = {source.source_id: source.family for source in inputs}

    def candidates(sources: tuple[SelectionInput, ...]) -> Iterator[Candidate]:
        seen_ids: set[str] = set()
        seen_contracts: set[str] = set()
        for source in sources:
            release = releases.get(source.source_id)
            rows = (
                _released_records(release, attempts[source.source_id])
                if release is not None
                else _admitted_records(source)
            )
            for task, task_json in rows:
                source_counts = counts[source.source_id]
                source_counts.input += 1
                grader = _grader_kind(task)
                if grader not in config.recipe.admitted_graders:
                    raise ValueError(f"Training selection does not admit {grader} graders: {task.id}")
                if grader == GraderKind.HARNESS and "harbor" not in task.tags:
                    raise ValueError(f"Training task with a harness grader requires the harbor tag: {task.id}")
                exclusion = _panel_keys(task).exclusion
                if not heldout_keys.isdisjoint(exclusion):
                    source_counts.heldout_overlap += 1
                    continue
                if not excluded_keys.isdisjoint(exclusion) or (
                    excluded_digests and canonical_sha256(json.loads(task_json)) in excluded_digests
                ):
                    source_counts.excluded += 1
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
                yield Candidate(task.id, task_json, source.source_id, contract)

    def selected_records() -> Iterator[dict[str, Any]]:
        groups = (
            [(source.source_id, config.recipe.source_limits[source.source_id], (source,)) for source in inputs]
            if config.recipe.source_limits is not None
            else [
                (family, limit, tuple(source for source in inputs if source.family == family))
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
                task = TaskSpec.model_validate_json(row.task_json)
                family = families[row.source_id]
                yield {
                    TASK_SPEC_COLUMN: row.task_json,
                    **TaskStrata(
                        family=family,
                        aai_category=AAII_CATEGORIES.get(family, "untagged"),
                        source=task.source.dataset,
                        difficulty="unrated" if row.source_id in releases else "untagged",
                        agentic="agentic" in task.tags,
                    ).model_dump(),
                }

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
        excluded = []
        digests = set()
        for source in config.exclusions:
            tasks = _panel_source_tasks(source, environments)
            if source.format == PanelFormat.TASK_DIGEST:
                digests.update(canonical_sha256(task.model_dump(mode="json")) for task in tasks)
            else:
                excluded.extend(tasks)
        counts = write_training(
            config, selection.full, tuple(excluded), directory / TRAIN_FILENAME, excluded_digests=frozenset(digests)
        )
        annotations = tuple(_manifest_records(selection.full, {task.id for task in selection.sample}))
        strata = {
            row["task_id"]: TaskStrata.model_validate({key: row[key] for key in TaskStrata.model_fields})
            for row in annotations
        }
        write_parquet_file(
            task_records(iter(selection.sample), strata), str(directory / VALIDATION_FILENAME), schema=TASK_SCHEMA
        )
        write_parquet_file(
            task_records(iter(selection.full), strata), str(directory / FULL_VALIDATION_FILENAME), schema=TASK_SCHEMA
        )

        harness_datasets: set[str] = set()

        def manifest_rows():
            yield from annotations
            for batch in pq.ParquetFile(directory / TRAIN_FILENAME).iter_batches(batch_size=PARQUET_BATCH_ROWS):
                for value in batch.to_pylist():
                    task = TaskSpec.model_validate_json(value[TASK_SPEC_COLUMN])
                    tags = TaskStrata.model_validate({key: value[key] for key in TaskStrata.model_fields})
                    row = _selection_row(task, tags, state=MembershipState.ACTIVE)
                    if row.grader_kind == GraderKind.HARNESS:
                        harness_datasets.add(task.source.dataset)
                    yield row.model_dump(mode="json")

        write_parquet_file(manifest_rows(), str(directory / TASK_MANIFEST_FILENAME), schema=TASK_MANIFEST_SCHEMA)
        recipe_document = config.recipe.model_dump(mode="json")
        recipe_document["admitted_graders"] = sorted(config.recipe.admitted_graders)
        summary = {
            "selection_strata": (
                SelectionDefinition(
                    name=config.recipe.name,
                    version=config.recipe.version,
                    sampling="seeded source or family quotas; seeded validation per family",
                    task_files=(TRAIN_FILENAME, FULL_VALIDATION_FILENAME),
                    active_files=(TRAIN_FILENAME, VALIDATION_FILENAME),
                ).model_dump(mode="json")
            ),
            "recipe": recipe_document,
            "sources": [asdict(source) for source in config.sources],
            "source_counts": {source: asdict(count) for source, count in counts.items()},
            "training_rows": sum(count.selected for count in counts.values()),
            "training_family_counts": dict(sorted(config.recipe.family_limits.items())),
            "validation_rows": len(selection.sample),
            "full_validation_rows": len(selection.full),
            "validation_family_counts": selection.family_counts,
            "validation_task_ids": [task.id for task in selection.sample],
        }
        if harness_datasets:
            summary["harness_graded_datasets"] = sorted(harness_datasets)
        task_files = (TRAIN_FILENAME, VALIDATION_FILENAME, FULL_VALIDATION_FILENAME)
        _publish_selection(directory, config.output_path, task_files, summary)


def _publish_selection(directory: Path, output_path: str, task_files: tuple[str, ...], summary: dict[str, Any]) -> None:
    """Upload the task files and manifest, then the summary with each task file's grader-kind counts."""
    grader_kinds = {name.removesuffix(".parquet"): _grader_counts(directory / name) for name in task_files}
    summary = {**summary, "grader_kind_counts": grader_kinds}
    (directory / SUMMARY_FILENAME).write_text(json.dumps(summary))
    read_selection_manifest(str(directory))
    destination = StoragePath(output_path)
    if (destination / SUMMARY_FILENAME).exists():
        raise FileExistsError(f"selection version is already published: {output_path}")
    destination.mkdirs()
    for name in (*task_files, TASK_MANIFEST_FILENAME):
        (destination / name).upload_from(str(directory / name))
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


def _canonical_ids(sources: dict[str, str]) -> dict[str, str]:
    """Resolve historical recipe source keys to generated identities, refusing a silent collapse."""
    canonical: dict[str, str] = {}
    for key, family in sources.items():
        resolved = canonical_source_id(key)
        if resolved in canonical:
            raise ValueError(f"Selection sources collapse onto one identity: {resolved}")
        canonical[resolved] = family
    return canonical


def _canonical_eval_lite_sources(parts: tuple[EvalLiteSource, ...]) -> tuple[EvalLiteSource, ...]:
    """Resolve historical panel source keys to generated identities, refusing a silent collapse."""
    canonical: dict[str, EvalLiteSource] = {}
    for part in parts:
        resolved = canonical_source_id(part.source_id)
        if resolved in canonical:
            raise ValueError(f"Panel sources collapse onto one identity: {resolved}")
        canonical[resolved] = part.model_copy(update={"source_id": resolved})
    return tuple(canonical.values())


def task_selection_step(curated: dict[str, ArtifactStep], recipe: SelectionRecipe) -> ArtifactStep[Artifact]:
    """Bind admitted sources and fixed evaluation exclusions to a versioned training selection."""
    limits = (
        {canonical_source_id(source_id): limit for source_id, limit in recipe.source_limits.items()}
        if recipe.source_limits is not None
        else None
    )
    recipe = recipe.model_copy(update={"sources": _canonical_ids(recipe.sources), "source_limits": limits})
    missing = recipe.sources.keys() - curated.keys()
    if missing:
        raise ValueError(f"Selection requires curation sources: {', '.join(sorted(missing))}")
    if uncategorized := sorted({source.family for source in recipe.released} - AAII_CATEGORIES.keys()):
        raise ValueError(f"Released families require AAII categories: {', '.join(uncategorized)}")
    source_families = {**recipe.sources, **{source.uri: source.family for source in recipe.released}}
    if len(source_families) != len(recipe.sources) + len(recipe.released):
        raise ValueError("Selection sources must have distinct IDs")
    if not source_families or set(source_families.values()) != recipe.family_limits.keys():
        raise ValueError("Selection sources must cover exactly the families with quotas")
    if recipe.source_limits is not None:
        if recipe.source_limits.keys() != source_families.keys():
            raise ValueError("Source quotas must cover exactly the selection sources")
        family_totals: Counter[str] = Counter()
        for source, limit in recipe.source_limits.items():
            family_totals[source_families[source]] += limit
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


def select_eval_lite(
    recipe: EvalLiteRecipe,
    sources: Mapping[str, Iterator[TaskSpec]],
    excluded: Iterator[TaskSpec],
    rates: Mapping[str, EvalLiteRate],
    retained: Iterator[TaskSpec],
) -> EvalLiteSelection:
    """Select canonical tasks from source iterators, assigning duplicates to the first source."""
    excluded_keys = {key for task in excluded for key in _panel_keys(task).exclusion}
    selected: list[tuple[TaskSpec, EvalLiteSource]] = []
    taken = {key for task in retained for key in _panel_keys(task).exact}
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
    if recipe.pass_rates is not None and recipe.parent is None:
        _check_pass_rates(tuple(task for task, _ in selected), rates)
    return EvalLiteSelection(tuple(selected), audit)


def _joint_subset(rows: Mapping[str, SelectionTaskRow], quotas: Mapping[str, int], seed: int) -> set[str]:
    if quotas.keys() != {row.family for row in rows.values()}:
        raise ValueError("Fast-panel quotas must cover every selected family")
    selected = set()
    for family, quota in quotas.items():
        cells = {}
        for row in rows.values():
            if row.family == family and row.state != MembershipState.RETIRED:
                cells.setdefault((row.source, row.topic, row.difficulty), []).append(row.task_id)
        total = sum(map(len, cells.values()))
        if total < quota:
            raise ValueError(f"EvalLite active family shortfall: {family}: requested {quota}, eligible {total}")
        counts = {cell: quota * len(ids) // total for cell, ids in cells.items()}
        remainder = quota - sum(counts.values())
        order = sorted(
            cells, key=lambda cell: (-(quota * len(cells[cell]) % total), seeded_order(json.dumps(cell), seed))
        )
        for cell in order[:remainder]:
            counts[cell] += 1
        for cell, ids in cells.items():
            selected.update(sorted(ids, key=lambda task_id: seeded_order(task_id, seed))[: counts[cell]])
    return selected


def prepare_eval_lite(config: EvalLiteConfig) -> None:
    """Publish a retained task pool and a stratified active membership with reference evidence."""
    recipe = config.recipe
    rates = _eval_lite_measurements(recipe.pass_rates)
    raw_tasks, rows = {}, {}
    reference_revisions = set()
    if recipe.parent is not None:
        parent = read_selection_manifest(recipe.parent.root)
        if parent is None or (parent.definition.name, parent.definition.version) != (
            recipe.parent.name,
            recipe.parent.version,
        ):
            raise ValueError("EvalLite parent identity differs from its stored selection")
        reference_revisions.update(parent.definition.reference_revisions)
        for value in _parquet_rows(StoragePath(recipe.parent.root) / FULL_VALIDATION_FILENAME, TASK_SPEC_COLUMN):
            raw = value[TASK_SPEC_COLUMN]
            task = TaskSpec.model_validate_json(raw)
            raw_tasks[task.id] = raw
            rows[task.id] = parent.tasks[task.id].model_copy(
                update={"task_digest": canonical_sha256(json.loads(raw)), "agentic": "agentic" in task.tags}
            )
    selection = (
        select_eval_lite(
            recipe,
            {part.source_id: _panel_source_tasks(config.sources[part.source_id], {}) for part in recipe.sources},
            (task for source in (*recipe.training, recipe.development) for task in _panel_source_tasks(source, {})),
            rates,
            (TaskSpec.model_validate_json(raw) for raw in raw_tasks.values()),
        )
        if recipe.sources
        else EvalLiteSelection((), {})
    )
    for task, part in selection.tasks:
        row = _selection_row(
            task,
            TaskStrata(
                family=part.family, aai_category=part.aaii_category, source=task.source.dataset, topic=part.topic
            ),
            state=MembershipState.CANDIDATE,
        ).model_copy(update={"license": part.license})
        if task.id in rows:
            if rows[task.id].task_digest != row.task_digest:
                raise ValueError(f"EvalLite task content changed across versions: {task.id}")
            continue
        raw_tasks[task.id], rows[task.id] = task.model_dump_json(), row
    if recipe.pass_rates is not None:
        reference_revisions.add(recipe.pass_rates.model_revision)
    for task_id, row in rows.items():
        references = list(row.references)
        measurement = rates.get(task_id)
        if measurement is not None:
            if measurement.task_digest != row.task_digest:
                raise ValueError(f"Pass-rate evidence measures different task content: {task_id}")
            reference = ReferenceMeasurement(
                task_digest=measurement.task_digest,
                model_revision=measurement.model_revision,
                attempted=measurement.attempted,
                scored=measurement.samples,
                passed=measurement.passed,
                mean_score=measurement.mean_score,
            )
            if reference not in references:
                references.append(reference)
        saturated = bool(reference_revisions) and all(
            bool(model_rows := [ref for ref in references if ref.model_revision == revision])
            and all(
                ref.attempted == ref.scored == PASS_RATE_ATTEMPTS and ref.mean_score in (0.0, 1.0) for ref in model_rows
            )
            and len({ref.mean_score for ref in model_rows}) == 1
            for revision in reference_revisions
        )
        measured = 1 - measurement.mean_score if measurement is not None else row.measured_difficulty
        rows[task_id] = row.model_copy(
            update={
                "references": references,
                "measured_difficulty": measured,
                "difficulty": (
                    recipe.difficulty_bands.tag(1 - measured if measured is not None else None)
                    if recipe.difficulty_bands is not None
                    else row.difficulty
                ),
                "state": (
                    MembershipState.RETIRED
                    if saturated or row.state == MembershipState.RETIRED
                    else MembershipState.CANDIDATE
                ),
                "retirement_reason": (
                    "saturated for every declared reference model" if saturated else row.retirement_reason
                ),
            }
        )
    active = _joint_subset(rows, recipe.fast_counts, recipe.seed)
    for task_id in active:
        rows[task_id] = rows[task_id].model_copy(update={"state": MembershipState.ACTIVE})
    definition = SelectionDefinition(
        name=recipe.name,
        version=recipe.version,
        parent=recipe.parent,
        family_quotas=recipe.fast_counts,
        sampling=(
            "proportional joint source/topic/difficulty cells within fixed family quotas; "
            "largest remainders; seeded ties and task order"
        ),
        difficulty_bands=recipe.difficulty_bands,
        reference_revisions=tuple(sorted(reference_revisions)),
    )
    with TemporaryDirectory(prefix="eval-lite-") as temporary:
        directory = Path(temporary)
        for filename, ids in ((FULL_VALIDATION_FILENAME, sorted(raw_tasks)), (VALIDATION_FILENAME, sorted(active))):
            write_parquet_file(
                ({TASK_SPEC_COLUMN: raw_tasks[task_id], **rows[task_id].strata.model_dump()} for task_id in ids),
                str(directory / filename),
                schema=TASK_SCHEMA,
            )
        write_parquet_file(
            (row.model_dump(mode="json") for _, row in sorted(rows.items())),
            str(directory / TASK_MANIFEST_FILENAME),
            schema=TASK_MANIFEST_SCHEMA,
        )
        summary = {
            "selection_strata": definition.model_dump(mode="json"),
            "recipe": recipe.model_dump(mode="json"),
            "source_counts": {key: dict(value) for key, value in selection.counts.items()},
            "full_validation_rows": len(rows),
            "validation_rows": len(active),
            "validation_family_counts": dict(Counter(rows[task_id].family for task_id in active)),
        }
        _publish_selection(directory, config.output_path, (FULL_VALIDATION_FILENAME, VALIDATION_FILENAME), summary)


def eval_lite_step(curated: dict[str, ArtifactStep], recipe: EvalLiteRecipe) -> ArtifactStep[Artifact]:
    """Bind panel sources and training/development exclusions to their versioned artifacts."""
    recipe = recipe.model_copy(update={"sources": _canonical_eval_lite_sources(recipe.sources)})
    missing = {part.source_id for part in recipe.sources} - curated.keys()
    if missing:
        raise ValueError(f"Panel requires corpus sources: {', '.join(sorted(missing))}")
    inputs = {part.source_id: curated[part.source_id] for part in recipe.sources}
    excluded = (*recipe.training, recipe.development)
    exclusions = _pinned_inputs(f"{recipe.name}/exclude", excluded)
    parent_step = (
        None
        if recipe.parent is None
        else ArtifactStep.adopt(
            f"{SELECTION_INPUT_PREFIX}/{recipe.name}/parent", recipe.parent.version, recipe.parent.root
        )
    )
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
            "parent": (
                recipe.parent.model_copy(update={"root": context.artifact_path(parent_step)})
                if recipe.parent is not None and parent_step is not None
                else None
            ),
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
        deps=(
            *inputs.values(),
            *exclusions,
            *((measurement_step,) if measurement_step is not None else ()),
            *((parent_step,) if parent_step is not None else ()),
        ),
        run=prepare_eval_lite,
        build_config=build_config,
    )

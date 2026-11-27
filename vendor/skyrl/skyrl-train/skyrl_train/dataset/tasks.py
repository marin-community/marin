"""Convert source rows and lowered task specifications for the rollout loader."""

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from itertools import batched
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Dataset
from loguru import logger
from marinskyrl.task_sources import PreparedTaskCacheSource
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.harness.shell import task_prompt as marin_task_prompt
from rolloutengine.harness.assistant_prefix import ASSISTANT_PREFIX_HARNESS, assistant_prefix_prompt
from rolloutengine.models import TASK_ROLLOUT_COLUMN, MachineRuntimeSpec, TaskRolloutSpec
from rolloutengine.validation import task_machine_reason
from skyrl_train.config.rollout_validation import FAMILY_COLUMN
from skyrl_gym.rollout_specs import is_shell_harness, rollout_spec, shell_harness
from skyrl_gym.source_task import session_parameters, source_task
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    EnvironmentRequirements,
    NoGrader,
    Source,
    TaskSpec,
    TextMessage,
    grader_environment,
    grades_with_judge,
    task_subset,
)
from taskcompendium.runtime.shell import resolve_shell_tool
from transformers import PreTrainedTokenizerBase

from skyrl_train.dataset.dataset import PromptDataset, _prompt_token_count, dataset_num_proc, read_prompt_files
from skyrl_train.dataset.prepared_tasks import (
    PREPARED_TASK_BATCH_SIZE,
    PROMPT_LENGTH_COLUMN,
    load_prepared_tasks,
    prepared_task_cache_key,
    write_prepared_tasks,
)
from skyrl_train.inference_engines.chat_template import assistant_prefix_token_ids
from skyrl_train.rollouts.group_grader import task_group_grader
from skyrl_train.task_inputs import TaskInput, TrajectoryMetadata, task_messages
from finestore.rl.strata import SelectionManifest, TaskEvaluationIdentity, TaskStrata, read_selection_manifest

TASKCOMPENDIUM_ENVIRONMENT = "taskcompendium"
TASK_SPEC_COLUMN = "task_spec"
TASK_EVALUATION_IDENTITY = "task_evaluation_identity"
TASK_SCHEMA = pa.schema([pa.field(TASK_ROLLOUT_COLUMN, pa.string(), nullable=False)])
PARQUET_BATCH_SIZE = 1024


def task_evaluation_identity(
    task: dict, selection: SelectionManifest | None, source_id: str | None
) -> TaskEvaluationIdentity:
    digest = canonical_sha256(task)
    if selection is None:
        strata = TaskStrata.model_validate({"source": source_id, "agentic": "agentic" in task.get("tags", ())})
        return TaskEvaluationIdentity(task_digest=digest, strata=strata)
    identity = selection.identity(task["id"])
    if identity.task_digest != digest:
        raise ValueError(f"evaluation task content differs from selection manifest: {task['id']}")
    return identity


def with_evaluation_identities(dataset: Dataset, identities: Iterable[TaskEvaluationIdentity | None]) -> Dataset:
    return dataset.add_column(
        TASK_EVALUATION_IDENTITY,
        [identity.model_dump_json() if identity is not None else None for identity in identities],
    )


def task_prompt(lowered: TaskRolloutSpec) -> dict:
    """Prepare public messages and private worker inputs from a lowered task."""
    task = lowered.task
    return {
        TASK_ROLLOUT_COLUMN: lowered.model_dump_json(),
        "prompt": task_messages(lowered),
        "env_class": TASKCOMPENDIUM_ENVIRONMENT if is_shell_harness(lowered.harness) else lowered.harness,
        "data_source": task.source.dataset,
    }


class TaskDataset(PromptDataset):
    """Render public conversation fields and retain private lowered tasks for workers."""

    def prepare_dataset(self, dataset):
        return dataset.map(
            _stored_task_prompt, num_proc=dataset_num_proc(len(dataset), self.num_workers)
        ).with_transform(_restore_task_prompts)

    @staticmethod
    def prompt_token_count(
        tokenizer: PreTrainedTokenizerBase, row: dict, prompt_key: str, *, chat_template_kwargs: dict | None = None
    ) -> int:
        return task_prompt_token_count(tokenizer, row, prompt_key, chat_template_kwargs=chat_template_kwargs)

    def label(self, index: int) -> str:
        """The task id, which unlike the default row-number uid survives prompt-length filtering."""
        return TaskRolloutSpec.model_validate_json(self.dataframe[index][TASK_ROLLOUT_COLUMN]).task.id

    def collate_fn(self, rows) -> list[dict]:
        inputs = []
        for _, environment, extras, uid in rows:
            lowered = TaskRolloutSpec.model_validate_json(extras[TASK_ROLLOUT_COLUMN])
            if isinstance(lowered.task.grader, NoGrader) and {"extras", "config"} <= lowered.task.grader.contract.keys():
                extras = {**session_parameters(lowered.task).extras, **extras}
            extra_info = extras.get("extra_info") or {}
            ultra = extra_info.get("nemotron_ultra")
            metric_tags = (
                (f"nemotron_ultra/coverage/{ultra['blend']}/{ultra['agent']}",)
                if ultra is not None and "blend" in ultra and "agent" in ultra
                else ()
            )
            stored_identity = extras.get(TASK_EVALUATION_IDENTITY)
            identity = TaskEvaluationIdentity.model_validate_json(stored_identity) if stored_identity else None
            subset = (
                identity.strata.source
                if identity is not None and identity.panel_name is not None
                else task_subset(lowered.task)
            )
            inputs.append(
                {
                    "uid": uid,
                    "input": TaskInput(
                        spec=lowered,
                        metadata=TrajectoryMetadata(
                            uid=uid,
                            metric_class=environment,
                            data_source=extra_info.get("data_source") or extras.get("data_source"),
                            teacher_route=extras.get("teacher_route"),
                            metric_tags=metric_tags,
                            task_id=lowered.task.id,
                            subset_id=subset,
                            evaluation_identity=identity,
                        ),
                        group_grader=task_group_grader(lowered),
                    ),
                }
            )
        return inputs


def _stored_prompt(lowered: TaskRolloutSpec) -> dict:
    prepared = task_prompt(lowered)
    # Session-specific extras stay in the serialized task: mixed task families
    # do not share an Arrow schema. Restore them when yielding a training row.
    stored = {key: prepared[key] for key in (TASK_ROLLOUT_COLUMN, "env_class", "data_source")}
    # Message fields vary across tasks. JSON preserves missing fields and nested
    # values without inferring a different Arrow struct in each map worker.
    stored["prompt"] = json.dumps(prepared["prompt"], ensure_ascii=False)
    return stored


def _stored_task_prompt(row: dict) -> dict:
    return _stored_prompt(TaskRolloutSpec.model_validate_json(row[TASK_ROLLOUT_COLUMN]))


def _restore_task_prompts(batch: dict) -> dict:
    return {**batch, "prompt": [json.loads(prompt) for prompt in batch["prompt"]]}


def task_prompt_token_count(
    tokenizer: PreTrainedTokenizerBase, row: dict, prompt_key: str, *, chat_template_kwargs: dict | None = None
) -> int:
    """Include native task tools in the initial prompt's length budget."""
    lowered = TaskRolloutSpec.model_validate_json(row[TASK_ROLLOUT_COLUMN])
    if lowered.harness == ASSISTANT_PREFIX_HARNESS:
        prompt = assistant_prefix_prompt(lowered.task)
        options = {**prompt.options, "chat_template_kwargs": chat_template_kwargs or {}}
        return len(assistant_prefix_token_ids(tokenizer, list(prompt.messages), options))
    if not is_shell_harness(lowered.harness):
        return _prompt_token_count(tokenizer, row[prompt_key], **(chat_template_kwargs or {}))
    shell_tool = resolve_shell_tool(lowered.task, shell_harness(lowered.harness).shell_tool)
    start = marin_task_prompt(lowered.task, shell_tool, ())
    return _prompt_token_count(tokenizer, list(start.messages), **(chat_template_kwargs or {}), **start.options)


def source_row_task(
    row: Mapping[str, Any], index: int, *, source_name: str, environment_configs: Mapping[str, dict]
) -> TaskRolloutSpec:
    """Convert a source row once, with explicit session and machine settings."""
    row = dict(row)
    content = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source = Source(
        dataset=row.get("data_source") or source_name,
        revision=f"sha256:{hashlib.sha256(content).hexdigest()}",
        row=str(index),
        importer_revision="skyrl-task-session-v2",
    )
    prompt = row.pop("prompt")
    session_name = row.pop("env_class")
    if not isinstance(session_name, str):
        raise ValueError("The source session identifier must be a string")
    config = dict(environment_configs.get(session_name, {}))
    machine = config.pop("machine", None)
    machines = config.pop("machines", {})
    if session_name == "nemotron_ultra":
        machine = machines.get(row["extra_info"]["nemotron_ultra"]["agent"], machine)
    elif session_name == "openenv":
        machine = machines.get(row["env_name"], machine)
    source_session = dict(environment_configs["session"])
    row_turn_limit = row.get("max_turns")
    if row_turn_limit is None:
        row_turn_limit = (row.get("extra_info") or {}).get("max_turns")
    if row_turn_limit is not None:
        source_session["max_turns"] = int(row_turn_limit)
    session = {**source_session, **config.pop("session", {}), "task_session": session_name}
    requirements = (
        EnvironmentRequirements()
        if machine is None
        else EnvironmentRequirements.model_validate(machine["requirements"])
    )
    runtime = None if machine is None else MachineRuntimeSpec.model_validate(machine["runtime"])
    task = source_task(prompt, row, config, source, environment=requirements)
    return rollout_spec(task, session, task_machine=runtime)


def _source_task_prompt(row: dict, index: int, *, source_name: str, environment_configs: Mapping[str, dict]) -> dict:
    lowered = source_row_task(row, index, source_name=source_name, environment_configs=environment_configs)
    # Curriculum sampling reads source metadata before collation.
    return {**session_parameters(lowered.task).extras, **task_prompt(lowered)}


def _canonical_task_prompt(row: dict, _index: int, *, environment_configs: Mapping[str, dict]) -> dict:
    """Bind runtime settings when a canonical TaskSpec enters the trainer."""
    task = TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN])
    config = environment_configs["task_spec"]
    final_event = task.context.events[-1]
    # Only the assistant-prefix harness keeps an unfinished assistant message last.
    continues_assistant = isinstance(final_event, TextMessage) and final_event.role == "assistant"
    harness = (
        ASSISTANT_PREFIX_HARNESS
        if continues_assistant
        else config["harness_by_dataset"].get(task.source.dataset, config["default_harness"])
    )
    session = {**config["session"], "task_session": harness}
    if task.source.dataset in config["max_turns_by_dataset"]:
        session["max_turns"] = config["max_turns_by_dataset"][task.source.dataset]
    machine = config["task_machine_by_dataset"].get(task.source.dataset)
    if machine is None:
        # A dataset mixing shell-simulator and Linux tasks binds machines by command semantics. The binding has
        # no default-config entry, so it appears in a native configuration only when a launch sets it.
        by_semantics = config.get("task_machine_by_command_semantics", {})
        machine = by_semantics.get(task.environment_requirements.command_semantics)
    task_machine = (
        MachineRuntimeSpec.model_validate(machine)
        if machine is not None and task_machine_reason(task) is not None
        else None
    )
    verifier = None
    grader_build = grader_environment(task.grader)
    if grader_build is not None and not grades_with_judge(task.grader):
        machine = config["image_verifier"] if grader_build.docker_image is not None else config["package_verifier"]
        verifier = MachineRuntimeSpec.model_validate(machine)
    lowered = rollout_spec(task, session, task_machine=task_machine, verifier_machine=verifier)
    return {**_stored_prompt(lowered), FAMILY_COLUMN: row.get(FAMILY_COLUMN)}


def write_tasks(path: Path, records: Iterable[TaskRolloutSpec]) -> None:
    """Write private lowered tasks in bounded Parquet batches."""
    with pq.ParquetWriter(path, TASK_SCHEMA) as writer:
        path.chmod(0o600)
        for batch in batched(records, PARQUET_BATCH_SIZE):
            writer.write_table(
                pa.Table.from_pydict(
                    {TASK_ROLLOUT_COLUMN: [record.model_dump_json() for record in batch]},
                    TASK_SCHEMA,
                )
            )


def cache_tasks(records: Iterable[TaskRolloutSpec], cache_dir: Path) -> Path:
    """Cache private task Parquet under its content digest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_dir) as directory:
        temporary_path = Path(directory) / "tasks.parquet"
        write_tasks(temporary_path, records)
        with temporary_path.open("rb") as contents:
            digest = hashlib.file_digest(contents, "sha256").hexdigest()
        path = cache_dir / f"{digest}.parquet"
        temporary_path.chmod(0o600)
        temporary_path.replace(path)
    return path


class SourceTaskDataset(TaskDataset):
    """Prepare source rows as prompts and private lowered tasks."""

    def __init__(
        self,
        datasets: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        *,
        environment_configs: Mapping[str, dict],
        num_workers: int = 8,
        chat_template_kwargs: dict | None = None,
        prepared_cache: PreparedTaskCacheSource | None = None,
    ):
        self.environment_configs = environment_configs
        self.prepared_cache = prepared_cache
        identities: dict[str, TaskEvaluationIdentity | None] = {}

        def prepare_source(source: str, dataset: Dataset) -> Dataset:
            selection = read_selection_manifest(str(Path(source).parent))
            column = TASK_SPEC_COLUMN if TASK_SPEC_COLUMN in dataset.column_names else TASK_ROLLOUT_COLUMN
            annotations = []
            for row in dataset:
                if column not in row:
                    if selection is not None:
                        raise ValueError(f"selection input has no canonical task identity: {source}")
                    annotations.append(None)
                    continue
                document = json.loads(row[column])
                task = document if column == TASK_SPEC_COLUMN else document["task"]
                task_id = task["id"]
                identity = task_evaluation_identity(task, selection, row.get("source"))
                if selection is not None and TASK_SPEC_COLUMN in row:
                    carried = {key: row[key] for key in TaskStrata.model_fields if row.get(key) is not None}
                    if carried != identity.strata.model_dump(include=set(carried)):
                        raise ValueError(f"training row strata differ from selection manifest: {task_id}")
                if task_id in identities and identities[task_id] != identity:
                    raise ValueError(f"task has conflicting selection identities in one input set: {task_id}")
                identities[task_id] = identity
                annotations.append(identity)
            return with_evaluation_identities(dataset, annotations)

        super().__init__(
            list(datasets),
            tokenizer,
            max_prompt_length,
            num_workers=num_workers,
            chat_template_kwargs=chat_template_kwargs,
            prepare_source=prepare_source,
        )

    def _read_files_and_tokenize(self):
        prepared = self._prepared_rows_with_lengths()
        if prepared is None:
            return super()._read_files_and_tokenize()
        lengths = prepared.data.column(PROMPT_LENGTH_COLUMN)
        prepared = prepared.remove_columns(PROMPT_LENGTH_COLUMN).with_transform(_restore_task_prompts)
        logger.info("Total dataset size: {}", len(prepared))
        eligible = pc.indices_nonzero(pc.less_equal(lengths, self.max_prompt_length))
        self.prompts_dropped_too_long = len(prepared) - len(eligible)
        self.dataframe = prepared.select(eligible.to_numpy()) if self.prompts_dropped_too_long else prepared
        logger.info("Filtered dataset size: {}", len(self.dataframe))

    def _prepared_rows_with_lengths(self) -> Dataset | None:
        cache = self.prepared_cache
        task_settings = self.environment_configs.get("task_spec")
        if cache is None or self.datasets != [cache.path] or task_settings is None:
            return None
        key = prepared_task_cache_key(cache, self.tokenizer, task_settings, self.chat_template_kwargs)
        if key is None:
            return None
        with pq.ParquetFile(cache.path) as source:
            source_columns = source.schema_arrow.names
            source_rows = source.metadata.num_rows
        if not source_rows or TASK_SPEC_COLUMN not in source_columns or TASK_ROLLOUT_COLUMN in source_columns:
            return None

        path = StoragePath(cache.cache_dir) / f"{key}.parquet"
        required_columns = {
            TASK_ROLLOUT_COLUMN,
            TASK_EVALUATION_IDENTITY,
            FAMILY_COLUMN,
            PROMPT_LENGTH_COLUMN,
            self.prompt_key,
            self.env_class_key,
        }
        # Keep the backing files openable for the lifetime of the trainer's dataset.
        directory = TemporaryDirectory(prefix="skyrl-prepared-tasks-")
        arrow_path = Path(directory.name) / "prepared.arrow"
        prepared = load_prepared_tasks(path, key, required_columns, source_rows, arrow_path)
        if prepared is None:
            logger.info("Prepared task cache miss: {}", path)
            prepared = self._prepare_cached_rows(arrow_path)
            # Dataset.map returns the contiguous worker shards in source order.
            write_prepared_tasks(path, key, prepared.data.schema, (shard["filename"] for shard in prepared.cache_files))
        else:
            logger.info("Prepared task cache hit: {}", path)
        self._prepared_directory = directory
        return prepared

    def _prepare_cached_rows(self, arrow_path: Path) -> Dataset:
        source = read_prompt_files(self.datasets, prepare_source=self._prepare_source)
        environment_configs = self.environment_configs
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        template_kwargs = self.chat_template_kwargs

        def prepare(row: dict, index: int) -> dict:
            result = _canonical_task_prompt(row, index, environment_configs=environment_configs)
            result[PROMPT_LENGTH_COLUMN] = task_prompt_token_count(
                tokenizer,
                {**result, prompt_key: json.loads(result[prompt_key])},
                prompt_key,
                chat_template_kwargs=template_kwargs,
            )
            return result

        return self._map_task_rows(source, prepare, cache_file_name=str(arrow_path))

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        if TASK_ROLLOUT_COLUMN in dataset.column_names:
            return super().prepare_dataset(dataset)
        if TASK_SPEC_COLUMN in dataset.column_names:
            return self._map_task_rows(
                dataset,
                partial(_canonical_task_prompt, environment_configs=self.environment_configs),
            ).with_transform(_restore_task_prompts)
        return self._map_task_rows(
            dataset,
            partial(
                _source_task_prompt, source_name=", ".join(self.datasets), environment_configs=self.environment_configs
            ),
        )

    def _map_task_rows(
        self, dataset: Dataset, mapper: Callable[[dict, int], dict], *, cache_file_name: str | None = None
    ) -> Dataset:
        def prepare(row: dict, index: int) -> dict:
            carrier = row.get(TASK_EVALUATION_IDENTITY)
            result = mapper({key: value for key, value in row.items() if key != TASK_EVALUATION_IDENTITY}, index)
            result[TASK_EVALUATION_IDENTITY] = carrier
            return result

        return dataset.map(
            prepare,
            with_indices=True,
            remove_columns=dataset.column_names,
            num_proc=dataset_num_proc(len(dataset), self.num_workers),
            keep_in_memory=cache_file_name is None,
            cache_file_name=cache_file_name,
            writer_batch_size=PREPARED_TASK_BATCH_SIZE,
        )

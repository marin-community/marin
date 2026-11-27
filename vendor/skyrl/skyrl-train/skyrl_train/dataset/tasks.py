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
import pyarrow.parquet as pq
from datasets import Dataset
from rolloutengine.harness.shell import task_prompt as marin_task_prompt
from rolloutengine.harness.assistant_prefix import ASSISTANT_PREFIX_HARNESS, assistant_prefix_prompt
from rolloutengine.models import MachineRuntimeSpec, TaskRolloutSpec
from rolloutengine.validation import task_machine_reason
from skyrl_train.config.rollout_validation import FAMILY_COLUMN
from skyrl_gym.rollout_specs import is_shell_harness, rollout_spec, shell_harness
from skyrl_gym.source_task import session_parameters, source_task
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    EnvironmentRequirements,
    NoGrader,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    grades_with_judge,
)
from taskcompendium.runtime.shell import resolve_shell_tool
from transformers import PreTrainedTokenizerBase

from skyrl_train.dataset.dataset import PromptDataset, _prompt_token_count, dataset_num_proc
from skyrl_train.inference_engines.chat_template import assistant_prefix_token_ids
from skyrl_train.dataset.harbor_import import HARBOR_TRIAL_SESSION
from skyrl_train.rollouts.group_grader import task_group_grader
from skyrl_train.task_inputs import TaskInput, TrajectoryMetadata, task_messages
from finestore.rl.strata import SelectionManifest, TaskEvaluationIdentity, TaskStrata, read_selection_manifest

TASKCOMPENDIUM_ENVIRONMENT = "taskcompendium"
TASK_ROLLOUT_COLUMN = "task_rollout_spec"
TASK_SPEC_COLUMN = "task_spec"
TASK_EVALUATION_IDENTITY = "task_evaluation_identity"
TASK_SCHEMA = pa.schema([pa.field(TASK_ROLLOUT_COLUMN, pa.string(), nullable=False)])
PARQUET_BATCH_SIZE = 1024


def task_evaluation_identity(task: dict, selection: SelectionManifest | None) -> TaskEvaluationIdentity:
    digest = canonical_sha256(task)
    if selection is None:
        return TaskEvaluationIdentity(task_digest=digest, strata=TaskStrata(agentic="agentic" in task.get("tags", ())))
    identity = selection.identity(task["id"])
    if identity.task_digest != digest:
        raise ValueError(f"evaluation task content differs from selection manifest: {task['id']}")
    return identity


def with_evaluation_identities(dataset: Dataset, identities: Iterable[TaskEvaluationIdentity | None]) -> Dataset:
    return dataset.add_column(
        TASK_EVALUATION_IDENTITY, [identity.model_dump_json() if identity is not None else None for identity in identities]
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

    def uid(self, index: int) -> str:
        """Recovery subsets retain the original prompt identity through filtering."""
        if "_source_uid" in self.dataframe.column_names:
            return self.dataframe[index]["_source_uid"]
        return super().uid(index)

    def label(self, index: int) -> str:
        """The task id, which unlike the default row-number uid survives prompt-length filtering."""
        return TaskRolloutSpec.model_validate_json(self.dataframe[index][TASK_ROLLOUT_COLUMN]).task.id

    def collate_fn(self, rows) -> list[dict]:
        inputs = []
        for _, environment, extras, uid in rows:
            lowered = TaskRolloutSpec.model_validate_json(extras[TASK_ROLLOUT_COLUMN])
            if (
                isinstance(lowered.task.grader, NoGrader)
                and lowered.harness != HARBOR_TRIAL_SESSION
                and {"extras", "config"} <= lowered.task.grader.contract.keys()
            ):
                extras = {**session_parameters(lowered.task).extras, **extras}
            extra_info = extras.get("extra_info") or {}
            ultra = extra_info.get("nemotron_ultra")
            metric_tags = (
                (f"nemotron_ultra/coverage/{ultra['blend']}/{ultra['agent']}",)
                if ultra is not None and "blend" in ultra and "agent" in ultra
                else ()
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
                            evaluation_identity=(
                                TaskEvaluationIdentity.model_validate_json(extras[TASK_EVALUATION_IDENTITY])
                                if extras.get(TASK_EVALUATION_IDENTITY) is not None
                                else None
                            ),
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


def required_backend_machine(config: Mapping[str, Any], backend: str) -> Mapping[str, Any]:
    """The configured machine template for a task that declares its required backend."""
    machine = config.get("task_machine_by_backend", {}).get(backend)
    if machine is None or machine["backend"] != backend:
        raise ValueError(f"No task_machine_by_backend template serves the required {backend} backend")
    return machine


def canonical_task_spec(task: TaskSpec, config: Mapping[str, Any]) -> TaskRolloutSpec:
    """Bind the configured harness and machines to a canonical task without allocating them."""
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
    required = task.environment_requirements.required_backend
    machine = (
        config["task_machine_by_dataset"].get(task.source.dataset)
        if required is None
        else required_backend_machine(config, required)
    )
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
    grader = task.grader
    if (
        isinstance(grader, ScriptGrader | VerifyitGrader)
        and grader.environment is not None
        and not (isinstance(grader, VerifyitGrader) and grades_with_judge(grader))
    ):
        machine = (
            config["image_verifier"] if grader.environment.docker_image is not None else config["package_verifier"]
        )
        verifier = MachineRuntimeSpec.model_validate(machine)
    return rollout_spec(task, session, task_machine=task_machine, verifier_machine=verifier)


def _canonical_task_prompt(row: dict, _index: int, *, environment_configs: Mapping[str, dict]) -> dict:
    """Bind runtime settings when a canonical TaskSpec enters the trainer."""
    task = TaskSpec.model_validate_json(row[TASK_SPEC_COLUMN])
    lowered = canonical_task_spec(task, environment_configs["task_spec"])
    prepared = _stored_prompt(lowered)
    if "teacher_route" in row:
        prepared["teacher_route"] = row["teacher_route"]
    return {**prepared, FAMILY_COLUMN: row.get(FAMILY_COLUMN)}


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
    """Prepare source rows as in-memory prompts and private lowered tasks."""

    def __init__(
        self,
        datasets: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        *,
        environment_configs: Mapping[str, dict],
        num_workers: int = 8,
        chat_template_kwargs: dict | None = None,
    ):
        self.environment_configs = environment_configs
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
                identity = task_evaluation_identity(task, selection)
                if selection is not None and TASK_SPEC_COLUMN in row:
                    carried = {key: row[key] for key in TaskStrata.model_fields if key in row}
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

    def _map_task_rows(self, dataset: Dataset, mapper: Callable[[dict, int], dict]) -> Dataset:
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
            keep_in_memory=True,
        )

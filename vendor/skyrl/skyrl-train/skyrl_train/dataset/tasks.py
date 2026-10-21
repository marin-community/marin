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
from rolloutengine.harness.tools import task_prompt as marin_task_prompt
from rolloutengine.harness.prefill import prefill_prompt
from rolloutengine.models import MachineRuntimeSpec, TaskRolloutSpec
from skyrl_gym.rollout_specs import is_shell_harness, rollout_spec, shell_harness
from skyrl_gym.source_task import session_parameters, source_task
from taskcompendium.models import EnvironmentRequirements, NoGrader, Source
from taskcompendium.runtime.shell import resolve_shell_tool
from transformers import PreTrainedTokenizerBase

from skyrl_train.dataset.dataset import PromptDataset, _prompt_token_count
from skyrl_train.inference_engines.prefill import prefill_token_ids
from skyrl_train.dataset.harbor_import import HARBOR_TRIAL_SESSION
from skyrl_train.rollouts.group_grader import task_group_grader
from skyrl_train.task_inputs import PREFILL_HARNESS, TaskInput, TrajectoryMetadata, task_messages

TASKCOMPENDIUM_ENVIRONMENT = "taskcompendium"
TASK_ROLLOUT_COLUMN = "task_rollout_spec"
TASK_SCHEMA = pa.schema([pa.field(TASK_ROLLOUT_COLUMN, pa.string(), nullable=False)])
PARQUET_BATCH_SIZE = 1024


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
        return dataset.map(_stored_task_prompt, num_proc=self.num_workers).with_transform(_restore_task_prompts)

    @staticmethod
    def prompt_token_count(
        tokenizer: PreTrainedTokenizerBase, row: dict, prompt_key: str, *, chat_template_kwargs: dict | None = None
    ) -> int:
        return task_prompt_token_count(tokenizer, row, prompt_key, chat_template_kwargs=chat_template_kwargs)

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
                        ),
                        group_grader=task_group_grader(lowered),
                    ),
                }
            )
        return inputs


def _stored_task_prompt(row: dict) -> dict:
    prepared = task_prompt(TaskRolloutSpec.model_validate_json(row[TASK_ROLLOUT_COLUMN]))
    # Session-specific extras stay in the serialized task: mixed task families
    # do not share an Arrow schema. Restore them when yielding a training row.
    stored = {key: prepared[key] for key in (TASK_ROLLOUT_COLUMN, "env_class", "data_source")}
    # Message fields vary across tasks. JSON preserves missing fields and nested
    # values without inferring a different Arrow struct in each map worker.
    stored["prompt"] = json.dumps(prepared["prompt"], ensure_ascii=False)
    return stored


def _restore_task_prompts(batch: dict) -> dict:
    return {**batch, "prompt": [json.loads(prompt) for prompt in batch["prompt"]]}


def task_prompt_token_count(
    tokenizer: PreTrainedTokenizerBase, row: dict, prompt_key: str, *, chat_template_kwargs: dict | None = None
) -> int:
    """Include native task tools in the initial prompt's length budget."""
    lowered = TaskRolloutSpec.model_validate_json(row[TASK_ROLLOUT_COLUMN])
    if lowered.harness == PREFILL_HARNESS:
        prompt = prefill_prompt(lowered.task)
        options = {**prompt.options, "chat_template_kwargs": chat_template_kwargs or {}}
        return len(prefill_token_ids(tokenizer, list(prompt.messages), options))
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
    ):
        self.environment_configs = environment_configs
        super().__init__(list(datasets), tokenizer, max_prompt_length, num_workers=num_workers)

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        return self._map_task_rows(
            dataset,
            partial(
                _source_task_prompt, source_name=", ".join(self.datasets), environment_configs=self.environment_configs
            ),
        )

    def _map_task_rows(self, dataset: Dataset, mapper: Callable[[dict, int], dict]) -> Dataset:
        return dataset.map(
            mapper,
            with_indices=True,
            remove_columns=dataset.column_names,
            num_proc=self.num_workers,
            keep_in_memory=True,
        )

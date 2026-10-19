"""Prepare mixed Nemotron rows as portable tasks before rollout execution."""

from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset
from skyrl_gym.source_task import session_parameters
from rolloutengine.spec import LoweredTaskSpec, TaskSessionSpec
from taskcompendium.models import NoGrader
from transformers import PreTrainedTokenizerBase

from skyrl_train.dataset.harbor import HARBOR_ID_PREFIX, materialize_harbor_tasks
from skyrl_train.dataset.tasks import (
    LOWERED_TASK_COLUMN,
    PARQUET_BATCH_SIZE,
    SourceTaskDataset,
    source_row_task,
    task_prompt,
)


def terminal_task_index(path: Path) -> dict[str, LoweredTaskSpec]:
    """Index lowered Harbor tasks by their package and source instance identities."""
    result = {}
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=[LOWERED_TASK_COLUMN]):
            for row in batch.to_pylist():
                record = LoweredTaskSpec.model_validate_json(row[LOWERED_TASK_COLUMN])
                identifiers = {
                    record.task.id.casefold(),
                    *(
                        tag.removeprefix(HARBOR_ID_PREFIX)
                        for tag in record.task.tags
                        if tag.startswith(HARBOR_ID_PREFIX)
                    ),
                }
                for identifier in identifiers:
                    if identifier in result and result[identifier] != record:
                        raise ValueError(f"Duplicate terminal-bench task ID {identifier!r}")
                    result[identifier] = record
    return result


def resolve_terminal_task(lowered: LoweredTaskSpec, terminals: Mapping[str, LoweredTaskSpec]) -> LoweredTaskSpec:
    """Resolve a terminal source row before rollout execution."""
    task = lowered.task
    parameters = session_parameters(task)
    extras = parameters.extras
    ultra = (extras.get("extra_info") or {}).get("nemotron_ultra")
    if ultra is None:
        return lowered
    if not all(isinstance(ultra.get(key), str) and ultra[key] for key in ("blend", "agent")):
        raise ValueError("Nemotron Ultra tasks require non-empty blend and agent fields")
    if ultra.get("route") != "terminal_bench":
        return lowered
    identifier = ultra.get("terminal_bench_instance_id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("A terminal-bench row must provide terminal_bench_instance_id")
    if identifier.casefold() not in terminals:
        raise ValueError(f"Terminal-bench task {identifier!r} is absent from the configured task data")
    terminal = terminals[identifier.casefold()]
    return terminal.model_copy(
        update={
            "task": terminal.task.model_copy(update={"id": task.id, "source": task.source}),
        }
    )


def _nemotron_task_prompt(
    row: dict,
    index: int,
    *,
    source_name: str,
    environment_configs: Mapping[str, dict],
    terminals: Mapping[str, LoweredTaskSpec],
) -> dict:
    source = source_row_task(row, index, source_name=source_name, environment_configs=environment_configs)
    extras = session_parameters(source.task).extras
    return {**extras, **task_prompt(resolve_terminal_task(source, terminals))}


class NemotronTaskDataset(SourceTaskDataset):
    """Convert answer, tool, and terminal source rows to portable tasks."""

    def __init__(
        self,
        datasets: Sequence[str],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        *,
        environment_configs: Mapping[str, dict],
        terminal_bench_data: Sequence[str | Mapping[str, Any]],
        cache_dir: Path,
        grader_override: NoGrader | None = None,
        num_workers: int = 8,
    ):
        cache_dir = cache_dir.expanduser()
        self.terminals = terminal_task_index(
            materialize_harbor_tasks(
                terminal_bench_data,
                cache_dir=cache_dir,
                grader_override=grader_override,
                session=TaskSessionSpec.model_validate({**environment_configs["session"], "task_session": "shellbox"}),
            )
        )
        super().__init__(
            datasets,
            tokenizer,
            max_prompt_length,
            environment_configs=environment_configs,
            num_workers=num_workers,
        )

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        return self._map_task_rows(
            dataset,
            partial(
                _nemotron_task_prompt,
                source_name=", ".join(self.datasets),
                environment_configs=self.environment_configs,
                terminals=self.terminals,
            ),
        )

import json
import os
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from rolloutengine.models import MachineRuntimeSpec
from shellbox.machine import Backend
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, ExecutionRequirements
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import SourceTaskDataset, TaskDataset, source_row_task, write_tasks
from skyrl_train.task_inputs import task_messages


class _StubTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt):
        return messages if isinstance(messages, str) else "".join(message["content"] for message in messages)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("row_count", [3, 8])
def test_task_preprocessing_preserves_filtered_rows_and_forks_only_large_inputs(
    tmp_path, monkeypatch, fork_records, native, row_count
):
    monkeypatch.setattr("skyrl_train.dataset.dataset.PARALLEL_DATASET_MIN_ROWS", 4)
    rows = [
        {
            "prompt": [{"role": "user", "content": "x" * 200 if i % 3 == 1 else f"Question {i}"}],
            "env_class": "gsm8k",
            "reward_spec": {"ground_truth": str(i)},
            "data_source": "fixture",
            "teacher_route": "math",
        }
        for i in range(row_count)
    ]
    configs = OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True)
    path = tmp_path / ("tasks.parquet" if native else "source.jsonl")
    if native:
        tasks = [
            source_row_task(row, i, source_name="fixture", environment_configs=configs) for i, row in enumerate(rows)
        ]
        write_tasks(path, tasks)
        dataset = TaskDataset(
            [str(path)],
            _StubTokenizer(),
            100,
            minimum_verifier_timeout=configs["task_spec"]["minimum_verifier_timeout"],
            num_workers=2,
        )
    else:
        path.write_text("\n".join(json.dumps(row) for row in rows))
        dataset = SourceTaskDataset([str(path)], _StubTokenizer(), 100, environment_configs=configs, num_workers=2)
    indices = [i for i in range(row_count) if i % 3 != 1]
    prepared = dataset.collate_fn([dataset[i] for i in range(len(dataset))])
    assert [task_messages(row["input"].spec) for row in prepared] == [rows[i]["prompt"] for i in indices]
    assert [row["input"].spec.task.source.row for row in prepared] == [str(i) for i in indices]
    assert [row["input"].metadata.data_source for row in prepared] == ["fixture"] * len(indices)
    assert all(row["input"].metadata.teacher_route == "math" for row in prepared)
    assert all(row["input"].spec.limits.verifier_timeout == 1800 for row in prepared)
    assert bool(tuple(fork_records.iterdir())) == (row_count == 8)


@pytest.mark.parametrize("stored", [False, True])
def test_frozen_tasks_bind_verifier_floor_and_typed_local_actor(tmp_path, stored):
    configs = OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True)
    lowered = source_row_task(
        {
            "prompt": [{"role": "user", "content": "Compute 2 + 2."}],
            "env_class": "gsm8k",
            "reward_spec": {"ground_truth": "4"},
        },
        0,
        source_name="fixture",
        environment_configs=configs,
    )
    task = lowered.task.model_copy(
        update={
            "execution_requirements": ExecutionRequirements(
                cpus=1,
                memory_mb=2048,
                storage_mb=2048,
                agent_timeout_sec=600,
                verifier_timeout_sec=2400,
            ),
            "environment_requirements": EnvironmentRequirements(
                required_backend=Backend.LOCAL,
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_image="fixture@sha256:" + "a" * 64,
            ),
        }
    )
    path = tmp_path / "frozen.parquet"
    if stored:
        # Stored inputs carry their selected machine; only the common runtime deadline is rebound.
        runtime = MachineRuntimeSpec.model_validate(configs["task_spec"]["required_actor"])
        write_tasks(path, [lowered.model_copy(update={"task": task, "task_machine": runtime})])
    else:
        pq.write_table(pa.table({"task_spec": [task.model_dump_json()]}), path)
    dataset = SourceTaskDataset([str(path)], _StubTokenizer(), 100, environment_configs=configs, num_workers=1)
    loaded = dataset.collate_fn([dataset[0]])[0]["input"].spec
    assert loaded.task == task
    assert loaded.task_machine.backend == Backend.LOCAL
    assert loaded.task_machine.network.value == "deny"
    assert loaded.limits.verifier_timeout == 2400
    assert (
        loaded.limits.attempt_timeout is None or loaded.limits.attempt_timeout >= 2400 + loaded.limits.cleanup_timeout
    )


@pytest.fixture
def fork_records(tmp_path):
    records = tmp_path / "forks"
    records.mkdir()
    enabled = True

    def record_fork():
        if enabled:
            (records / str(os.getpid())).touch()

    os.register_at_fork(after_in_child=record_fork)
    try:
        yield records
    finally:
        enabled = False

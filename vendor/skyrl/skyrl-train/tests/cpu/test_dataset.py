import json
import os
from collections import Counter
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import SourceTaskDataset, TaskDataset, source_row_task, write_tasks
from skyrl_train.task_inputs import task_messages
from skyrl_train.domain_sampling import DomainWeightedOrder
from tests.cpu.task_specs import text_task


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
        dataset = TaskDataset([str(path)], _StubTokenizer(), 100, num_workers=2)
    else:
        path.write_text("\n".join(json.dumps(row) for row in rows))
        dataset = SourceTaskDataset([str(path)], _StubTokenizer(), 100, environment_configs=configs, num_workers=2)
    indices = [i for i in range(row_count) if i % 3 != 1]
    prepared = dataset.collate_fn([dataset[i] for i in range(len(dataset))])
    assert [task_messages(row["input"].spec) for row in prepared] == [rows[i]["prompt"] for i in indices]
    assert [row["input"].spec.task.source.row for row in prepared] == [str(i) for i in indices]
    assert [row["input"].metadata.data_source for row in prepared] == ["fixture"] * len(indices)
    assert all(row["input"].metadata.teacher_route == "math" for row in prepared)
    assert bool(tuple(fork_records.iterdir())) == (row_count == 8)


def test_task_spec_curriculum_routes_survive_preprocessing_and_reach_collation(tmp_path):
    configs = OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True)
    tasks = [text_task(f"fixture-{i}", source=f"source-{i % 2}") for i in range(6)]
    routes = [f"route-{i % 2}" for i in range(6)]
    path = tmp_path / "admitted.parquet"
    pq.write_table(pa.table({"task_spec": [task.model_dump_json() for task in tasks], "teacher_route": routes}), path)
    dataset = SourceTaskDataset([str(path)], _StubTokenizer(), 100, environment_configs=configs, num_workers=1)
    order = DomainWeightedOrder(dataset.dataframe, {"route-0": .5, "route-1": .5}, seed=17, window_size=4)
    indices = [order.next_index() for _ in range(4)]
    prepared = dataset.collate_fn([dataset[index] for index in indices])
    assert Counter(row["input"].metadata.teacher_route for row in prepared) == {"route-0": 2, "route-1": 2}
    for index, row in zip(indices, prepared, strict=True):
        assert row["input"].spec.task == tasks[index]
        assert row["input"].metadata.teacher_route == routes[index]


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

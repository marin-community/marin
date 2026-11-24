import json
import os
from dataclasses import replace
from importlib.util import spec_from_file_location
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marinskyrl.task_sources import PreparedTaskCacheSource
from omegaconf import OmegaConf
from skyrl_train import task_inputs
from skyrl_train.config.rollout_validation import FAMILY_COLUMN, GROUPS_PER_FAMILY_KEY
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.prepared_tasks import PROMPT_LENGTH_COLUMN
from skyrl_train.dataset.tasks import TASK_SPEC_COLUMN, SourceTaskDataset, TaskDataset, source_row_task, write_tasks
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.task_inputs import task_messages
from taskcompendium.models import ExitCodeReward, GraderMachine, ScriptGrader
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from tests.cpu.task_specs import NATIVE_ENVIRONMENT, text_task, write_selection_panel


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
        cache = PreparedTaskCacheSource(str(path), "legacy@1", str(tmp_path / "prepared"))
        dataset = SourceTaskDataset(
            [str(path)], _StubTokenizer(), 100, environment_configs=configs, num_workers=2, prepared_cache=cache
        )
    indices = [i for i in range(row_count) if i % 3 != 1]
    prepared = dataset.collate_fn([dataset[i] for i in range(len(dataset))])
    assert [task_messages(row["input"].spec) for row in prepared] == [rows[i]["prompt"] for i in indices]
    assert [row["input"].spec.task.source.row for row in prepared] == [str(i) for i in indices]
    assert [row["input"].metadata.data_source for row in prepared] == ["fixture"] * len(indices)
    assert all(row["input"].metadata.teacher_route == "math" for row in prepared)
    assert bool(tuple(fork_records.iterdir())) == (row_count == 8)


@pytest.mark.parametrize(
    "families,error",
    [
        (None, "requires a family column"),
        ([None] * 4 + ["math"] * 3 + ["agentic"], "requires a family on every training row"),
        (["math"] * 4 + ["agentic"] * 4, "requires 2 rows for family 'agentic', found 1"),
    ],
    ids=["missing-column", "null-before-named", "undersized-after-filtering"],
)
def test_training_context_checks_family_pools_on_prepared_rows(tmp_path, monkeypatch, families, error):
    monkeypatch.setattr("skyrl_train.dataset.dataset.PARALLEL_DATASET_MIN_ROWS", 4)
    config = get_default_config()
    config.trainer.train_batch_size = 2
    config.trainer.rollout_buffer.batch_policy = "full_batch"
    config.trainer.rollout_buffer.max_staleness_steps = 1
    OmegaConf.update(config, GROUPS_PER_FAMILY_KEY, {"math": 1, "agentic": 1}, force_add=True)
    config.data.sampling.kind = None
    configs = OmegaConf.to_container(config.environment.task_sessions, resolve=True)
    tasks = [
        source_row_task(
            {
                "prompt": [{"role": "user", "content": "x" * 200 if 4 <= index <= 6 else f"Question {index}"}],
                "env_class": "gsm8k",
                "reward_spec": {"ground_truth": str(index)},
            },
            index,
            source_name="fixture",
            environment_configs=configs,
        )
        for index in range(8)
    ]
    path = tmp_path / "tasks.parquet"
    if families is None:
        write_tasks(path, tasks)
    else:
        pq.write_table(
            pa.table({TASK_SPEC_COLUMN: [task.task.model_dump_json() for task in tasks], FAMILY_COLUMN: families}), path
        )
    dataset = SourceTaskDataset([str(path)], _StubTokenizer(), 100, environment_configs=configs, num_workers=2)
    assert dataset.prompts_dropped_too_long == 3
    if families is not None:
        assert list(dataset.dataframe.with_format(None)[FAMILY_COLUMN]) == families[:4] + families[-1:]
    with pytest.raises(ValueError, match=error):
        TrainingContext.from_config(config, dataset, workers=None)


def _selection_tokenizer(*, split_punctuation=False, prefix=""):
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace() if split_punctuation else WhitespaceSplit()
    return TokenizersBackend(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template=(
            prefix
            + "{% if tools %} available tools {% endif %}"
            + "{% for message in messages %}{{ message.content }} {% endfor %}"
            + "{{ extra | default('') }}"
        ),
    )


@pytest.fixture
def prepared_selection(tmp_path):
    configs = OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True)
    selection = tmp_path / "selection"
    selection.mkdir()
    path = selection / "validation.parquet"
    grader = ScriptGrader(
        argv=("true",), placement=GraderMachine(environment=NATIVE_ENVIRONMENT), reward=ExitCodeReward()
    )
    tasks = [
        text_task(f"fixture:{index}", prompt=prompt).model_copy(update={"grader": grader})
        for index, prompt in ((4, "start!?"), (1, "middle " * 200), (3, "end!?"))
    ]
    write_selection_panel(selection, tasks, "fixture", "2026.11.19.1", lambda _: {"family": "math"})
    rows = pq.read_table(path).append_column(FAMILY_COLUMN, pa.array(["math"] * len(tasks)))
    pq.write_table(rows, path)
    cache = PreparedTaskCacheSource(str(path), "fixture@2026.11.19.1", str(tmp_path / "prepared"))
    return cache, configs


@pytest.mark.parametrize("cache_location", ["local", "object"])
def test_prepared_selection_reuses_lengths_across_budgets_and_preserves_rollout_inputs(
    prepared_selection, monkeypatch, cache_location
):
    monkeypatch.setattr("skyrl_train.dataset.dataset.PARALLEL_DATASET_MIN_ROWS", 2)
    monkeypatch.setattr("skyrl_train.dataset.prepared_tasks.PREPARED_TASK_BATCH_SIZE", 1)
    cache, configs = prepared_selection
    if cache_location == "object":
        cache = replace(cache, cache_dir=f"memory://{Path(cache.path).parent.parent.name}/prepared")
    full = SourceTaskDataset(
        [cache.path], _selection_tokenizer(), 1000, environment_configs=configs, prepared_cache=cache, num_workers=2
    )
    reference = SourceTaskDataset([cache.path], _selection_tokenizer(), 100, environment_configs=configs, num_workers=2)

    def unavailable(*_args, **_kwargs):
        pytest.fail("A prepared cache hit should not render or tokenize prompts")

    monkeypatch.setattr(TokenizersBackend, "apply_chat_template", unavailable)
    warm = SourceTaskDataset(
        [cache.path], _selection_tokenizer(), 100, environment_configs=configs, prepared_cache=cache
    )

    assert [full.label(i) for i in range(len(full))] == ["fixture:4", "fixture:1", "fixture:3"]
    assert [warm.label(i) for i in range(len(warm))] == ["fixture:4", "fixture:3"]
    assert warm.prompts_dropped_too_long == 1
    assert [warm[i] for i in range(len(warm))] == [reference[i] for i in range(len(reference))]
    assert warm.collate_fn([warm[0], warm[1]]) == reference.collate_fn([reference[0], reference[1]])


@pytest.mark.parametrize("damage", ["truncated", "invalid last batch"])
def test_prepared_selection_rebuilds_a_damaged_cache(prepared_selection, monkeypatch, damage):
    monkeypatch.setattr("skyrl_train.dataset.prepared_tasks.PREPARED_TASK_BATCH_SIZE", 1)
    cache, configs = prepared_selection
    original = SourceTaskDataset(
        [cache.path], _selection_tokenizer(), 100, environment_configs=configs, prepared_cache=cache
    )
    (cache_file,) = tuple(Path(cache.cache_dir).glob("*.parquet"))
    if damage == "truncated":
        cache_file.write_bytes(b"PAR1")
    else:
        table = pq.read_table(cache_file)
        lengths = table[PROMPT_LENGTH_COLUMN].to_pylist()
        lengths[-1] = None
        table = table.set_column(
            table.schema.get_field_index(PROMPT_LENGTH_COLUMN), PROMPT_LENGTH_COLUMN, pa.array(lengths, type=pa.int64())
        )
        pq.write_table(table, cache_file, row_group_size=1)
    recovered = SourceTaskDataset(
        [cache.path], _selection_tokenizer(), 100, environment_configs=configs, prepared_cache=cache
    )
    assert [recovered[i] for i in range(len(recovered))] == [original[i] for i in range(len(original))]
    assert pq.read_table(cache_file)[PROMPT_LENGTH_COLUMN].null_count == 0


def test_prepared_selection_recomputes_lengths_when_installed_lowering_changes(
    prepared_selection, tmp_path, monkeypatch
):
    cache, configs = prepared_selection
    installed_source = tmp_path / "task_inputs.py"
    installed_source.write_bytes(Path(task_inputs.__file__).read_bytes())
    monkeypatch.setattr(task_inputs, "__spec__", spec_from_file_location(task_inputs.__name__, installed_source))
    original = SourceTaskDataset(
        [cache.path], _selection_tokenizer(), 100, environment_configs=configs, prepared_cache=cache
    )

    with installed_source.open("a") as stream:
        stream.write("\n# Another installed revision of task lowering.\n")
    rebuilt = SourceTaskDataset(
        [cache.path], _selection_tokenizer(), 100, environment_configs=configs, prepared_cache=cache
    )

    assert [rebuilt[i] for i in range(len(rebuilt))] == [original[i] for i in range(len(original))]
    assert len(tuple(Path(cache.cache_dir).glob("*.parquet"))) == 2


@pytest.mark.parametrize("changed", ["backend", "template", "template kwargs", "task settings", "provenance"])
def test_prepared_selection_invalidates_when_preparation_inputs_change(prepared_selection, changed):
    cache, configs = prepared_selection
    original_tokenizer = _selection_tokenizer()
    original = SourceTaskDataset(
        [cache.path], original_tokenizer, 1000, environment_configs=configs, prepared_cache=cache
    )
    limit = original.prompt_token_count(original_tokenizer, original.dataframe[0], "prompt")
    if changed == "provenance":
        path = Path(cache.path).with_name("task_manifest.parquet")
        rows = pq.read_table(path).to_pylist()
        rows[0]["family"] = "other"
        pq.write_table(pa.Table.from_pylist(rows), path)
        with pytest.raises(ValueError, match="training row strata differ from selection manifest"):
            SourceTaskDataset(
                [cache.path], _selection_tokenizer(), 1000, environment_configs=configs, prepared_cache=cache
            )
        return

    tokenizer = _selection_tokenizer(
        split_punctuation=changed == "backend", prefix="different " if changed == "template" else ""
    )
    if changed == "task settings":
        configs["task_spec"]["session"]["max_turns"] = 2
        limit = 1000
    kwargs = {"extra": "additional words"} if changed == "template kwargs" else {}
    prepared = SourceTaskDataset(
        [cache.path], tokenizer, limit, environment_configs=configs, chat_template_kwargs=kwargs, prepared_cache=cache
    )
    reference = SourceTaskDataset(
        [cache.path], tokenizer, limit, environment_configs=configs, chat_template_kwargs=kwargs
    )
    assert len(prepared) == (3 if changed == "task settings" else 0)
    assert [prepared[i] for i in range(len(prepared))] == [reference[i] for i in range(len(reference))]
    assert len(tuple(Path(cache.cache_dir).glob("*.parquet"))) == 2


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

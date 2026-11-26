import errno
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
from skyrlrecipe.declared import PromptAcquisition
from skyrl_train import task_inputs
from skyrl_train.config.rollout_validation import FAMILY_COLUMN, GROUPS_PER_FAMILY_KEY
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.prepared_tasks import PROMPT_LENGTH_COLUMN
from skyrl_train.dataset.tasks import TASK_SPEC_COLUMN, SourceTaskDataset, TaskDataset, source_row_task, write_tasks
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.rollouts.context import TrainingContext, prompt_order_from_config
from skyrl_train.rollouts.prompt_order import JudgedGroup
from skyrl_train.task_inputs import task_messages
from taskcompendium.models import ExitCodeReward, GraderMachine, ScriptGrader
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from tests.cpu.task_specs import NATIVE_ENVIRONMENT, text_task, write_selection_panel


class _StubTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt):
        return messages if isinstance(messages, str) else "".join(message["content"] for message in messages)


def _update_prepared_prompt(dataset, row, update):
    dataframe = dataset.dataframe.with_format(None)
    prompts = list(dataframe["prompt"])
    prompts[row] = json.dumps(update(json.loads(prompts[row])))
    dataset.dataframe = dataframe.remove_columns("prompt").add_column("prompt", prompts)


@pytest.mark.parametrize(
    "kind",
    ["prompt-thompson-contextual", "prompt-thompson-within-source-cost-aware", "prompt-thompson-within-source-pass-cost-aware"],
)
def test_prompt_thompson_dispatches_prepared_training_rows_and_rejects_evaluation_overlap(tmp_path, kind):
    config = get_default_config()
    config.trainer.train_batch_size = 3
    config.generator.n_samples_per_prompt = 2
    config.data.sampling.kind = kind
    configs = OmegaConf.to_container(config.environment.task_sessions)

    def dataset(name, tasks):
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.table({TASK_SPEC_COLUMN: [task.model_dump_json() for task in tasks]}), path)
        return SourceTaskDataset([str(path)], _StubTokenizer(), 100, environment_configs=configs)

    tasks = [text_task(f"math-{index}", prompt=f"Compute {index} plus one.", source="math") for index in range(3)]
    training = dataset("train", [text_task("too-long", prompt="x" * 300), *tasks])
    validation = dataset("validation", [text_task("validation", prompt="Find a quotient.", source="math")])
    order = prompt_order_from_config(config, training, held_out=[validation])
    assert {training.label(order.next_index()) for _ in range(3)} == {task.id for task in tasks}

    private_changes = [
        task.model_copy(update={"grader": task.grader.model_copy(update={"reason": "Different private grader text"})})
        for task in tasks
    ]
    revised_order = prompt_order_from_config(config, dataset("revised", private_changes), held_out=[validation])
    revised_order.load_state_dict(order.state_dict())
    assert [revised_order.next_index() for _ in range(6)] == [order.next_index() for _ in range(6)]

    def text_blocks(messages):
        for message in messages:
            content = message["content"]
            message["content"] = [{"type": "text", "text": part} for part in (content[:3], content[3:])]
        return messages

    duplicate = text_task("another-task-id", prompt="Compute 1 plus one.", source="math")
    evaluation_duplicate = dataset("duplicate", [duplicate])
    _update_prepared_prompt(evaluation_duplicate, 0, text_blocks)
    with pytest.raises(ValueError, match="held-out public prompt"):
        prompt_order_from_config(config, training, held_out=[validation, evaluation_duplicate])

    training_duplicates = dataset("training-duplicates", [*tasks, duplicate])
    _update_prepared_prompt(training_duplicates, len(tasks), text_blocks)
    with pytest.raises(ValueError, match="distinct training prompts"):
        prompt_order_from_config(config, training_duplicates, held_out=[validation])

    config.data.sampling.reversion_mass = 1.0
    with pytest.raises(ValueError, match=r"data.sampling.reversion_mass.*Set it to 0"):
        prompt_order_from_config(config, training, held_out=[validation])


@pytest.mark.parametrize("kind", list(PromptAcquisition))
def test_prompt_thompson_checks_unscheduled_primary_panel(tmp_path, monkeypatch, kind):
    config = get_default_config()
    config.data.sampling.kind = kind.value
    config.trainer.eval_interval = 0
    config.trainer.train_batch_size = 1
    config.generator.n_samples_per_prompt = 2
    training, held_out = (tmp_path / name for name in ("training.parquet", "held-out.parquet"))
    for path, prompts in (
        (training, ["Compute one plus two.", "List three colors."]),
        (held_out, ["Compute one plus two."]),
    ):
        tasks = [text_task(f"{path.stem}-{index}", prompt=prompt) for index, prompt in enumerate(prompts)]
        pq.write_table(pa.table({TASK_SPEC_COLUMN: [task.model_dump_json() for task in tasks]}), path)
    config.data.train_data = [str(training)]
    config.data.val_data = [str(held_out)]
    experiment = BasePPOExp.__new__(BasePPOExp)
    experiment.cfg = config
    experiment.evaluation_profiles = {}
    experiment.tokenizer = _StubTokenizer()

    monkeypatch.setattr(
        "skyrl_train.rollouts.context.start_rollout_buffer",
        lambda _: pytest.fail("A leaked validation prompt reached rollout actor startup"),
    )
    eval_dataset = experiment.get_eval_dataset()
    assert eval_dataset is None
    with pytest.raises(ValueError, match="held-out public prompt"):
        experiment.get_trainer(
            cfg=config,
            tracker=None,
            tokenizer=experiment.tokenizer,
            train_dataset=experiment.get_train_dataset(),
            eval_dataset=eval_dataset,
            inference_engine_client=None,
            trajectory_runner=None,
            colocate_pg=None,
        )


def test_prompt_thompson_uses_public_tool_calls_without_transport_ids(tmp_path):
    config = get_default_config()
    config.trainer.train_batch_size = 2
    config.generator.n_samples_per_prompt = 2
    config.data.sampling.kind = "prompt-thompson-contextual"
    sessions = OmegaConf.to_container(config.environment.task_sessions)

    def dataset(name, task_ids):
        path = tmp_path / f"{name}.parquet"
        tasks = [text_task(uid, prompt="Check the calendar.") for uid in task_ids]
        pq.write_table(pa.table({TASK_SPEC_COLUMN: [task.model_dump_json() for task in tasks]}), path)
        return SourceTaskDataset([str(path)], _StubTokenizer(), 100, environment_configs=sessions)

    def tool_call(call_id, arguments):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "calendar_lookup", "arguments": arguments}}
            ],
        }

    training = dataset("train", ["first", "second"])
    _update_prepared_prompt(training, 0, lambda messages: [*messages, tool_call("call-a", '{"day":1,"month":6}')])
    _update_prepared_prompt(training, 1, lambda messages: [*messages, tool_call("call-b", '{"day":2,"month":6}')])
    order = prompt_order_from_config(config, training)
    assert {training.label(order.next_index()) for _ in range(2)} == {"first", "second"}

    held_out = dataset("held-out", ["third"])
    _update_prepared_prompt(held_out, 0, lambda messages: [*messages, tool_call("call-c", '{"month": 6, "day": 1}')])
    with pytest.raises(ValueError, match="held-out public prompt"):
        prompt_order_from_config(config, training, held_out=[held_out])


@pytest.mark.parametrize("kind,epsilon", [("adaptive-source", 0.2), ("adaptive-source-thompson", 0.05)])
def test_adaptive_source_tracks_verified_group_spread_after_filtering_and_resume(tmp_path, kind, epsilon):
    tasks = [
        text_task("too-long", prompt="x" * 300, source="excluded"),
        *(text_task(f"math-{i}", source="math") for i in range(5)),
        *(text_task(f"logic-{i}", source="logic") for i in range(8)),
    ]
    path = tmp_path / "train.parquet"
    pq.write_table(pa.table({TASK_SPEC_COLUMN: [task.model_dump_json() for task in tasks]}), path)
    config = get_default_config()
    config.trainer.train_batch_size = 4
    config.generator.n_samples_per_prompt = 2
    config.data.sampling.kind = kind
    config.data.sampling.epsilon = epsilon
    config.data.sampling.seed = 61
    OmegaConf.update(config, "data.sampling.domain_weights", {"math": 1, "logic": 1}, force_add=True)
    dataset = SourceTaskDataset(
        [str(path)], _StubTokenizer(), 100, environment_configs=OmegaConf.to_container(config.environment.task_sessions)
    )
    order = prompt_order_from_config(config, dataset)
    groups = [
        # Optimization rewards credit answer formatting on math, which has no verifier spread.
        JudgedGroup("0", (0.0, 0.8), (1.0, 1.0)),
        JudgedGroup("5", (0.0, 0.0), (0.5, 0.9)),
        JudgedGroup("6", (0.0, 0.5), (None, 0.5)),
    ]
    metrics = order.observe(groups)
    if kind == "adaptive-source":
        # Beta(1,1) plus one observation yields means of 1/3 and 2/3.
        assert metrics["curriculum/math/weight"] == pytest.approx(0.1 + 0.8 / 3)
    assert metrics["curriculum/logic/graded_groups"] == 1
    assert metrics["curriculum/logic/groups"] == 2
    # With the same evidence and random seed, an explicit math prior must shift the mix.
    OmegaConf.update(config, "data.sampling.domain_weights", {"math": 9, "logic": 1}, force_add=True)
    weighted = prompt_order_from_config(config, dataset).observe(groups)
    assert weighted["curriculum/math/weight"] > metrics["curriculum/math/weight"]
    OmegaConf.update(config, "data.sampling.domain_weights", {"math": 1, "logic": 1}, force_add=True)
    first = order.next_index()
    resumed = prompt_order_from_config(config, dataset)
    resumed.load_state_dict(order.state_dict())
    tail = [order.next_index() for _ in range(3)]
    assert tail == [resumed.next_index() for _ in range(3)]
    assert len({first, *tail}) == 4
    # Cross window boundaries with the checkpointed weights, then refresh both samplers.
    assert [order.next_index() for _ in range(40)] == [resumed.next_index() for _ in range(40)]
    refreshed = order.observe(groups)
    assert resumed.observe(groups) == refreshed
    draws = [order.next_index() for _ in range(400)]
    assert draws == [resumed.next_index() for _ in range(400)]
    # Both rules should favor the mixed logic verifier after repeated evidence.
    for _ in range(40):
        order.observe(groups)
    draws = [order.next_index() for _ in range(400)]
    assert sum(index >= 5 for index in draws) > 300
    assert all(len(set(draws[pos : pos + 4])) == 4 for pos in range(0, len(draws), 4))


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
        (["math"] * 3 + ["agentic"] * 5, "requires 3 distinct prompts for family 'agentic'"),
    ],
    ids=["missing-column", "null-before-named", "no-spare-after-filtering"],
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
        text_task(f"fixture:{index}", prompt=prompt, source=source).model_copy(update={"grader": grader})
        for index, prompt, source in ((4, "start!?", "letters"), (1, "middle " * 200, "letters"), (3, "end!?", "words"))
    ]
    families = {task.id: "strings" if task.source.dataset == "letters" else "tokens" for task in tasks}
    write_selection_panel(selection, tasks, "fixture", "2026.11.19.1", lambda task: {"family": families[task.id]})
    rows = pq.read_table(path).append_column(FAMILY_COLUMN, pa.array([families[task.id] for task in tasks]))
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
    prepared = warm.dataframe.with_format(None)
    assert list(prepared[FAMILY_COLUMN]) == ["strings", "tokens"]
    assert list(prepared["data_source"]) == ["letters", "words"]
    assert warm.prompts_dropped_too_long == 1
    assert [warm[i] for i in range(len(warm))] == [reference[i] for i in range(len(reference))]
    assert warm.collate_fn([warm[0], warm[1]]) == reference.collate_fn([reference[0], reference[1]])


def test_qwen_prepared_cache_reuses_only_matching_thinking_mode(prepared_selection, monkeypatch):
    cache, configs = prepared_selection
    tokenizer = Qwen2Tokenizer(
        chat_template="{% for m in messages %}{{ m.content }}{% endfor %}{% if enable_thinking %}<think>{% endif %}"
    )

    def dataset(thinking):
        return SourceTaskDataset(
            [cache.path],
            tokenizer,
            1000,
            environment_configs=configs,
            prepared_cache=cache,
            chat_template_kwargs={"enable_thinking": thinking},
        )

    first = dataset(False)
    dataset(True)
    assert len(list(Path(cache.cache_dir).glob("*.parquet"))) == 2
    monkeypatch.setattr(tokenizer, "apply_chat_template", lambda *_a, **_kw: pytest.fail("warm read retokenized"))
    warm = dataset(False)
    assert [warm[i] for i in range(len(warm))] == [first[i] for i in range(len(first))]


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


def test_prepared_selection_surfaces_driver_scratch_write_failure(prepared_selection, monkeypatch):
    cache, configs = prepared_selection
    SourceTaskDataset([cache.path], _selection_tokenizer(), 100, environment_configs=configs, prepared_cache=cache)

    def no_space(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(pa.ipc, "new_stream", no_space)
    with pytest.raises(OSError, match="No space left on device"):
        SourceTaskDataset([cache.path], _selection_tokenizer(), 100, environment_configs=configs, prepared_cache=cache)


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

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from fray.cluster import ResourceConfig
from levanter.data.sharded_datasource import ShardedDataSource
from levanter.data.text import HfDatasetSourceConfig, TraceChatEvaluationFormat, UrlDatasetSourceConfig
from levanter.models.llama import LlamaConfig
from levanter.tokenizers import load_tokenizer
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.tracker.wandb import WandbConfig
from marin.evaluation import trace_masked_eval as trace_masked_eval_module
from marin.evaluation.trace_masked_eval import (
    DEFAULT_TRACE_MASKED_EVAL_WANDB_PROJECT,
    FirstRowsShardedDataSource,
    TraceMaskedEvalConfig,
    TraceMaskedEvalDatasetConfig,
    TraceMaskedEvalOnPodConfig,
    TraceRowAdapterConfig,
    _binary_auroc,
    _completed_dataset_metrics,
    _contrastive_outcome_summary,
    _grouped_binary_auroc,
    _is_completed_dataset_result,
    _load_or_create_results,
    _record_dataset_result,
    _run_with_retries,
    _source_for_dataset,
    _write_results,
    default_trace_masked_eval,
)

from experiments.chat_templates.llama3pt1_chat_template import LLAMA_3_1_CHAT_TEMPLATE
from experiments.marin_models import MARIN_CHAT_TEMPLATE


class FailingAfterLimitSource(ShardedDataSource[int]):
    def __init__(self):
        self.rows_read = 0

    @property
    def shard_names(self):
        return ["data"]

    def open_shard_at_row(self, shard_name: str, row: int):
        if shard_name != "data":
            raise ValueError(f"Unknown shard {shard_name!r}")

        for value in range(row, 3):
            if value >= 2:
                raise RuntimeError("read past requested rows")
            self.rows_read += 1
            yield value


def test_default_trace_masked_eval_configures_wandb_and_json_trackers():
    step = default_trace_masked_eval(
        name="trace-smoke",
        checkpoint="gs://marin-us-central1/checkpoints/example",
        checkpoint_is_hf=True,
        model=LlamaConfig(
            max_seq_len=16,
            hidden_dim=16,
            intermediate_dim=32,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
        ),
        tokenizer="gpt2",
        datasets={
            "dummy": TraceMaskedEvalDatasetConfig(
                source=HfDatasetSourceConfig(id="dummy/dataset", splits=["train"]),
                split="train",
                max_examples=1,
            )
        },
        resource_config=ResourceConfig.with_cpu(cpu=1),
        wandb_tags=("trace_masked_eval", "unit-test"),
        wandb_group="trace-evals",
    )

    assert isinstance(step.config, TraceMaskedEvalOnPodConfig)
    tracker = step.config.trace_masked_eval_config.trainer.tracker
    assert isinstance(tracker, tuple)
    assert len(tracker) == 2

    wandb_tracker = tracker[0]
    json_tracker = tracker[1]
    assert isinstance(wandb_tracker, WandbConfig)
    assert isinstance(json_tracker, JsonFileTrackerConfig)
    assert wandb_tracker.project == DEFAULT_TRACE_MASKED_EVAL_WANDB_PROJECT
    assert wandb_tracker.name == "trace-smoke"
    assert wandb_tracker.tags == ["trace_masked_eval", "unit-test"]
    assert wandb_tracker.group == "trace-evals"


def test_first_rows_source_does_not_pull_one_extra_row():
    source = FailingAfterLimitSource()
    limited_source = FirstRowsShardedDataSource(source, max_rows=2)

    assert list(limited_source.open_shard("data")) == [0, 1]
    assert source.rows_read == 2


def test_llama_template_preserves_empty_tool_call_messages_and_tool_call_content():
    tokenizer = load_tokenizer("gpt2")
    processor = TraceChatEvaluationFormat(
        messages_field="messages",
        chat_template=LLAMA_3_1_CHAT_TEMPLATE,
        loss_tags=("assistant", "assistant_text", "tool_call", "observation", "final_assistant"),
    ).build_preprocessor(tokenizer)

    result = processor(
        [
            {
                "messages": [
                    {"role": "user", "content": "Please look this up.", "tool_calls": []},
                    {
                        "role": "assistant",
                        "content": "thinking before call",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search",
                                    "arguments": {"query": "marin"},
                                }
                            }
                        ],
                    },
                    {"role": "tool", "content": "tool observation", "tool_calls": []},
                    {"role": "assistant", "content": "done", "tool_calls": []},
                ]
            }
        ]
    )[0]

    rendered = tokenizer.decode(result["input_ids"])
    assert "Please look this up." in rendered
    assert "thinking before call" in rendered
    assert "search" in rendered
    assert "tool observation" in rendered
    assert "done" in rendered

    masks = result["trace_masks"]
    assert masks["assistant"].sum() > masks["tool_call"].sum()
    assert masks["assistant_text"].sum() > 0
    assert masks["assistant_text"].sum() < masks["assistant"].sum()
    assert masks["tool_call"].sum() > 0
    assert masks["observation"].sum() > 0
    assert masks["final_assistant"].sum() > 0


def test_marin_template_supports_multiple_tool_calls():
    tokenizer = load_tokenizer("gpt2")
    processor = TraceChatEvaluationFormat(
        messages_field="messages",
        chat_template=MARIN_CHAT_TEMPLATE,
        loss_tags=("assistant", "assistant_text", "tool_call"),
    ).build_preprocessor(tokenizer)

    result = processor(
        [
            {
                "messages": [
                    {"role": "user", "content": "Use both tools.", "tool_calls": []},
                    {
                        "role": "assistant",
                        "content": "calling tools",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search",
                                    "arguments": {"query": "marin"},
                                }
                            },
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": {"path": "README.md"},
                                }
                            },
                        ],
                    },
                ]
            }
        ]
    )[0]

    rendered = tokenizer.decode(result["input_ids"])
    assert "calling tools" in rendered
    assert "search" in rendered
    assert "read_file" in rendered
    assert result["trace_masks"]["assistant_text"].sum() > 0
    assert result["trace_masks"]["tool_call"].sum() > 0


def test_trace_row_adapter_adds_patch_and_outcome_targets(tmp_path):
    row = {
        "trajectory": [
            {"role": "system", "system_prompt": "You are a coding agent."},
            {"role": "user", "content": [{"type": "text", "text": "Fix the bug."}]},
            {"role": "ai", "text": "I will inspect the repo."},
        ],
        "test_result": {"git_patch": "diff --git a/a.py b/a.py"},
        "resolved": False,
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(row) + "\n")

    tokenizer = load_tokenizer("gpt2")
    trace_format = TraceChatEvaluationFormat(
        messages_field="messages",
        chat_template=LLAMA_3_1_CHAT_TEMPLATE,
        loss_tags=("patch", "outcome"),
    )
    dataset_config = TraceMaskedEvalDatasetConfig(
        source=UrlDatasetSourceConfig(train_urls=[str(path)]),
        split="train",
        trace_format=trace_format,
        row_adapter=TraceRowAdapterConfig(
            input_messages_field="trajectory",
            patch_field="test_result.git_patch",
            outcome_field="resolved",
        ),
    )

    adapted_row = next(iter(_source_for_dataset(dataset_config)))
    assert adapted_row["trace_outcome_label"] == "INCORRECT"
    assert adapted_row["messages"][0]["content"] == "You are a coding agent."
    assert adapted_row["messages"][1]["content"] == "Fix the bug."
    assert adapted_row["messages"][2]["role"] == "assistant"

    processed = trace_format.build_preprocessor(tokenizer)([adapted_row])[0]
    input_ids = processed["input_ids"]
    masks = processed["trace_masks"]
    patch_text = tokenizer.decode(input_ids[masks["patch"] > 0].tolist(), skip_special_tokens=False)
    outcome_text = tokenizer.decode(input_ids[masks["outcome"] > 0].tolist(), skip_special_tokens=False)

    assert "Final Patch:" in patch_text
    assert "diff --git" in patch_text
    assert "INCORRECT" in outcome_text


def test_trace_row_adapter_limits_large_trace_before_derived_targets():
    row = {
        "trajectory": [
            {"role": "system", "content": "system context"},
            {"role": "user", "content": "original task"},
            {"role": "assistant", "content": "assistant " + ("a" * 20)},
            {"role": "tool", "content": "tool " + ("b" * 20)},
            {"role": "assistant", "content": "assistant " + ("c" * 20)},
        ],
        "patch": "diff --git " + ("d" * 20),
        "resolved": True,
    }
    trace_format = TraceChatEvaluationFormat(messages_field="messages", loss_tags=("patch", "outcome"))
    dataset_config = TraceMaskedEvalDatasetConfig(
        source=UrlDatasetSourceConfig(train_urls=[]),
        split="train",
        trace_format=trace_format,
        row_adapter=TraceRowAdapterConfig(
            input_messages_field="trajectory",
            patch_field="patch",
            outcome_field="resolved",
            max_trace_messages=3,
            preserve_initial_trace_messages=1,
            max_message_chars=6,
            max_patch_chars=8,
        ),
    )

    adapted_row = trace_masked_eval_module._adapt_trace_row(row, trace_format, dataset_config.row_adapter)

    assert [message["role"] for message in adapted_row["messages"]] == [
        "system",
        "tool",
        "assistant",
        "assistant",
        "assistant",
    ]
    assert adapted_row["messages"][0]["content"] == "ontext"
    assert adapted_row["messages"][1]["content"] == "bbbbbb"
    assert adapted_row["messages"][2]["content"] == "cccccc"
    assert adapted_row["messages"][3]["content"] == "Final Patch:\ndddddddd"
    assert adapted_row["messages"][4]["content"] == "CORRECT"

    preserve_only = trace_masked_eval_module._limited_trace_messages(
        row["trajectory"],
        TraceRowAdapterConfig(max_trace_messages=1, preserve_initial_trace_messages=1),
    )
    assert preserve_only == [row["trajectory"][0]]


def test_trace_row_adapter_prefix_fraction_omits_patch_before_full_trace():
    row = {
        "trajectory": [
            {"role": "system", "content": "system context"},
            {"role": "user", "content": "original task"},
            {"role": "assistant", "content": "inspect files"},
            {"role": "tool", "content": "ls output"},
        ],
        "patch": "diff --git a/a.py b/a.py",
        "resolved": True,
    }
    trace_format = TraceChatEvaluationFormat(messages_field="messages", loss_tags=("outcome",))
    row_adapter = TraceRowAdapterConfig(
        input_messages_field="trajectory",
        patch_field="patch",
        outcome_field="resolved",
        task_id_field="instance_id",
    )

    adapted_row = trace_masked_eval_module._adapt_trace_row(
        row,
        trace_format,
        row_adapter,
        include_patch=False,
        prefix_fraction=0.5,
    )

    assert [message["content"] for message in adapted_row["messages"]] == [
        "system context",
        "original task",
        "CORRECT",
    ]
    assert adapted_row["trace_task_id"] == ""
    assert adapted_row["trace_record_id"] == ""


def test_source_for_dataset_applies_row_prefix_fraction(tmp_path):
    row = {
        "trajectory": [
            {"role": "system", "content": "system context"},
            {"role": "user", "content": "original task"},
            {"role": "assistant", "content": "inspect files"},
            {"role": "tool", "content": "ls output"},
        ],
        "patch": "diff --git a/a.py b/a.py",
        "resolved": True,
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(row) + "\n")
    trace_format = TraceChatEvaluationFormat(messages_field="messages", loss_tags=("patch", "outcome"))
    dataset_config = TraceMaskedEvalDatasetConfig(
        source=UrlDatasetSourceConfig(train_urls=[str(path)]),
        split="train",
        trace_format=trace_format,
        row_adapter=TraceRowAdapterConfig(
            input_messages_field="trajectory",
            patch_field="patch",
            outcome_field="resolved",
        ),
        row_prefix_fraction=0.5,
    )

    adapted_row = next(iter(trace_masked_eval_module._source_for_dataset(dataset_config)))

    assert [message["content"] for message in adapted_row["messages"]] == [
        "system context",
        "original task",
        "Final Patch:\ndiff --git a/a.py b/a.py",
        "CORRECT",
    ]


def test_source_for_dataset_allows_zero_row_prefix_fraction(tmp_path):
    row = {
        "trajectory": [
            {"role": "system", "content": "system context"},
            {"role": "user", "content": "original task"},
            {"role": "assistant", "content": "inspect files"},
            {"role": "tool", "content": "ls output"},
        ],
        "patch": "diff --git a/a.py b/a.py",
        "resolved": True,
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(row) + "\n")
    trace_format = TraceChatEvaluationFormat(messages_field="messages", loss_tags=("patch", "outcome"))
    dataset_config = TraceMaskedEvalDatasetConfig(
        source=UrlDatasetSourceConfig(train_urls=[str(path)]),
        split="train",
        trace_format=trace_format,
        row_adapter=TraceRowAdapterConfig(
            input_messages_field="trajectory",
            patch_field="patch",
            outcome_field="resolved",
        ),
        row_prefix_fraction=0.0,
    )

    adapted_row = next(iter(trace_masked_eval_module._source_for_dataset(dataset_config)))

    assert [message["content"] for message in adapted_row["messages"]] == [
        "Final Patch:\ndiff --git a/a.py b/a.py",
        "CORRECT",
    ]


def test_contrastive_outcome_row_adds_judge_prompt_without_gold_label():
    row = {
        "trajectory": [
            {"role": "system", "content": "system context"},
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "done"},
        ],
        "patch": "diff --git a/a.py b/a.py",
        "resolved": True,
    }
    trace_format = TraceChatEvaluationFormat(messages_field="messages", loss_tags=("outcome",))
    row_adapter = TraceRowAdapterConfig(
        input_messages_field="trajectory",
        patch_field="patch",
        outcome_field="resolved",
    )

    adapted_row = trace_masked_eval_module._adapt_contrastive_outcome_row(
        row,
        trace_format,
        row_adapter,
        row_adapter.negative_outcome_label,
    )

    messages = adapted_row["messages"]
    assert adapted_row["trace_outcome_label"] == "CORRECT"
    assert adapted_row["trace_outcome_candidate_label"] == "INCORRECT"
    assert [message["role"] for message in messages[-3:]] == ["assistant", "user", "assistant"]
    assert messages[-3]["content"].startswith("Final Patch:")
    assert "predict whether" in messages[-2]["content"]
    assert messages[-1] == {
        "role": "assistant",
        "content": "INCORRECT",
        "loss_tags": ["outcome"],
    }


def test_prepare_contrastive_candidate_scores_candidate_label_tokens():
    tokenizer = load_tokenizer("gpt2")
    row = {
        "trajectory": [{"role": "user", "content": "fix it"}],
        "patch": "diff --git a/a.py b/a.py",
        "resolved": False,
    }
    trace_format = TraceChatEvaluationFormat(
        messages_field="messages",
        chat_template=LLAMA_3_1_CHAT_TEMPLATE,
        loss_tags=("outcome",),
        slice_strategy="right",
    )
    row_adapter = TraceRowAdapterConfig(
        input_messages_field="trajectory",
        patch_field="patch",
        outcome_field="resolved",
    )

    tokens, loss_weight = trace_masked_eval_module._prepare_contrastive_candidate(
        row,
        trace_format,
        row_adapter,
        tokenizer,
        max_eval_length=128,
        candidate_label="INCORRECT",
    )

    assert tokens.shape == (128,)
    assert loss_weight.shape == (128,)
    assert loss_weight.sum() > 0


def test_contrastive_outcome_summary_reports_accuracy_and_auroc():
    auroc, defined = _binary_auroc([2.0, -1.0, 1.0, -0.5], [True, False, True, False])
    assert auroc == 1.0
    assert defined
    same_task_auroc, same_task_defined, groups, groups_with_pairs, pairs, mean_group_auroc = _grouped_binary_auroc(
        [2.0, -1.0, 1.0, -0.5],
        [True, False, True, False],
        ["task-a", "task-a", "task-b", "task-b"],
    )
    assert same_task_auroc == 1.0
    assert same_task_defined
    assert groups == 2
    assert groups_with_pairs == 2
    assert pairs == 2
    assert mean_group_auroc == 1.0

    summary = _contrastive_outcome_summary(
        margins=[2.0, -1.0, 1.0, -0.5],
        normalized_margins=[2.0, -1.0, 1.0, -0.5],
        gold_is_correct=[True, False, True, False],
        group_ids=["task-a", "task-a", "task-b", "task-b"],
        correct_logprobs=[-1.0, -3.0, -2.0, -4.0],
        incorrect_logprobs=[-3.0, -2.0, -3.0, -3.5],
        correct_token_counts=[1.0, 1.0, 1.0, 1.0],
        incorrect_token_counts=[1.0, 1.0, 1.0, 1.0],
    )

    assert summary["accuracy"] == 1.0
    assert summary["auroc"] == 1.0
    assert summary["auroc_defined"] == 1.0
    assert summary["normalized_accuracy"] == 1.0
    assert summary["normalized_auroc"] == 1.0
    assert summary["normalized_auroc_defined"] == 1.0
    assert summary["positive_examples"] == 2.0
    assert summary["negative_examples"] == 2.0
    assert summary["same_task_auroc"] == 1.0
    assert summary["same_task_auroc_defined"] == 1.0
    assert summary["same_task_groups"] == 2.0
    assert summary["same_task_groups_with_pairs"] == 2.0
    assert summary["same_task_pairs"] == 2.0
    assert summary["same_task_normalized_auroc"] == 1.0
    assert summary["same_task_normalized_auroc_defined"] == 1.0


def test_run_with_retries_retries_transient_failures(monkeypatch):
    calls = 0
    sleeps: list[float] = []

    def flaky_operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("temporary failure")
        return "ok"

    monkeypatch.setattr(trace_masked_eval_module.time, "sleep", lambda delay: sleeps.append(delay))

    result = _run_with_retries(
        "flaky dataset",
        flaky_operation,
        max_attempts=3,
        initial_delay=1.0,
        max_delay=1.5,
    )

    assert result == "ok"
    assert calls == 3
    assert sleeps == [1.0, 1.5]


def test_trace_masked_eval_results_checkpoint_supports_resume(tmp_path):
    config = TraceMaskedEvalConfig(
        checkpoint_path="gs://marin-us-central1/checkpoints/example",
        checkpoint_is_hf=True,
        tokenizer="gpt2",
        max_eval_length=32,
        output_path=str(tmp_path),
    )
    dataset_config = TraceMaskedEvalDatasetConfig(
        source=UrlDatasetSourceConfig(train_urls=[str(tmp_path / "traces.jsonl")]),
        split="train",
        max_examples=1,
    )
    metrics = {
        "trace_masked_eval/openhands/outcome/loss": 0.25,
        "trace_masked_eval/openhands/outcome/count": 4.0,
    }

    results = _load_or_create_results(config)
    _record_dataset_result(results, "openhands", dataset_config, metrics)
    _write_results(config.output_path, results)

    loaded = _load_or_create_results(config)
    loaded_datasets = loaded["datasets"]
    assert isinstance(loaded_datasets, dict)
    assert _is_completed_dataset_result(loaded_datasets["openhands"])
    assert loaded["status"] == "partial"
    assert loaded_datasets["openhands"]["metrics"] == metrics
    assert _completed_dataset_metrics(loaded) == metrics


def test_record_dataset_result_preserves_artifacts(tmp_path):
    results = {
        "datasets": {},
        "status": "partial",
        "completed_datasets": 0,
    }
    dataset_config = TraceMaskedEvalDatasetConfig(
        source=UrlDatasetSourceConfig(train_urls=[str(tmp_path / "traces.jsonl")]),
        split="train",
        max_examples=1,
    )

    _record_dataset_result(
        results,
        "openhands",
        dataset_config,
        {"trace_masked_eval/openhands/outcome/loss": 0.25},
        artifacts={"outcome_example_scores": str(tmp_path / "examples.jsonl")},
    )

    dataset_result = results["datasets"]["openhands"]
    assert isinstance(dataset_result, dict)
    assert dataset_result["artifacts"] == {"outcome_example_scores": str(tmp_path / "examples.jsonl")}


def test_write_jsonl_records_writes_output(tmp_path):
    path = trace_masked_eval_module._outcome_example_scores_path(str(tmp_path), "openhands")
    trace_masked_eval_module._write_jsonl_records(
        path,
        [
            {"example_index": 0, "margin": 1.0},
            {"example_index": 1, "margin": -1.0},
        ],
    )

    assert path.endswith("openhands.outcome_examples.jsonl")
    assert (tmp_path / "examples" / "openhands.outcome_examples.jsonl").read_text().splitlines() == [
        '{"example_index": 0, "margin": 1.0}',
        '{"example_index": 1, "margin": -1.0}',
    ]

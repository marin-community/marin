# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from fray.cluster import ResourceConfig
from experiments.chat_templates.llama3pt1_chat_template import LLAMA_3_1_CHAT_TEMPLATE
from levanter.data.sharded_datasource import ShardedDataSource
from levanter.data.text import TraceChatEvaluationFormat
from levanter.data.text import HfDatasetSourceConfig
from levanter.data.text import UrlDatasetSourceConfig
from levanter.models.llama import LlamaConfig
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.tokenizers import load_tokenizer
from marin.evaluation import trace_masked_eval as trace_masked_eval_module
from marin.evaluation.trace_masked_eval import (
    DEFAULT_TRACE_MASKED_EVAL_WANDB_PROJECT,
    FirstRowsShardedDataSource,
    TraceMaskedEvalConfig,
    TraceRowAdapterConfig,
    TraceMaskedEvalDatasetConfig,
    TraceMaskedEvalOnPodConfig,
    _completed_dataset_metrics,
    _is_completed_dataset_result,
    _load_or_create_results,
    _record_dataset_result,
    _run_with_retries,
    _source_for_dataset,
    _write_results,
    default_trace_masked_eval,
)


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
        loss_tags=("assistant", "tool_call", "observation", "final_assistant"),
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
    assert masks["tool_call"].sum() > 0
    assert masks["observation"].sum() > 0
    assert masks["final_assistant"].sum() > 0


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

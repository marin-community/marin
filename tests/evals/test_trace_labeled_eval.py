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
from marin.evaluation.trace_labeled_eval import (
    DEFAULT_TRACE_LABELED_EVAL_WANDB_PROJECT,
    FirstRowsShardedDataSource,
    TraceLabeledEvalDatasetConfig,
    TraceLabeledEvalOnPodConfig,
    TraceRowAdapterConfig,
    _source_for_dataset,
    trace_labeled_eval_step,
)

from experiments.chat_templates.llama3pt1_chat_template import LLAMA_3_1_CHAT_TEMPLATE


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


def test_trace_labeled_eval_step_configures_wandb_and_json_trackers():
    step = trace_labeled_eval_step(
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
            "dummy": TraceLabeledEvalDatasetConfig(
                source=HfDatasetSourceConfig(id="dummy/dataset", splits=["train"]),
                split="train",
                max_examples=1,
            )
        },
        resource_config=ResourceConfig.with_cpu(cpu=1),
        wandb_tags=("trace_labeled_eval", "unit-test"),
        wandb_group="trace-evals",
    )

    assert isinstance(step.config, TraceLabeledEvalOnPodConfig)
    tracker = step.config.trace_labeled_eval_config.trainer.tracker
    assert isinstance(tracker, tuple)
    assert len(tracker) == 2

    wandb_tracker = tracker[0]
    json_tracker = tracker[1]
    assert isinstance(wandb_tracker, WandbConfig)
    assert isinstance(json_tracker, JsonFileTrackerConfig)
    assert wandb_tracker.project == DEFAULT_TRACE_LABELED_EVAL_WANDB_PROJECT
    assert wandb_tracker.name == "trace-smoke"
    assert wandb_tracker.tags == ["trace_labeled_eval", "unit-test"]
    assert wandb_tracker.group == "trace-evals"


def test_first_rows_source_does_not_pull_one_extra_row():
    source = FailingAfterLimitSource()
    limited_source = FirstRowsShardedDataSource(source, max_rows=2)

    assert list(limited_source.open_shard("data")) == [0, 1]
    assert source.rows_read == 2


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
    dataset_config = TraceLabeledEvalDatasetConfig(
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
    labels = processed["loss_labels"]
    patch_text = tokenizer.decode(input_ids[labels == 4].tolist(), skip_special_tokens=False)
    outcome_text = tokenizer.decode(input_ids[labels == 5].tolist(), skip_special_tokens=False)

    assert "Final Patch:" in patch_text
    assert "diff --git" in patch_text
    assert "INCORRECT" in outcome_text

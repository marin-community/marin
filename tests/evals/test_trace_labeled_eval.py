# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from levanter.data.sharded_datasource import FirstRowsShardedDataSource, ShardedDataSource
from levanter.data.text.datasets import UrlDatasetSourceConfig
from levanter.data.text.trace_chat import TraceChatEvaluationFormat
from levanter.tokenizers import load_tokenizer
from marin.evaluation.trace_labeled_eval import (
    DEFAULT_OUTCOME_JUDGE_PROMPT,
    TraceLabeledEvalDatasetConfig,
    TraceRowAdapterConfig,
    _source_for_dataset,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

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


def _write_local_tokenizer(path, corpus: list[str]) -> None:
    tokens = ["[UNK]"]
    seen = set(tokens)
    for text in corpus:
        for token in text.split():
            if token not in seen:
                seen.add(token)
                tokens.append(token)

    tokenizer = Tokenizer(WordLevel(vocab={token: index for index, token in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"unk_token": "[UNK]"}))


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
            {"role": "ai", "text": "I will inspect the repo.", "thinking": "Need to inspect failing tests."},
        ],
        "test_result": {"git_patch": "diff --git a/a.py b/a.py"},
        "resolved": False,
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(row) + "\n")

    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    _write_local_tokenizer(
        tokenizer_path,
        [
            "You are a coding agent.",
            "Fix the bug.",
            "I will inspect the repo.",
            "Need to inspect failing tests.",
            "Final Patch:\ndiff --git a/a.py b/a.py",
            DEFAULT_OUTCOME_JUDGE_PROMPT,
            "INCORRECT",
            "INCORRECT<|eot_id|>",
        ],
    )
    tokenizer = load_tokenizer(str(tokenizer_path))
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
    assert adapted_row["messages"][2]["reasoning_content"] == "Need to inspect failing tests."
    assert adapted_row["messages"][-2] == {
        "role": "user",
        "content": DEFAULT_OUTCOME_JUDGE_PROMPT,
    }
    assert adapted_row["messages"][-1] == {
        "role": "assistant",
        "content": "INCORRECT",
        "loss_tags": ["outcome"],
    }

    processed = trace_format.build_preprocessor(tokenizer)([adapted_row])[0]
    input_ids = processed["input_ids"]
    labels = processed["loss_labels"]
    patch_text = tokenizer.decode(input_ids[labels == 4].tolist(), skip_special_tokens=False)
    outcome_text = tokenizer.decode(input_ids[labels == 5].tolist(), skip_special_tokens=False)

    assert "Final Patch:" in patch_text
    assert "diff --git" in patch_text
    assert "INCORRECT" in outcome_text

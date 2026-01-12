# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from experiments.train_test_overlap.ngram_new.overlap_map import _build_test_index, _process_training_shard

TOKENIZER_NAME = "whitespace_lower"


def _write_jsonl(path, records):
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _build_index(tmp_path, eval_path):
    eval_specs = [{"path": str(eval_path), "eval_dataset": "evalset"}]
    return _build_test_index(
        eval_specs=eval_specs,
        n_values=[15],
        stride=0,
        tokenizer_name=TOKENIZER_NAME,
        eval_text_field="text",
        output_path=str(tmp_path / "out"),
        skip_existing=True,
        track_progress=False,
    )


def test_overlap_map_single_ngram(tmp_path):
    eval_text = "one two three four five six seven eight nine ten " "eleven twelve thirteen fourteen fifteen"
    eval_path = tmp_path / "eval.jsonl"
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(eval_path, [{"id": "eval-1", "text": eval_text}])
    _write_jsonl(train_path, [{"id": "train-1", "text": eval_text}])

    index_path, meta_path, counts = _build_index(tmp_path, eval_path)
    assert counts == {"evalset": 1}

    records = list(
        _process_training_shard(
            iter([{"path": str(train_path)}]),
            index_path=index_path,
            meta_path=meta_path,
            n_values=[15],
            stride=0,
            tokenizer_name=TOKENIZER_NAME,
            text_field="text",
            log_progress=False,
        )
    )

    assert len(records) == 1
    record = records[0]
    expected_span = [[0, len(eval_text)]]
    assert record["eval_dataset"] == "evalset"
    assert record["eval_path"] == str(eval_path)
    assert record["eval_row"] == 0
    assert record["eval_text"] == eval_text
    assert record["eval_instance_id"] == "eval-1"
    assert record["n"] == 15
    assert record["ngram"] == eval_text
    assert record["eval_offsets"] == expected_span
    assert record["train_path"] == str(train_path)
    assert record["train_row"] == 0
    assert record["train_text"] == eval_text
    assert record["train_ngram"] == eval_text
    assert record["train_offsets"] == expected_span
    assert record["train_doc_id"] == "train-1"


def test_overlap_map_multiple_train_offsets(tmp_path):
    eval_text = "one two three four five six seven eight nine ten " "eleven twelve thirteen fourteen fifteen"
    eval_path = tmp_path / "eval.jsonl"
    train_path = tmp_path / "train.jsonl"
    repeated_text = (
        "one two three four five six seven eight nine ten "
        "eleven twelve thirteen fourteen fifteen "
        "one two three four five six seven eight nine ten "
        "eleven twelve thirteen fourteen fifteen"
    )
    _write_jsonl(eval_path, [{"id": "eval-1", "text": eval_text}])
    _write_jsonl(
        train_path,
        [
            {"id": "train-1", "text": eval_text},
            {"id": "train-2", "text": repeated_text},
        ],
    )

    index_path, meta_path, _counts = _build_index(tmp_path, eval_path)
    records = list(
        _process_training_shard(
            iter([{"path": str(train_path)}]),
            index_path=index_path,
            meta_path=meta_path,
            n_values=[15],
            stride=0,
            tokenizer_name=TOKENIZER_NAME,
            text_field="text",
            log_progress=False,
        )
    )

    assert len(records) == 2
    records_by_row = {record["train_row"]: record for record in records}
    first = records_by_row[0]
    second = records_by_row[1]

    expected_first = [[0, len(eval_text)]]
    expected_second = [
        [0, len(eval_text)],
        [len(eval_text) + 1, len(repeated_text)],
    ]

    assert first["train_doc_id"] == "train-1"
    assert first["train_offsets"] == expected_first
    assert second["train_doc_id"] == "train-2"
    assert second["train_offsets"] == expected_second


def test_overlap_map_example_ngram_sequence(tmp_path):
    eval_text = "A[1] A[2] A[3] A[4] A[5] A[6] A[7] A[8] " "A[9] A[10] A[11] A[12] A[13] A[14] A[15]"
    eval_path = tmp_path / "eval.jsonl"
    train_path = tmp_path / "train.jsonl"
    train_text = "prefix A[1] A[2] A[3] A[4] A[5] A[6] A[7] A[8] " "A[9] A[10] A[11] A[12] A[13] A[14] A[15] suffix"
    _write_jsonl(eval_path, [{"id": "eval-1", "text": eval_text}])
    _write_jsonl(train_path, [{"id": "train-1", "text": train_text}])

    index_path, meta_path, _counts = _build_index(tmp_path, eval_path)
    records = list(
        _process_training_shard(
            iter([{"path": str(train_path)}]),
            index_path=index_path,
            meta_path=meta_path,
            n_values=[15],
            stride=0,
            tokenizer_name=TOKENIZER_NAME,
            text_field="text",
            log_progress=False,
        )
    )

    assert len(records) == 1
    record = records[0]
    expected_ngram = "a[1] a[2] a[3] a[4] a[5] a[6] a[7] a[8] " "a[9] a[10] a[11] a[12] a[13] a[14] a[15]"
    eval_start = 0
    eval_end = len(eval_text)
    train_start = train_text.index(eval_text)
    train_end = train_start + len(eval_text)

    assert record["ngram"] == expected_ngram
    assert record["eval_offsets"] == [[eval_start, eval_end]]
    assert record["train_offsets"] == [[train_start, train_end]]

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.testing.tokenizer import stage_gpt2_tokenizer
from levanter.tokenizers import load_tokenizer
from levanter.trainer import TrainerConfig
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_collection import student_row
from experiments.post_training.russell_rsi.teacher_coverage_loader import (
    CoverageLoaderProofConfig,
    run_coverage_loader_proof,
)


def save_collection(rows, directory, pins):
    record = {
        "status": "passed",
        "accepted": [
            {"row": row.example, "row_sha256": compact_json_sha256(row.example), "witness": asdict(row)} for row in rows
        ],
    }
    raw = json.dumps(record, sort_keys=True).encode()
    (directory / "collection.json").write_bytes(raw)
    pins["collection_sha256"] = hashlib.sha256(raw).hexdigest()
    metadata = {
        "rows": 16,
        "passes": 2,
        "batch_size": 8,
        "optimizer_updates": 4,
        "example_exposures": 32,
        "sha256": pins["jsonl_sha256"],
        "collection_sha256": compact_json_sha256(record),
    }
    raw = json.dumps(metadata, sort_keys=True).encode()
    (directory / "dataset.json").write_bytes(raw)
    pins["dataset_sha256"] = hashlib.sha256(raw).hexdigest()


@pytest.fixture
def coverage_loader_inputs(tmp_path):
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    source = Path(__file__).resolve().parents[3] / "lib/levanter/tests"
    tokenizer = load_tokenizer(str(stage_gpt2_tokenizer(source, tokenizer_path)))
    rows = []
    for index in range(16):
        messages = [
            {"role": "system", "content": "SYSTEMMARKER"},
            {"role": "user", "content": "USERMARKER"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "REASONINGMARKER",
                "tool_calls": [
                    {
                        "id": "call-shell",
                        "type": "function",
                        "function": {"name": "shell", "arguments": json.dumps({"command": f"echo example{index}"})},
                    }
                ],
            },
            {"role": "tool", "name": "shell", "tool_call_id": "call-shell", "content": "TOOLMARKER"},
            {"role": "assistant", "content": f"FINALMARKER{index} " + "complete output " * (800 if index == 0 else 3)},
        ]
        rows.append(student_row(messages, {}, tokenizer))
    path = tmp_path / "train.jsonl"
    payload = "".join(json.dumps(row.example, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(payload)
    component = DatasetComponent(
        source=UrlDatasetSourceConfig(train_urls=[str(path)]),
        cache_dir=str(tmp_path / "cache"),
        format=ChatLmDatasetFormat(chat_template=MARIN_CHAT_TEMPLATE, pack=False),
    )
    config = TrainLmConfig(
        data=LmDataConfig(
            tokenizer=str(tokenizer_path),
            components={"teacher": component},
            mixture_block_size=8,
        ),
        trainer=TrainerConfig(train_batch_size=8, num_train_steps=4, seed=13),
        train_seq_len=16384,
    )
    pins = {"jsonl_uri": str(path), "jsonl_sha256": hashlib.sha256(payload).hexdigest(), "source_commit": "synthetic"}
    save_collection(rows, tmp_path, pins)
    return config, tuple(rows), pins, tokenizer


def test_coverage_loader_two_passes_keep_full_assistant_targets(coverage_loader_inputs, tmp_path):
    config, rows, pins, _ = coverage_loader_inputs
    output_path = StoragePath(str(tmp_path / "loader-proof.json"))
    proof = run_coverage_loader_proof(CoverageLoaderProofConfig(config, pins, str(tmp_path), str(output_path)))
    assert json.loads(output_path.read_text()) == proof
    assert proof["example_exposures"] == 32
    assert Counter(proof["row_exposure_counts"].values()) == {2: 16}
    expected = Counter(proof["canonical_row_sha256"])
    assert all(
        Counter(item for batch in proof["batch_row_sha256"][start : start + 2] for item in batch) == expected
        for start in (0, 2)
    )
    assert proof["supervised_tokens"] == 2 * sum(sum(row.assistant_mask[1:]) for row in rows)
    assert run_coverage_loader_proof(CoverageLoaderProofConfig(config, pins, str(tmp_path), str(output_path))) == proof


def test_coverage_loader_stale_cache_cannot_certify_changed_source(coverage_loader_inputs, tmp_path):
    config, rows, pins, _ = coverage_loader_inputs
    run_coverage_loader_proof(CoverageLoaderProofConfig(config, pins, str(tmp_path), str(tmp_path / "first-proof.json")))
    changed = student_row(
        [{"role": "user", "content": "replacement"}, {"role": "assistant", "content": "different target"}],
        {},
        config.data.the_tokenizer,
    )
    changed_rows = (changed, *rows[1:])
    payload = "".join(json.dumps(row.example, sort_keys=True) + "\n" for row in changed_rows).encode()
    Path(pins["jsonl_uri"]).write_bytes(payload)
    changed_pins = {**pins, "jsonl_sha256": hashlib.sha256(payload).hexdigest()}
    save_collection(changed_rows, tmp_path, changed_pins)
    proof_path = StoragePath(str(tmp_path / "changed-proof.json"))
    with pytest.raises(ValueError, match="full-token witness"):
        run_coverage_loader_proof(CoverageLoaderProofConfig(config, changed_pins, str(tmp_path), str(proof_path)))
    assert not proof_path.exists()


@pytest.mark.parametrize("field", ["input_ids", "assistant_mask"])
def test_coverage_loader_rejects_saved_witness_preprocessing_mismatch(coverage_loader_inputs, tmp_path, field):
    config, _, pins, _ = coverage_loader_inputs
    collection = json.loads((tmp_path / "collection.json").read_bytes())
    witness = collection["accepted"][0]["witness"][field]
    witness[10] = witness[10] + 1
    raw = json.dumps(collection, sort_keys=True).encode()
    (tmp_path / "collection.json").write_bytes(raw)
    pins["collection_sha256"] = hashlib.sha256(raw).hexdigest()
    dataset = json.loads((tmp_path / "dataset.json").read_bytes())
    dataset["collection_sha256"] = compact_json_sha256(collection)
    raw = json.dumps(dataset, sort_keys=True).encode()
    (tmp_path / "dataset.json").write_bytes(raw)
    pins["dataset_sha256"] = hashlib.sha256(raw).hexdigest()
    output = tmp_path / "bad-witness-proof.json"
    with pytest.raises(ValueError, match="saved full-token witness"):
        run_coverage_loader_proof(CoverageLoaderProofConfig(config, pins, str(tmp_path), str(output)))
    assert not output.exists()

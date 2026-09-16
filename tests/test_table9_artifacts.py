# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from experiments.domain_phase_mix import evaluate_table9_accuracy as inference


def test_remote_checkpoint_tokenizers_share_offline_staged_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    uri = f"memory://table9-{tmp_path.name}/checkpoint"
    raw = Tokenizer(models.WordLevel({"[UNK]": 0, "x": 1, "[EOS]": 2}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    inference.write_verified(uri + "/tokenizer.json", raw.to_str().encode())
    inference.write_verified(
        uri + "/tokenizer_config.json",
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "[UNK]",
                "eos_token": "[EOS]",
                "pad_token": "[EOS]",
            }
        ).encode(),
    )
    inference.write_verified(uri + "/model.safetensors", b"must not download model weights")
    hf_tokenizer, tokenizer = inference.checkpoint_tokenizers(uri, tmp_path / "cache")
    assert tokenizer.encode("x x", add_special_tokens=False) == hf_tokenizer.encode("x x", add_special_tokens=False)
    assert tokenizer.eos_token_id == hf_tokenizer.eos_token_id == 2
    staged = Path(hf_tokenizer.name_or_path)
    assert (staged / "tokenizer.json").is_file()
    assert not (staged / "model.safetensors").exists()
    assert tokenizer.as_hf_tokenizer().encode("x x", add_special_tokens=False) == [1, 1]


@pytest.fixture
def encode():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "x": 1, "a": 2, "long": 3, "answer": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return lambda text: tokenizer.encode(text, add_special_tokens=False).ids


def test_native_choice_token_counts_preserve_normalized_ranking(encode):
    request = {"context": "x ", "choices": ["a", "long answer"], "gold_index": 1, "doc_id": 0}
    sample = inference.scored_choices([request], [(-2.0, False), (-3.0, False)], encode)[0]
    assert sample["token_counts"] == [1, 2]
    assert sample["metrics"] == {"acc": 0.0, "acc_per_token": 1.0, "acc_per_char": 1.0}
    with pytest.raises(ValueError, match="score count"):
        inference.scored_choices([request], [(-2.0, False)], encode)


@pytest.mark.parametrize("mode", ["choices", "generation"])
def test_memory_probe_fills_context_without_evaluation_documents(encode, mode):
    requests = inference.memory_probe_requests(encode, mode, 8, 8192)
    assert len(requests) == 8
    for request in requests:
        context, continuation = request.args
        assert request.doc == {}
        if mode == "choices":
            assert len(encode(context + continuation)) == 8192
            assert len(encode(context)) == 8191
        else:
            assert len(encode(context)) + continuation["max_gen_toks"] == 8192
            assert continuation["until"] == []


@pytest.mark.parametrize(
    "override",
    [{"zone": "us-east5-a"}, {"tpu_type": "v6e-16"}, {"batch_size": 4}, {"max_length": 4096}],
)
def test_evaluation_rejects_wrong_zone_multihost_or_changed_protocol(override):
    plan = {"tpu_type": "v6e-4", "zone": "us-east5-b", "region": "us-east5", "batch_size": 8, "max_length": 8192}
    with pytest.raises(ValueError):
        inference.evaluation_resources(plan | override)


def test_memory_success_cannot_transfer_to_another_checkpoint_or_slice(tmp_path):
    plan = {"output_root": str(tmp_path), "tpu_type": "v6e-4", "batch_size": 8, "max_length": 8192}
    row = {"name": "checkpoint", "checkpoint_uri": "checkpoint-one"}
    root = inference.result_root(plan, row, "memory_choices", 0)
    marker = inference.memory_probe_identity(plan, row, "choices") | {"count": 8}
    inference.write_verified(root + "/SUCCESS.json", inference.existing.canonical_json(marker))
    assert inference.completed_memory_probe(plan, row, "choices") == marker
    assert inference.completed_memory_probe(plan | {"tpu_type": "v6e-8"}, row, "choices") is None
    assert inference.completed_memory_probe(plan, row | {"checkpoint_uri": "checkpoint-two"}, "choices") is None
    assert inference.completed_memory_probe(plan, row, "generation") is None
    marker["count"] = 2
    inference.write_verified(root + "/SUCCESS.json", inference.existing.canonical_json(marker))
    with pytest.raises(ValueError, match="provenance/count"):
        inference.completed_memory_probe(plan, row, "choices")


def test_canary_does_not_complete_full_evaluation_and_tampering_is_rejected(tmp_path):
    task = "basic_skills_arithmetic"
    row = {"name": "checkpoint"}
    plan = {
        "output_root": str(tmp_path),
        "rows": [row],
        "request_manifest": {"tasks": {task: {"count": 2, "metric": "acc_per_token"}}},
    }
    root = inference.result_root(plan, row, task, 1)
    samples = [{"doc_id": 0, "metrics": {"acc_per_token": 1.0}}]
    payload = gzip.compress(json.dumps(samples).encode())
    artifact = inference.write_verified(root + "/samples.json.gz", payload)
    marker = {
        "row": row,
        "protocol_sha256": inference.digest(inference.protocol(plan)),
        "limit": 1,
        "task": task,
        "count": 1,
        "stage": "scored",
        "artifact": artifact,
    }
    inference.write_verified(root + "/SUCCESS.json", inference.existing.canonical_json(marker))
    assert inference.completed_task(plan, row, task, 1) == marker
    assert inference.completed_task(plan, row, task, 0) is None
    inference.write_verified(root + "/samples.json.gz", payload + b"changed")
    with pytest.raises(ValueError, match="samples changed"):
        inference.completed_task(plan, row, task, 1)

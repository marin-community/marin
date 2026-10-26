# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from marin.drafting import DraftTrainingConfig, draft_serving, prepare_coding_sample


@pytest.mark.parametrize("corrupt", [False, True])
def test_export_roundtrip_serves_original_target_and_checks_weights(tmp_path, corrupt):
    """A serving export works after the worker and its native checkpoints are gone."""
    export = tmp_path / "export"
    export.mkdir()
    weights = b"opaque durable weights"
    (export / "model.safetensors").write_bytes(weights)
    config = DraftTrainingConfig(
        target="s3://bucket/target",
        tokenizer="namespace/tokenizer",
        tokenizer_revision="immutable",
        data="prompts",
        output=str(export),
    )
    (export / "training.json").write_text(json.dumps({"config": asdict(config)}))
    digest = hashlib.sha256(weights).hexdigest()
    (export / "export_manifest.json").write_text(json.dumps({"files": {"model.safetensors": {"sha256": digest}}}))
    if corrupt:
        (export / "model.safetensors").write_bytes(b"interrupted upload")
        with pytest.raises(ValueError, match="Serving draft bytes"):
            with draft_serving(str(export)):
                pytest.fail("Corrupt weights were offered for serving")
        return
    with draft_serving(str(export)) as (model, engine):
        assert model.weights == "s3://bucket/target"
        assert (model.tokenizer, model.tokenizer_revision) == ("namespace/tokenizer", "immutable")
        assert model.max_model_len == 65536
        assert engine.speculative.model.identity == digest
        staged = Path(engine.speculative.model.uri)
        assert (staged / "model.safetensors").read_bytes() == weights
    assert not staged.exists()


def test_coding_sample_ignores_answers_and_corpus_order(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.json"
    rows = [
        {"instruction": instruction, "input": "", "output": "original answer"}
        for instruction in ("Sort a list", "Reverse a string", "Parse JSON", " Sort a list ")
    ]
    corpus.write_text(json.dumps(rows))
    monkeypatch.setattr("marin.drafting.hf_hub_download", lambda *args, **kwargs: str(corpus))
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    prepare_coding_sample(str(first), sample_size=3)
    corpus.write_text(json.dumps([dict(row, output="changed answer") for row in reversed(rows)]))
    prepare_coding_sample(str(second), sample_size=3)
    assert first.read_bytes() == second.read_bytes()
    prompts = [json.loads(line) for line in first.read_text().splitlines()]
    assert len({row["task"] for row in prompts}) == 3
    assert {row["messages"][0]["content"] for row in prompts} == {"Sort a list", "Reverse a string", "Parse JSON"}
    assert all(row["task"] == row["id"] for row in prompts)
    receipt = json.loads(Path(str(first) + ".source.json").read_text())
    assert receipt["sample_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()

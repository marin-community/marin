# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from marin.drafting import DraftTrainingConfig, _publish_checkpoint, prepare_coding_sample, restore_training_run


def test_completed_epoch_survives_object_store_roundtrip(tmp_path):
    """Recovery selects the published epoch despite a later interrupted upload."""
    root = tmp_path / "original"
    checkpoint = root / "checkpoints/0"
    checkpoint.mkdir(parents=True)
    (root / "data").mkdir()
    for name in ("d2t", "t2d"):
        np.save(root / f"data/{name}-4.npy", np.arange(4))
    progress = {"epoch": 0, "local_step": 0, "global_step": 3}
    (checkpoint / "training_state.json").write_text(json.dumps(progress))
    (checkpoint / "config.json").write_text(json.dumps({"draft_vocab_size": 4}))
    (checkpoint / "val_metrics.json").write_text(json.dumps({"loss_epoch": 2.5}))
    # Preserve real serialized optimizer/scheduler state, including scalar tensors.
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    parameter.sum().backward()
    optimizer.step()
    scheduler.step()
    torch.save(optimizer.state_dict(), checkpoint / "optimizer_state_dict.pt")
    torch.save(scheduler.state_dict(), checkpoint / "scheduler_state_dict.pt")
    (checkpoint / "model.safetensors").write_bytes(b"opaque model bytes")
    (root / "checkpoints/checkpoint_best").symlink_to("0", target_is_directory=True)
    remote = tmp_path / "object-store"
    config = DraftTrainingConfig("target", "tokenizer", "revision", "prompts", str(remote), epochs=2)
    record = {"config": asdict(config)}
    _publish_checkpoint(root, config, record, 0)
    # Simulate loss of a worker during publication of the next epoch.
    (remote / "checkpoints/1").mkdir()
    (remote / "checkpoints/1/model.safetensors").write_bytes(b"unfinished upload")
    (remote / "checkpoints/1/val_metrics.json").write_text('{"loss_epoch": 1.0}')
    restored = tmp_path / "new-worker"
    restored_config, restored_record = restore_training_run(str(remote), restored)
    assert restored_config == config
    assert restored_record["checkpoint"]["global_step"] == 3
    assert sorted(path.name for path in (restored / "checkpoints").iterdir()) == ["0", "checkpoint_best"]
    assert (restored / "checkpoints/checkpoint_best").resolve() == restored / "checkpoints/0"
    for original in checkpoint.iterdir():
        assert (restored / "checkpoints/0" / original.name).read_bytes() == original.read_bytes()
    reloaded_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1.0)
    reloaded_optimizer.load_state_dict(torch.load(restored / "checkpoints/0/optimizer_state_dict.pt", weights_only=True))
    assert reloaded_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    torch.testing.assert_close(
        next(iter(reloaded_optimizer.state.values()))["exp_avg"], optimizer.state[parameter]["exp_avg"], rtol=0, atol=0
    )
    for name in ("d2t", "t2d"):
        np.testing.assert_array_equal(np.load(restored / f"data/{name}-4.npy"), np.arange(4))


def test_resume_requires_optimizer_state(tmp_path):
    source = tmp_path / "run"
    checkpoint = source / "checkpoints/0"
    checkpoint.mkdir(parents=True)
    (source / "run.json").write_text(
        json.dumps(
            {
                "config": asdict(DraftTrainingConfig("target", "tokenizer", "revision", "prompts", str(source))),
                "checkpoint": {"epoch": 0},
            }
        )
    )
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"serving weights alone")
    with pytest.raises(ValueError, match=r"optimizer_state_dict\.pt"):
        restore_training_run(str(source), tmp_path / "worker")


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

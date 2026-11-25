# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from rigging.filesystem.storage_path import StoragePath
from rigging.training_journal import TrainingJournal

from experiments.post_training.long_loop.archive import ArchiveInput, seal_archive, verify_archive
from experiments.post_training.long_loop.archive_stage import native_training_inputs


def produce(root, producer, stage, omit_phase=None, local_directory=None):
    journal = TrainingJournal(str(root), "cpu-fixture", stage, producer, local_directory=local_directory)
    phases = ("init", "data_load", "compile", "first_step", "steady_state", "checkpoint_save", "export")
    if stage == "rl":
        phases += ("inference_engine_init_and_compile", "save_checkpoints", "queue_hf_export")
    for name in phases:
        if name == omit_phase:
            continue
        with journal.events.span(name):
            pass
    for step in range(3):
        journal.log({"loss": 1 / (step + 1)}, step=step, commit=False)
        journal.log({"tokens": step * 64}, step=step, commit=True)
    journal.finish(status="completed")
    return journal


def test_archive_requires_every_producer_and_keeps_all_metric_calls(tmp_path):
    outputs = tmp_path / "task-outputs"
    outputs.mkdir()
    sft, rl = tmp_path / "sft", tmp_path / "rl"
    first = produce(sft, "rank-0", "sft", local_directory=str(outputs))
    produce(rl, "trainer", "rl")
    produce(rl / "exports/step-2", "exporter", "rl_export")
    with pytest.raises(ValueError, match="producer coverage"):
        native_training_inputs(str(sft), str(rl), 2, 2)
    produce(sft, "rank-1", "sft")
    inputs, coverage = native_training_inputs(str(sft), str(rl), 2, 2)
    assert coverage["sft"]["verified"] == 2
    sealed = tmp_path / "sealed"
    seal_archive(str(sealed), [ArchiveInput(**item) for item in inputs], {})
    history = sealed / f"objects/telemetry/sft/rank-0/{first.attempt}/history.jsonl"
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    assert [(row["step"], row["metrics"]) for row in rows] == [
        pair for step in range(3) for pair in ((step, {"loss": 1 / (step + 1)}), (step, {"tokens": step * 64}))
    ]
    assert history.read_bytes() == first.history.read_bytes()
    assert first.directory.is_relative_to(outputs)
    history.write_text(history.read_text() + "{}\n")
    with pytest.raises(ValueError, match="verification"):
        verify_archive(str(sealed))


def test_archive_rejects_missing_or_open_native_phase_journal(tmp_path):
    sft, rl = tmp_path / "sft", tmp_path / "rl"
    journal = produce(sft, "rank-0", "sft")
    produce(rl, "trainer", "rl")
    produce(rl / "exports/step-2", "exporter", "rl_export")
    (journal.destination / "phases.jsonl").write_text("")
    with pytest.raises(ValueError, match="Telemetry file changed"):
        native_training_inputs(str(sft), str(rl), 1, 2)


def test_archive_requires_native_export_phase_even_with_valid_history_hashes(tmp_path):
    sft, rl = tmp_path / "sft", tmp_path / "rl"
    produce(sft, "rank-0", "sft", omit_phase="export")
    produce(rl, "trainer", "rl")
    produce(rl / "exports/step-2", "exporter", "rl_export")
    with pytest.raises(ValueError, match="Missing native phases"):
        native_training_inputs(str(sft), str(rl), 1, 2)


def test_interrupted_producer_attempt_blocks_complete_archive(tmp_path):
    sft, rl = tmp_path / "sft", tmp_path / "rl"
    produce(sft, "rank-0", "sft")
    produce(rl, "trainer", "rl")
    produce(rl / "exports/step-2", "exporter", "rl_export")
    TrainingJournal(str(sft), "cpu-fixture", "sft", "rank-0")
    with pytest.raises(ValueError, match="no completion receipt"):
        native_training_inputs(str(sft), str(rl), 1, 2)


def test_declared_phase_gap_retains_verified_history_without_certifying_phase_coverage(tmp_path):
    sft, rl = tmp_path / "sft", tmp_path / "rl"
    produce(sft, "rank-0", "sft")
    journal = produce(rl, "trainer", "rl", omit_phase="first_step")
    produce(rl / "exports/step-2", "exporter", "rl_export")
    with pytest.raises(ValueError, match="Missing native phases"):
        native_training_inputs(str(sft), str(rl), 1, 2)
    inputs, coverage = native_training_inputs(str(sft), str(rl), 1, 2, {"rl": ["first_step"]})
    assert coverage["rl"]["missing_phases"] == ["first_step"]
    assert coverage["sft"]["missing_phases"] == []
    history = next(item for item in inputs if item["name"] == f"telemetry/rl/trainer/{journal.attempt}/history.jsonl")
    assert StoragePath(history["source"]).read_bytes() == journal.history.read_bytes()
    with pytest.raises(ValueError, match="Declared native phase gaps differ"):
        native_training_inputs(str(sft), str(rl), 1, 2, {"rl": ["first_step", "steady_state"]})
    (journal.destination / "history.jsonl").write_text("changed history")
    with pytest.raises(ValueError, match="Telemetry file changed"):
        native_training_inputs(str(sft), str(rl), 1, 2, {"rl": ["first_step"]})

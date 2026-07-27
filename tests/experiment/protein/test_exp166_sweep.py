# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np

from experiments.protein.exp166_sweep import (
    CONTACTS_V1_TOKEN_IDS,
    POINTS,
    Initialization,
    StageCheckpointConfig,
    Trial,
    _stage_checkpoint,
    _tags,
    shuffle_amino_acid_statements,
)


def _document(sequence_statements: list[tuple[int, int]], structure_tokens: list[int]) -> list[int]:
    return [
        CONTACTS_V1_TOKEN_IDS["<contacts-v1>"],
        CONTACTS_V1_TOKEN_IDS["<begin_sequence>"],
        *(token for statement in sequence_statements for token in statement),
        CONTACTS_V1_TOKEN_IDS["<begin_statements>"],
        *structure_tokens,
    ]


def _sections(tokens: np.ndarray) -> list[tuple[list[tuple[int, int]], np.ndarray]]:
    document_id = CONTACTS_V1_TOKEN_IDS["<contacts-v1>"]
    begin_sequence_id = CONTACTS_V1_TOKEN_IDS["<begin_sequence>"]
    begin_statements_id = CONTACTS_V1_TOKEN_IDS["<begin_statements>"]
    starts = np.flatnonzero(tokens == document_id)
    sections = []
    for document_index, start in enumerate(starts):
        stop = int(starts[document_index + 1]) if document_index + 1 < starts.size else tokens.size
        begin_sequence = start + int(np.flatnonzero(tokens[start:stop] == begin_sequence_id)[0])
        begin_statements = start + int(np.flatnonzero(tokens[start:stop] == begin_statements_id)[0])
        sequence = tokens[begin_sequence + 1 : begin_statements].reshape(-1, 2)
        sections.append(([tuple(statement) for statement in sequence.tolist()], tokens[begin_statements:stop]))
    return sections


def test_shuffle_amino_acid_statements_changes_order_and_preserves_document_meaning():
    first_statements = [(143, 86), (144, 87), (145, 88), (146, 89), (3, 143), (4, 146)]
    second_statements = [(201, 98), (202, 99), (203, 100), (204, 101), (3, 201), (4, 204)]
    original = np.asarray(
        [
            *_document(first_statements, [5, 143, 146, 10, 1]),
            *_document(second_statements, [5, 201, 204, 10, 1]),
        ],
        dtype=np.int32,
    )

    augmented, stats = shuffle_amino_acid_statements(original, np.random.default_rng(166))
    another_view, _ = shuffle_amino_acid_statements(original, np.random.default_rng(167))

    assert stats.documents == 2
    assert stats.residue_statements == 8
    assert stats.moved_statements > 0
    assert stats.changed_token_positions > 0
    assert not np.array_equal(augmented, another_view)

    original_sections = _sections(original)
    augmented_sections = _sections(augmented)
    for (original_statements, original_structure), (augmented_statements, augmented_structure) in zip(
        original_sections, augmented_sections, strict=True
    ):
        assert sorted(augmented_statements) == sorted(original_statements)
        assert np.array_equal(augmented_structure, original_structure)


def test_exp117_tags_use_compact_checkpoint_identity():
    for point in POINTS:
        tags = _tags(Trial(point, Initialization.EXP117), "europe-west4", num_train_steps=2)

        assert f"source_checkpoint=exp117/{point.key}" in tags
        assert all(1 <= len(tag) <= 64 for tag in tags)


def test_stage_checkpoint_copies_latest_complete_checkpoint(tmp_path):
    source = tmp_path / "source"
    older = source / "checkpoints" / "step-3"
    latest = source / "checkpoints" / "step-7"
    destination = tmp_path / "destination"
    for checkpoint, step in ((older, 3), (latest, 7)):
        (checkpoint / "d").mkdir(parents=True)
        (checkpoint / "metadata.json").write_text(json.dumps({"step": step, "timestamp": f"2026-07-27T00:00:0{step}"}))
        (checkpoint / "manifest.ocdbt").write_bytes(f"manifest-{step}".encode())
        (checkpoint / "d" / "tensor").write_bytes(f"tensor-{step}".encode())

    config = StageCheckpointConfig(
        source_run_path=str(source),
        output_path=str(destination),
        transfer_budget_gb=1,
    )
    _stage_checkpoint(config)

    copied = destination / "checkpoints" / "step-7"
    assert (copied / "metadata.json").read_bytes() == (latest / "metadata.json").read_bytes()
    assert (copied / "manifest.ocdbt").read_bytes() == b"manifest-7"
    assert (copied / "d" / "tensor").read_bytes() == b"tensor-7"
    assert not (destination / "checkpoints" / "step-3").exists()

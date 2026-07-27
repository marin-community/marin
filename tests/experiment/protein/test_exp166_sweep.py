# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np
import pytest
from marin.rl.placement import marin_prefix_for_region

from experiments.protein.exp166_sweep import (
    CONTACTS_V1_TOKEN_IDS,
    EXP117_VERSION,
    POINTS,
    SEED_NAMESPACE,
    TRIALS,
    Initialization,
    SeededCheckpointConfig,
    Trial,
    _tags,
    _verify_seeded_checkpoint,
    exp117_checkpoint,
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


def test_top_six_exp117_points_generate_twelve_logical_trials():
    assert len(POINTS) == 6
    assert len(TRIALS) == 12
    assert [point.exp117_loss for point in POINTS] == sorted(point.exp117_loss for point in POINTS)
    assert [point.batch_size for point in POINTS].count(64) == 3
    assert [point.batch_size for point in POINTS].count(128) == 3
    assert [point.exp117_run for point in POINTS[-2:]] == [
        "prot-exp117-cv1-s02-1_5b-e8-lr3p162e-4-wd1p6-bs128-us-east1",
        "prot-exp117-cv1-s02-1_5b-e8-lr1e-3-wd0p2-bs64-us-east5",
    ]


def _write_checkpoint(root, step: int) -> None:
    checkpoint = root / "checkpoints" / f"step-{step}"
    (checkpoint / "d").mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(json.dumps({"step": step, "timestamp": f"2026-07-27T00:00:0{step}"}))
    (checkpoint / "manifest.ocdbt").write_bytes(f"manifest-{step}".encode())
    (checkpoint / "d" / "tensor").write_bytes(f"tensor-{step}".encode())


def test_verify_seeded_checkpoint_accepts_a_region_local_seed(tmp_path):
    seed = tmp_path / "seed"
    _write_checkpoint(seed, 7)

    _verify_seeded_checkpoint(
        SeededCheckpointConfig(checkpoint_root=str(seed), region="us-east5", region_prefix=str(tmp_path))
    )


def test_verify_seeded_checkpoint_rejects_an_unseeded_region(tmp_path):
    unseeded = tmp_path / "unseeded"
    (unseeded / "checkpoints").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        _verify_seeded_checkpoint(
            SeededCheckpointConfig(checkpoint_root=str(unseeded), region="us-east5", region_prefix=str(tmp_path))
        )


def test_exp117_seed_records_provenance_without_pinning_a_region():
    """The seed name carries its exp117 source; the region comes from the executor prefix."""
    point = next(p for p in POINTS if p.exp117_region == "europe-west4")
    step = exp117_checkpoint(point, "us-west4")

    assert step.name == f"{SEED_NAMESPACE}/{point.exp117_run}"
    assert step.version == EXP117_VERSION
    # Namespaced away from the real exp117 run directory, so seeding a point into
    # its own origin region cannot overwrite the source it was copied from.
    assert step.name.startswith(f"{SEED_NAMESPACE}/")
    assert step.name != f"checkpoints/protein/{point.exp117_run}"


def test_every_point_seeds_a_distinct_artifact():
    names = {exp117_checkpoint(point, "us-east5").name for point in POINTS}

    assert len(names) == len(POINTS)


def test_verify_seeded_checkpoint_refuses_a_seed_outside_the_execution_region(tmp_path):
    """A resolution bug must fail loudly, never silently read another region."""
    seed = tmp_path / "elsewhere"
    _write_checkpoint(seed, 7)

    with pytest.raises(RuntimeError, match="outside region"):
        _verify_seeded_checkpoint(
            SeededCheckpointConfig(checkpoint_root=str(seed), region="us-east5", region_prefix="gs://marin-us-east5")
        )


def test_exp117_seed_is_pinned_to_the_execution_region_not_the_origin():
    point = next(p for p in POINTS if p.exp117_region == "europe-west4")

    config = exp117_checkpoint(point, "us-west4").build_config(
        SimpleNamespace(output_path="gs://marin-us-west4/x", is_fingerprint=False)
    )

    assert config.region == "us-west4"
    assert config.region_prefix == marin_prefix_for_region("us-west4")
    assert marin_prefix_for_region(point.exp117_region) not in config.region_prefix

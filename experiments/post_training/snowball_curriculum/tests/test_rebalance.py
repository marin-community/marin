# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A frozen ranked prefix gives the same smaller sample as its original population."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.test_support import arithmetic_task
from taskcompendium.models import ConversationInput, TextMessage
from taskcompendium.pipeline.sampling import seeded_order

from experiments.post_training.snowball_curriculum.bundle import file_sha256
from experiments.post_training.snowball_curriculum.native import select_native_closure_tasks
from experiments.post_training.snowball_curriculum.rebalance import retained_contracts, write_rebalanced_pool
from experiments.post_training.snowball_curriculum.selection import SourceSpec
from experiments.post_training.snowball_curriculum.tests.closure_rows import closure_row
from experiments.post_training.snowball_curriculum.v0 import run


def test_retained_contracts_reproduces_full_population_sample_from_cached_prefix(tmp_path):
    population = [
        {"source_id": source, "difficulty": difficulty, "contract_id": f"{source}-{difficulty}-{index}"}
        for source, difficulty, count in [("a", "easy", 12), ("a", "hard", 8), ("b", "unknown", 10)]
        for index in range(count)
    ]
    ordered = sorted(population, key=lambda row: seeded_order(row["contract_id"], 17))
    old_quotas = {("a", "easy"): 3, ("a", "hard"): 3, ("b", "unknown"): 4}
    retained = []
    for key, count in old_quotas.items():
        retained.extend([row for row in ordered if (row["source_id"], row["difficulty"]) == key][:count])
    retained.sort(key=lambda row: seeded_order(row["contract_id"], 17))
    pq.write_table(pa.Table.from_pylist(retained), tmp_path / "train.parquet")
    (tmp_path / "selection.json").write_text(
        json.dumps(
            {
                "seed": 17,
                "eligible_difficulty_counts": {"a": {"easy": 12, "hard": 8}, "b": {"unknown": 10}},
                "difficulty_quotas": {"a": {"easy": 3, "hard": 3}, "b": {"unknown": 4}},
            }
        )
    )
    selected, quotas = retained_contracts(tmp_path, {"a": 4, "b": 3})
    assert quotas == {"a": {"easy": 2, "hard": 2}, "b": {"unknown": 3}}
    expected = set()
    for source, difficulty, count in [("a", "easy", 2), ("a", "hard", 2), ("b", "unknown", 3)]:
        candidates = [
            row["contract_id"] for row in ordered if row["source_id"] == source and row["difficulty"] == difficulty
        ]
        expected.update(candidates[:count])
    assert selected == expected


def test_v0_pool_is_a_direct_rebalance_input_with_complete_nested_inventory(tmp_path):
    sources = []
    for family, count in (("calendar", 1), ("reasoning_gym", 3), ("structural", 1)):
        path = tmp_path / f"{family}.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "task_json": (
                            arithmetic_task(
                                id=f"{family}-{index}",
                                context=ConversationInput(
                                    events=(TextMessage(role="user", content=f"{family} question {index}"),)
                                ),
                            ).model_dump_json()
                        )
                    }
                )
                + "\n"
                for index in range(count)
            )
        )
        sources.append(SourceSpec(family, "train", (str(path),), "admitted", catalog_snapshot_id="frozen"))
    previous = tmp_path / "previous"
    recipe = {
        "extensions": {source.source_id: source.source_id for source in sources},
        "catalog_uri": "catalog",
        "catalog_snapshot_id": "frozen",
        "migration_receipt": "receipt",
    }
    run(sources=sources, recipe=recipe, output=str(previous), target=5, seed=7)
    native_train, native_validation = tmp_path / "native.parquet", tmp_path / "heldout.parquet"
    pq.write_table(pa.Table.from_pylist([closure_row("new-native")]), native_train)
    pq.write_table(pa.Table.from_pylist([closure_row("heldout", tokens=(99,))]), native_validation)
    native = select_native_closure_tasks(
        str(native_train),
        str(native_validation),
        target=1,
        seed=7,
        expected_train_sha256=file_sha256(StoragePath(str(native_train))),
        expected_validation_sha256=file_sha256(StoragePath(str(native_validation))),
    )
    plan = {
        "seed": 7,
        "native_rows": 1,
        "version": "fixture-2",
        "previous_version": "fixture-1",
        "family_quotas": {"calendar": 1, "reasoning_gym": 2, "structural": 2},
        "difficulty_policy": "unknown",
        "selection_policy": "fixture",
    }
    target = tmp_path / "target"
    write_rebalanced_pool(
        previous,
        target,
        plan,
        native,
        native_train=native_train,
        native_validation=native_validation,
        expected_manifest_sha256=file_sha256(StoragePath(str(previous / "artifact-manifest.json"))),
    )
    json_rows = [json.loads(line) for line in (target / "tasks.jsonl").read_text().splitlines()]
    assert pq.read_table(target / "train.parquet").to_pylist() == json_rows
    assert len(json_rows) == 5
    originals = {row["contract_id"]: row["task_json"] for row in pq.read_table(previous / "train.parquet").to_pylist()}
    for row in json_rows:
        if row["contract_kind"] == "task_spec":
            assert row["task_json"] == originals[row["contract_id"]]
        else:
            assert row["lowered_task_spec"] == closure_row("new-native")["lowered_task_spec"]
    entries = json.loads((target / "artifact-manifest.json").read_text())["files"]
    assert "provenance/predecessor-recipe.json" in {entry["path"] for entry in entries}
    assert "train.parquet" in {entry["path"] for entry in entries}
    assert all(file_sha256(StoragePath(str(target / entry["path"]))) == entry["sha256"] for entry in entries)

    (previous / "recipe.json").write_text("{}")
    with pytest.raises(ValueError, match="payload differs"):
        write_rebalanced_pool(
            previous,
            tmp_path / "corrupted",
            plan,
            native,
            native_train=native_train,
            native_validation=native_validation,
            expected_manifest_sha256=file_sha256(StoragePath(str(previous / "artifact-manifest.json"))),
        )
    assert not (tmp_path / "corrupted").exists()

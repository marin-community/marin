# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native imports keep token replay and historical closure scoring intact."""

import hashlib
import json
from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.snowball_curriculum.native import (
    native_closure_task,
    score_native_completion,
    select_native_closure_tasks,
)
from experiments.post_training.snowball_curriculum.tests.closure_rows import closure_row


def test_native_import_preserves_exact_payload_tokens_and_region_scoring():
    row = closure_row("original")
    imported = native_closure_task(**row)
    assert imported.lowered_task_spec == row["lowered_task_spec"]
    assert imported.prompt_token_ids == (128000, 42, 128005)
    assert imported.task_id == "original"
    score = score_native_completion(imported, "1} trailing assistant output")
    assert score.success and score.trailing_content and not score.whole_response_success
    assert not score_native_completion(imported, "2}").success
    assert not score_native_completion(imported, "}").success
    same_contract = native_closure_task(**closure_row("alias"))
    changed_reference = native_closure_task(**closure_row("different-answer", expected='{"a":2}'))
    changed_tokens = native_closure_task(**closure_row("different-prompt", tokens=(128000, 99, 128005)))
    changed_runtime = deepcopy(row)
    native = json.loads(changed_runtime["lowered_task_spec"])
    native["session"]["max_turns"] = 2
    changed_runtime["lowered_task_spec"] = json.dumps(native)
    assert same_contract.contract_id == imported.contract_id
    assert changed_reference.contract_id != imported.contract_id
    assert changed_reference.public_content_id == imported.public_content_id
    assert changed_tokens.contract_id != imported.contract_id
    assert changed_tokens.public_content_id != imported.public_content_id
    assert native_closure_task(**changed_runtime).contract_id != imported.contract_id


def test_native_selection_uses_token_holdouts_dedupes_aliases_and_balances_strata(tmp_path):
    train_rows = [
        closure_row("a", tokens=(1, 2, 3)),
        closure_row("a-alias", tokens=(1, 2, 3)),
        closure_row("b", tokens=(4, 5, 6)),
        closure_row("heldout-under-other-id", tokens=(7, 8, 9)),
        closure_row("c", tokens=(10, 11, 12), family="json_nested"),
    ]
    validation_rows = [closure_row("validation", tokens=(7, 8, 9))]
    train, validation = tmp_path / "train.parquet", tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(train_rows), train)
    pq.write_table(pa.Table.from_pylist(validation_rows), validation)
    arguments = {
        "expected_train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
        "expected_validation_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
    }
    selected = select_native_closure_tasks(str(train), str(validation), target=3, seed=42, **arguments)
    assert {row.task_id for row in selected.selected} == {"a", "b", "c"}
    assert selected.counts == {
        "validation": 1,
        "train": 5,
        "duplicate_contract": 1,
        "heldout_overlap": 1,
        "eligible": 3,
        "selected": 3,
    }
    assert selected.eligible_counts == {
        "json/json_object/continue_required": 2,
        "json/json_nested/continue_required": 1,
    }
    aliases = selected.selected_aliases[next(row.contract_id for row in selected.selected if row.task_id == "a")]
    assert {alias.task_id for alias in aliases} == {"a", "a-alias"}
    balanced = select_native_closure_tasks(str(train), str(validation), target=2, seed=42, **arguments)
    assert {row.family for row in balanced.selected} == {"json_object", "json_nested"}
    # A native file changing under its published identity must fail before selection.
    train.write_bytes(train.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="file hash differs"):
        select_native_closure_tasks(str(train), str(validation), target=2, seed=42, **arguments)


def test_native_selection_balances_families_before_controls(tmp_path):
    rows = [
        closure_row(
            str(i),
            tokens=(i,),
            family="json_object" if i < 4 else "json_nested",
            control="close_required" if i < 2 else "continue_required",
        )
        for i in range(8)
    ]
    train, validation = tmp_path / "train.parquet", tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(rows), train)
    pq.write_table(pa.Table.from_pylist([closure_row("validation", tokens=(99,))]), validation)
    result = select_native_closure_tasks(
        str(train),
        str(validation),
        target=4,
        seed=42,
        expected_train_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
        expected_validation_sha256=hashlib.sha256(validation.read_bytes()).hexdigest(),
    )
    assert result.family_quotas == {"json_nested": 2, "json_object": 2}
    assert result.stratum_quotas == {
        "json/json_object/close_required": 1,
        "json/json_object/continue_required": 1,
        "json/json_nested/continue_required": 2,
    }
    assert sorted(row.family for row in result.selected) == ["json_nested", "json_nested", "json_object", "json_object"]

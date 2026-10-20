# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Selection behavior across scarce sources, duplicates and heldout contracts."""

import json
from dataclasses import replace

import pytest
from taskcompendium.identity import canonical_sha256

from experiments.post_training.snowball_curriculum.selection import (
    TaskAlias,
    TaskCandidate,
    allocate_quotas,
    select_candidates,
    write_selection,
)


def test_capped_weights_redistribute_without_padding():
    assert allocate_quotas({"rare": 2, "large": 20, "excluded": 30}, 10, {"rare": 9, "large": 1}) == {
        "excluded": 0,
        "large": 8,
        "rare": 2,
    }
    assert allocate_quotas({"a": 2, "b": 3}, 10, {"a": 1, "b": 1}) == {"a": 2, "b": 3}


def test_selection_filters_eval_overlap_and_deduplicates_before_quotas(tmp_path):
    rows = [
        TaskCandidate("a", "a0", "shared", "public0", "easy", {}),
        TaskCandidate("b", "b0", "shared", "public0", "easy", {}),
        TaskCandidate("a", "a1", "leak", "eval-public", "hard", {}),
        TaskCandidate("eval", "eval0", "eval-contract", "eval-public", "unknown", {}, intended_use="eval"),
        TaskCandidate("b", "b1", "unique", "public1", "hard", {}),
    ]
    result = select_candidates(lambda: iter(rows), target=10, seed=42)
    reverse = select_candidates(lambda: reversed(rows), target=10, seed=42)
    assert result.selected == reverse.selected
    assert {(row.source_id, row.task_id) for row in result.selected} == {("a", "a0"), ("b", "b1")}
    assert result.eligible_counts == {"a": 1, "b": 1}
    assert result.shortfall == 8
    assert result.rejected_counts == {"duplicate_contract": 1, "non_train": 1, "heldout_overlap": 1}
    assert result.selected_aliases["shared"] == (TaskAlias("a", "a0"), TaskAlias("b", "b0"))
    assert result.selected_aliases == reverse.selected_aliases
    write_selection(result, (), str(tmp_path / "frozen"), seed=42, target=10)
    aliases = [json.loads(line) for line in (tmp_path / "frozen" / "selected-aliases.jsonl").read_text().splitlines()]
    assert next(row for row in aliases if row["contract_id"] == "shared")["aliases"] == [
        {"source_id": "a", "task_id": "a0"},
        {"source_id": "b", "task_id": "b0"},
    ]


def test_selection_retains_distinct_private_contracts_and_balances_difficulty():
    rows = [TaskCandidate("a", str(i), str(i), "same-public", "easy" if i < 8 else "hard", {}) for i in range(10)]
    result = select_candidates(lambda: iter(rows), target=4, seed=42)
    assert len(result.selected) == 4
    assert result.eligible_counts == {"a": 10}
    assert result.difficulty_counts == {"a/easy": 2, "a/hard": 2}
    assert result.eligible_difficulty_counts == {"a": {"easy": 8, "hard": 2}}
    excluded = select_candidates(lambda: iter(rows), target=4, seed=42, heldout_public_content_ids=["same-public"])
    assert excluded.selected == ()
    assert excluded.shortfall == 4
    different_seed = select_candidates(lambda: iter(rows), target=4, seed=17)
    assert {row.task_id for row in different_seed.selected} != {row.task_id for row in result.selected}


def test_selection_detects_identity_changes_between_passes():
    rows = [TaskCandidate("a", "a0", "contract", "public", "unknown", {})]
    inputs = iter([rows, [replace(rows[0], contract_id="changed")]])
    with pytest.raises(ValueError, match="identities changed"):
        select_candidates(lambda: iter(next(inputs)), target=1, seed=0)


def test_historical_holdouts_exclude_changed_contracts_by_id_and_public_context():
    context = [{"role": "user", "content": "A held-out question"}]
    rows = [
        TaskCandidate("train", "historical-id", "new-grader", "new-public", "unknown", {}),
        TaskCandidate("train", "new-id", "changed-runtime", "changed-public", "unknown", {"context": context}),
        TaskCandidate("train", "kept", "kept-contract", "kept-public", "unknown", {}),
    ]
    result = select_candidates(
        lambda: iter(rows),
        target=3,
        seed=0,
        heldout_task_ids=["historical-id"],
        heldout_context_ids=[canonical_sha256(context)],
    )
    assert [row.task_id for row in result.selected] == ["kept"]
    assert result.rejected_counts == {"heldout_overlap": 2}
    assert result.shortfall == 2


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
def test_heldout_alias_excludes_the_entire_contract_before_allocating_quotas(explicit, reverse):
    rows = [
        TaskCandidate("a", "winner", "shared", "shared-public", "unknown", {}),
        TaskCandidate("b", "heldout-alias", "shared", "shared-public", "unknown", {}),
        TaskCandidate("a", "kept", "kept", "kept-public", "unknown", {}),
    ]
    if not explicit:
        rows.append(
            TaskCandidate(
                "eval", "heldout-alias", "changed-grader", "changed-public", "unknown", {}, intended_use="eval"
            )
        )
    result = select_candidates(
        lambda: reversed(rows) if reverse else iter(rows),
        target=2,
        seed=0,
        heldout_task_ids=["heldout-alias"] if explicit else [],
    )
    assert [row.task_id for row in result.selected] == ["kept"]
    assert result.eligible_counts == {"a": 1}
    assert result.candidate_counts == ({"a": 2, "b": 1} if explicit else {"a": 2, "b": 1, "eval": 1})
    assert result.rejected_counts["duplicate_contract"] == 1
    assert result.rejected_counts["heldout_overlap"] == 1
    assert result.shortfall == 1

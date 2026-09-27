# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Family-level durability of the easy-overlap accuracy evaluator."""

import gzip
import json

import pytest

from experiments.domain_phase_mix.evaluate_mariner_ladder_accuracy import (
    EXPECTED_TASKS,
    MMLU_FAMILY,
    discard_partials,
    family_leaves,
    load_partial,
    merge_family_outputs,
    partial_uri,
    save_partial,
)
from experiments.evals.olmo_base_easy_overlap import OLMO_BASE_EASY_OVERLAP_TASKS


def family_output(leaves: dict[str, int], accuracy: float) -> dict:
    """The sections of a harness output that the evaluator validates, for one family."""
    return {
        "results": {
            name: {"alias": name, "acc,none": accuracy, "outputs": [{}] * count} for name, count in leaves.items()
        },
        "n-samples": {name: {"original": count, "effective": count} for name, count in leaves.items()},
        "samples": {name: [{"doc_id": i} for i in range(count)] for name, count in leaves.items()},
        "configs": {name: {"num_fewshot": 0 if name == "lambada_0shot" else 5} for name in leaves},
        "versions": {name: 1 for name in leaves},
        "averages": {"macro_avg_acc,none": accuracy},
    }


def test_families_partition_the_67_leaves():
    expected = sorted(EXPECTED_TASKS)
    leaves = [name for task in OLMO_BASE_EASY_OVERLAP_TASKS for name in family_leaves(task.task_alias, expected)]
    assert sorted(leaves) == sorted(EXPECTED_TASKS)
    assert len(family_leaves(MMLU_FAMILY, sorted(EXPECTED_TASKS))) == 57


def test_merged_families_reproduce_the_single_call_sections_and_averages():
    first = family_output({"arc_easy_5shot": 4, "sciq_5shot": 2}, 0.5)
    second = family_output({"piqa_5shot": 6}, 1.0)
    merged = merge_family_outputs([first, second])
    assert set(merged["results"]) == {"arc_easy_5shot", "sciq_5shot", "piqa_5shot"}
    assert merged["n-samples"]["piqa_5shot"] == {"original": 6, "effective": 6}
    # macro over the three leaves, micro weighted by their effective counts: (4*0.5 + 2*0.5 + 6*1.0) / 12
    assert merged["averages"]["macro_avg_acc,none"] == pytest.approx(2 / 3)
    assert merged["averages"]["micro_avg_acc,none"] == pytest.approx(0.75)
    with pytest.raises(ValueError, match="overlap"):
        merge_family_outputs([first, first])


def test_partial_family_outputs_round_trip_and_are_validated(tmp_path):
    plan = {"output_root": "file://" + str(tmp_path), "rows": []}
    row = {"name": "checkpoint"}
    leaves = {"arc_easy_5shot": 3}
    assert load_partial(plan, row, "arc_easy_5shot", leaves) is None
    save_partial(plan, row, "arc_easy_5shot", family_output(leaves, 0.25))
    assert load_partial(plan, row, "arc_easy_5shot", leaves)["results"]["arc_easy_5shot"]["acc,none"] == 0.25
    # a saved family whose document coverage differs from the plan cannot be resumed
    with pytest.raises(ValueError, match="Incomplete evaluation split"):
        load_partial(plan, row, "arc_easy_5shot", {"arc_easy_5shot": 4})
    # tampering is caught by the per-leaf validation
    uri = partial_uri(plan, row, "arc_easy_5shot")
    broken = family_output(leaves, 0.25)
    del broken["samples"]["arc_easy_5shot"][0]
    with open(uri.removeprefix("file://"), "wb") as handle:
        handle.write(gzip.compress(json.dumps(broken).encode()))
    with pytest.raises(ValueError, match="Incomplete or duplicated"):
        load_partial(plan, row, "arc_easy_5shot", leaves)
    discard_partials(plan, row)
    assert load_partial(plan, row, "arc_easy_5shot", leaves) is None

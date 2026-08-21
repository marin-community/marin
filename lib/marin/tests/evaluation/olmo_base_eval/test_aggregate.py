# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregation parity against the SC oracle.

Independent oracle: per-task BPB and the derived macro come from the SC
OLMoBaseEval 300m panel (`table9_oracle_fixture.json`, extracted from the SC
`fit_panel_table9_macro.csv` + wide results). The implementation must reproduce
both the MMLU bucket collapse and the unweighted-mean macro from those per-task
values. Guards the "wrong aggregation" / "wrong macro denominator" classes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from marin.evaluation.olmo_base_eval.aggregate import assemble_table9, collapse_mmlu, table9_macro
from marin.evaluation.olmo_base_eval.components import MMLU_BUCKETS, leaf_components

FIXTURE = json.loads((Path(__file__).parent / "data" / "table9_oracle_fixture.json").read_text())
ORACLE_RUNS = sorted(FIXTURE["runs"])


@pytest.mark.parametrize("run_name", ORACLE_RUNS)
def test_collapse_mmlu_reproduces_sc_buckets(run_name):
    run = FIXTURE["runs"][run_name]
    collapsed = collapse_mmlu(run["mmlu_subjects"])
    for bucket in MMLU_BUCKETS:
        assert collapsed[bucket] == pytest.approx(run["buckets"][bucket], abs=1e-9)


@pytest.mark.parametrize("run_name", ORACLE_RUNS)
def test_table9_macro_reproduces_sc_macro(run_name):
    run = FIXTURE["runs"][run_name]
    assert table9_macro(run["components"]) == pytest.approx(run["macro"], abs=1e-9)


@pytest.mark.parametrize("run_name", ORACLE_RUNS)
def test_assemble_table9_from_leaves_and_subjects_matches_oracle(run_name):
    run = FIXTURE["runs"][run_name]
    leaf_bpb = {task: run["components"][task] for task in leaf_components()}
    assembled = assemble_table9(leaf_bpb, run["mmlu_subjects"])
    # Every one of the 51 components matches the oracle, and the macro follows.
    for component, value in run["components"].items():
        assert assembled[component] == pytest.approx(value, abs=1e-9)
    assert table9_macro(assembled) == pytest.approx(run["macro"], abs=1e-9)


def test_macro_is_unweighted_not_instance_weighted():
    # A deliberately skewed component vector: if the macro were weighted by
    # anything other than 1/51 per component, this would not equal the plain mean.
    components = {name: float(i) for i, name in enumerate(FIXTURE["runs"][ORACLE_RUNS[0]]["components"])}
    expected = sum(components.values()) / 51
    assert table9_macro(components) == pytest.approx(expected, abs=1e-12)


def test_macro_raises_on_missing_component():
    run = FIXTURE["runs"][ORACLE_RUNS[0]]
    incomplete = dict(run["components"])
    incomplete.pop("lambada")
    with pytest.raises(ValueError, match="missing Table 9 components"):
        table9_macro(incomplete)

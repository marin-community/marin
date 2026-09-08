# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assert actual dump producer keys and scorer schema have explicit registry entries."""

from dataclasses import fields

import pytest

from experiments.post_training.async_rl_audit import EVAL_RESPONSE_METRICS
from experiments.post_training.math_eval.metrics_registry import METRICS, lookup
from experiments.post_training.math_eval.scoring import ScoredRow


def test_all_native_dump_eval_keys_have_named_meanings():
    fixture = {
        f"eval/{dataset}/{key}": 0
        for dataset in ("all", "g03-gsm8k", "aime")
        for key in (
            *EVAL_RESPONSE_METRICS,
            "avg_score",
            "pass_at_1",
            "pass_at_8",
            "contract_correct",
            "contract_completed",
        )
    }
    for key in fixture:
        metric = lookup(key)
        assert metric.source_file_line and metric.denominator and metric.notes
        assert "dump_aggregate" in metric.producer


def test_every_scored_field_has_explicit_documented_semantics():
    for field in fields(ScoredRow):
        assert lookup(field.name).producer == "harness"
    assert len({metric.name for metric in METRICS}) == len(METRICS)
    assert "all response sequences" == lookup("score_contract_completed").denominator
    assert "fractional RG" in lookup("eval/rg/pass_at_1").notes


@pytest.mark.parametrize("name", ["eval/all/new_unexplained_metric", "eval/all/pass_at_zero", "made_up_score"])
def test_registry_fails_closed_for_unknown_metrics(name):
    with pytest.raises(KeyError, match="Unregistered"):
        lookup(name)

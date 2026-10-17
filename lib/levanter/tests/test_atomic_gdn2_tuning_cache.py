# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Acceptance and retry behavior for the optional offline kernel tuning cache."""

import copy
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/bench/bench_atomic_gdn2.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("bench_atomic_gdn2_cache", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(benchmark)


@pytest.fixture
def accepted_pair():
    # Reduced recorded-result structure, with all acceptance fields retained.
    return [
        {
            "block_sizes": {"bt": 128, "bc": 64, "mb": 16},
            "mode": mode,
            "configuration_correctness_passed": True,
            "timing_accepted": True,
            "error": None,
            "steady_state_time": latency,
            "correctness": {name: {"passed": True, "finite": True} for name in names},
        }
        for mode, latency, names in (
            ("forward", 0.007, benchmark.FORWARD_NAMES),
            ("forward_backward", 0.019, benchmark.GRADIENT_NAMES),
        )
    ]


def test_tuning_ignores_faster_configuration_with_failed_paired_gate(accepted_pair):
    bad = copy.deepcopy(accepted_pair)
    for row in bad:
        row["block_sizes"]["mb"] = 32
        row["steady_state_time"] /= 2
    bad[0]["error"] = "correctness_failure"
    winner = benchmark.tuning_winner(accepted_pair + bad)
    assert winner["block_sizes"]["mb"] == 16
    assert benchmark.tuning_winner(bad) is None


def test_tuning_requires_every_input_gradient(accepted_pair):
    del accepted_pair[1]["correctness"]["dg"]
    assert benchmark.tuning_winner(accepted_pair) is None


@pytest.mark.parametrize("field,value", [("source", "new"), ("env", "new flags"), ("oracle", "cpu")])
def test_tuning_context_changes_force_full_sweep(accepted_pair, tmp_path, field, value):
    context = {"source": "old", "env": "old flags", "oracle": "tpu"}
    entries = {}
    benchmark.update_tuning_entry(entries, context, accepted_pair, tmp_path / "old.jsonl")
    context[field] = value
    assert list(benchmark.tuning_trials(entries, context, [(256, 16), (128, 16)], [])) == [(256, 16), (128, 16)]


def test_cache_hit_requires_fresh_measurement_and_replaces_historical_time(accepted_pair, tmp_path):
    context = {"source": "fixed"}
    entries = {}
    benchmark.update_tuning_entry(entries, context, accepted_pair, tmp_path / "old.jsonl")
    path = tmp_path / "cache.json"
    benchmark.save_tuning_cache(path, entries)
    restored = benchmark.load_tuning_cache(path)
    measured = []
    trials = benchmark.tuning_trials(restored, context, [(256, 16), (128, 16)], measured)
    assert next(trials) == (128, 16)
    # No current result means the next bounded alternative must still run.
    assert next(trials) == (256, 16)
    assert benchmark.tuning_winner(measured) is None
    trials = benchmark.tuning_trials(restored, context, [(256, 16), (128, 16)], measured)
    assert next(trials) == (128, 16)
    fresh = copy.deepcopy(accepted_pair)
    fresh[1]["steady_state_time"] = 0.023
    measured.extend(fresh)
    assert list(trials) == []
    benchmark.update_tuning_entry(restored, context, measured, tmp_path / "new.jsonl")
    benchmark.save_tuning_cache(path, restored)
    winner = benchmark.load_tuning_cache(path)[benchmark.tuning_key(context)]
    assert winner["objective_time"] == 0.023
    assert winner["results_path"] == str(tmp_path / "new.jsonl")


def test_failed_hit_exhausts_alternatives_and_retains_failure(accepted_pair, tmp_path):
    context = {"source": "fixed"}
    entries = {}
    benchmark.update_tuning_entry(entries, context, accepted_pair, tmp_path / "old.jsonl")
    measured = []
    trials = benchmark.tuning_trials(entries, context, [(256, 16), (128, 16), (128, 32)], measured)
    assert next(trials) == (128, 16)
    failed = copy.deepcopy(accepted_pair)
    failed[1]["error"] = "compile_failure"
    failed[1]["timing_accepted"] = False
    measured.extend(failed)
    benchmark.update_tuning_entry(entries, context, measured, tmp_path / "failed.jsonl")
    assert benchmark.tuning_key(context) not in entries
    assert next(trials) == (256, 16)
    first_alternative = copy.deepcopy(accepted_pair)
    for row in first_alternative:
        row["block_sizes"] = {"bt": 256, "bc": 128, "mb": 16}
    measured.extend(first_alternative)
    # Success after a failed cache hit must not truncate the remaining sweep.
    assert next(trials) == (128, 32)
    faster_alternative = copy.deepcopy(accepted_pair)
    for row in faster_alternative:
        row["block_sizes"]["mb"] = 32
        row["steady_state_time"] /= 2
    measured.extend(faster_alternative)
    assert list(trials) == []
    benchmark.update_tuning_entry(entries, context, measured, tmp_path / "recovered.jsonl")
    assert entries[benchmark.tuning_key(context)]["block_sizes"] == {"bt": 128, "bc": 64, "mb": 32}
    assert measured[1]["error"] == "compile_failure"

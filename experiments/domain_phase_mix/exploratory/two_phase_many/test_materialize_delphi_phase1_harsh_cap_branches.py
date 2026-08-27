# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import fsspec
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_phase1_harsh_cap_branches as materialize,
)


def write_terminal_records(records: list[dict[str, object]]) -> tuple[fsspec.AbstractFileSystem, str]:
    fs = fsspec.filesystem("memory")
    output_path = "/branch"
    with fs.open(f"{output_path}/checkpoints/eval_metrics.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return fs, output_path


def test_terminal_metrics_accepts_duplicate_scientific_result_with_different_timing() -> None:
    first = {
        "step": materialize.TERMINAL_STEP,
        "eval/loading_time": 1.0,
        "eval/total_time": 2.0,
        materialize.TARGET: 0.98,
    }
    second = {**first, "eval/loading_time": 3.0, "eval/total_time": 4.0}
    fs, output_path = write_terminal_records([first, second])

    assert materialize.terminal_metrics(fs, output_path) == {"bpb": 0.98}


def test_terminal_metrics_rejects_duplicate_scientific_result_with_different_metric() -> None:
    first = {"step": materialize.TERMINAL_STEP, materialize.TARGET: 0.98}
    second = {**first, materialize.TARGET: 0.99}
    fs, output_path = write_terminal_records([first, second])

    with pytest.raises(ValueError, match="Conflicting step"):
        materialize.terminal_metrics(fs, output_path)

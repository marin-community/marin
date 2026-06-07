# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

import scripts.analysis.eval_decon_val_sets as eval_decon


def test_require_unused_output_detects_json_tracker_child(monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = "gs://bucket/evals/run/step-1/metrics.jsonl"
    result_path = f"{output_dir}/eval_results.json"

    monkeypatch.setattr(eval_decon, "fsspec_exists", lambda path: path == result_path)
    monkeypatch.setattr(eval_decon, "fsspec_glob", lambda path: [])

    with pytest.raises(RuntimeError, match="eval output already exists"):
        eval_decon.require_unused_output(output_dir, force=False)

    eval_decon.require_unused_output(output_dir, force=True)


def test_require_unused_output_detects_tracker_children(monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = "gs://bucket/evals/run/step-1/metrics.jsonl"
    child_path = f"{output_dir}/partial.json"

    monkeypatch.setattr(eval_decon, "fsspec_exists", lambda path: False)
    monkeypatch.setattr(eval_decon, "fsspec_glob", lambda path: [child_path])

    with pytest.raises(RuntimeError, match=child_path):
        eval_decon.require_unused_output(output_dir, force=False)

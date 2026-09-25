# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks that explicit lm-eval shot counts match the saved effective task config."""

import json
from pathlib import Path

import pytest
from marin.evaluation.lm_eval import LmEvalRun, run_lm_eval
from marin.inference.types import OpenAIEndpoint, RunningModel


@pytest.mark.parametrize("effective", [0, 10])
def test_run_lm_eval_checks_effective_fewshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effective: int) -> None:
    def fake_run(_command: list[str], *, check: bool) -> None:
        results_dir = tmp_path / "model"
        results_dir.mkdir()
        (results_dir / "results_test.json").write_text(
            json.dumps({"configs": {"humaneval": {"num_fewshot": effective}}})
        )

    monkeypatch.setattr("marin.evaluation.lm_eval.subprocess.run", fake_run)
    model = RunningModel(endpoint=OpenAIEndpoint(base_url="http://localhost:8000/v1", model="test"), tokenizer=None)
    run = LmEvalRun(tasks=("humaneval",), num_fewshot=10)
    if effective == 0:
        with pytest.raises(ValueError, match="humaneval used 0 shots; requested 10"):
            run_lm_eval(model, run, str(tmp_path))
    else:
        assert run_lm_eval(model, run, str(tmp_path)) is None

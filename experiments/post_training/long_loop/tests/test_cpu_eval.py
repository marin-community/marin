# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import pytest
from marin.execution.lazy import OUT, apply

from experiments.post_training.long_loop.graph import execute_local_graph

REPO = Path(__file__).resolve().parents[4]


def run_cpu_eval(*, output: str):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        PYTHONPATH=f"{REPO}:{REPO}/vendor/skyrl:{REPO}/vendor/skyrl/skyrl-train:{REPO}/vendor/skyrl/skyrl-gym",
        RAY_ENABLE_UV_RUN_RUNTIME_ENV="0",
    )
    with (root / "cpu-eval.log").open("w") as log:
        result = subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(REPO / "vendor/skyrl"),
                "--frozen",
                "--extra",
                "cpu",
                "--no-group",
                "dev",
                "--with",
                "pytest==8.4.2",
                "python",
                str(Path(__file__).with_name("cpu_eval_fixture.py")),
                str(root),
            ],
            cwd=REPO,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )
    if result.returncode:
        raise RuntimeError(
            f"CPU eval fixture exited {result.returncode}:\n{(root / 'cpu-eval.log').read_text()[-16000:]}"
        )
    assert json.loads((root / "terminal.json").read_text())["result"]["state"] == "succeeded"


@pytest.mark.timeout(180)
def test_eval_only_commits_after_required_retention(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    stage = apply("long-loop/cpu-eval", run_cpu_eval, version="2026.11.09", output=OUT)
    execute_local_graph(stage)
    root = Path(stage.path())
    assert (root / ".artifact.json").is_file()
    records = json.loads((root / "retention-verified.json").read_text())
    assert len(records) == 2
    metrics = json.loads((root / "metrics.json").read_text())
    assert metrics["eval/all/num_scored"] == 2

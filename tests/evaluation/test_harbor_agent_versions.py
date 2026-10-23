# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline regressions for the Pi version drift reported in issue #10283."""

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from marin.evaluation.harbor.driver_config import _driver_environment

_ROOT = Path(__file__).parents[2]
_DRIVER = _ROOT / "lib/marin/src/marin/evaluation/harbor/trial_driver.py"


@pytest.mark.parametrize(
    ("agent_name", "policy_kwargs", "expected_policy_version", "expected_runtime_version"),
    [
        ("pi", {}, "0.87.0", "0.87.0"),
        ("pi", {"version": "1.3.0"}, "0.87.0", "0.87.0"),
        ("pi", {"version": "0.87.0"}, "0.87.0", "0.87.0"),
        ("terminus-2", {"version": "1.3.0"}, "1.3.0", "1.3.0"),
        ("terminus-2", {}, None, "9.9.9"),
        ("acp:pi-acp@0.0.33", {}, None, "9.9.9"),
    ],
)
def test_agent_version_survives_preflight_and_runtime_overlay(
    tmp_path, agent_name, policy_kwargs, expected_policy_version, expected_runtime_version
):
    task = tmp_path / "dataset" / "task"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('[task]\nname = "test/version-pin"\n')
    (task / "instruction.md").write_text("Complete the task.\n")
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "environment": {"type": "docker"},
                "agents": [{"name": agent_name, "kwargs": policy_kwargs}],
                "datasets": [{"path": "dataset"}],
            }
        )
    )
    model_kwargs = {
        "version": "9.9.9",
        "thinking_format": "chat-template",
        "model_info": {"max_input_tokens": 65536, "max_output_tokens": 16384},
    }
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps([{"path": str(source), "model_agent_kwargs": model_kwargs}]))
    command = [
        "uv",
        "run",
        "--isolated",
        "--project",
        str(_ROOT / "vendor/harbor"),
        "--extra",
        "runtime",
        "--no-default-groups",
        "--frozen",
        "python",
    ]
    environment = _driver_environment()
    policy = tmp_path / "policy.json"
    overlay = tmp_path / "overlay.json"
    overlay.write_text(
        json.dumps(
            {
                "job_name": "version-pin",
                "jobs_dir": str(tmp_path / "jobs"),
                "dataset_path": str(task.parent),
                "endpoint_url": "https://model.example/v1",
                "served_model": "test-model",
                "task_limit": 1,
                "model_agent_kwargs": model_kwargs,
                "verifier_env": {},
                "archive_root": str(tmp_path / "archive"),
                "archive_dataset": "dataset",
            }
        )
    )
    script = textwrap.dedent(
        """
        import io
        import json
        import sys
        from contextlib import redirect_stdout
        from pathlib import Path

        from marin.evaluation.harbor.trial_driver import effective_job_config, main

        driver, requests, policy, overlay = sys.argv[1:]
        sys.argv = [driver, "preflight", requests]
        output = io.StringIO()
        with redirect_stdout(output):
            main()
        (validated,) = json.loads(output.getvalue())
        Path(policy).write_text(validated["stable_policy_json"])
        effective = effective_job_config(Path(policy), Path(overlay))
        print(json.dumps({
            "stable": json.loads(validated["stable_policy_json"]),
            "effective": effective.model_dump(mode="json"),
        }))
        """
    )
    rendered = subprocess.run(
        [*command, "-c", script, str(_DRIVER), str(requests), str(policy), str(overlay)],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(rendered.stdout)
    stable = result["stable"]
    assert stable["agents"][0]["name"] == agent_name
    assert stable["agents"][0]["kwargs"].get("version") == expected_policy_version
    agent = result["effective"]["agents"][0]
    assert agent["name"] == agent_name
    assert agent["kwargs"]["version"] == expected_runtime_version
    assert agent["kwargs"]["thinking_format"] == "chat-template"
    assert agent["kwargs"]["model_info"]["max_output_tokens"] == 16384

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
from click.testing import CliRunner
from marin.experiment import cli as experiment_cli

from experiments.post_training.russell_rsi import launch_contract_tasks as contract_launch


@pytest.mark.parametrize("method,response_cap", [("teacher-statement-v1", 6), ("frozen-obligations-v1", 0)])
def test_contract_cpu_entrypoint_binds_pinned_inputs_without_execution(tmp_path, monkeypatch, method, response_cap):
    config = tmp_path / "contracts.json"
    config.write_text(
        json.dumps(
            {
                "manifest_uri": "/tmp/frozen-sources.json",
                "manifest_sha256": "a" * 64,
                "output_path": "/tmp/ignored-output",
                "relay_job": "relay",
                "stage": "prepare",
                "capabilities_uri": "",
                "capabilities_sha256": "",
                "response_cap": response_cap,
                "admission_concurrency": 2,
                "statement_review_uri": "",
                "statement_review_sha256": "",
                "prepared_manifest_uri": "",
                "prepared_manifest_sha256": "",
                "method": method,
                "max_contracts": 6,
                "observation_manifest_uri": "/tmp/observations.json",
                "observation_manifest_sha256": "b" * 64,
                "observation_source_manifest_uri": "/tmp/original-source.json",
                "observation_source_manifest_sha256": "c" * 64,
            }
        )
    )
    captured = []
    monkeypatch.setattr(experiment_cli, "_print_plan", captured.extend)
    result = CliRunner().invoke(
        contract_launch.main,
        [
            "--config-uri",
            str(config),
            "--config-sha256",
            hashlib.sha256(config.read_bytes()).hexdigest(),
            "--version",
            "2026.10.04",
        ],
    )
    assert result.exit_code == 0, str(result.exception) + result.output
    payload = json.loads(captured[0].fingerprint_payload())
    assert payload["manifest_sha256"] == "a" * 64
    assert payload["stage"] == "prepare"
    assert payload["response_cap"] == response_cap
    assert payload["method"] == method
    if method == "frozen-obligations-v1":
        monkeypatch.delenv(contract_launch.GLM_TOKEN_ENV, raising=False)
        invocations = []

        def remote_boundary(function, **options):
            def submit(value):
                invocations.append((options, value))

            return submit

        monkeypatch.setattr(contract_launch, "remote", remote_boundary)
        contract_launch.run_contract_tasks(contract_launch.ContractTasksConfig(**json.loads(config.read_text())))
        assert invocations[0][0]["env_vars"] == {}
        assert invocations[0][1].method == method


def test_contract_cpu_run_requires_iris_before_any_worker(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text("{}")
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    monkeypatch.setattr(contract_launch, "has_current_context", lambda: False)
    result = CliRunner().invoke(
        contract_launch.main,
        [
            "--config-uri",
            str(config),
            "--config-sha256",
            hashlib.sha256(config.read_bytes()).hexdigest(),
            "--version",
            "2026.10.04",
            "--run",
        ],
    )
    assert result.exit_code != 0
    assert "inside the CW02 Iris context" in result.output

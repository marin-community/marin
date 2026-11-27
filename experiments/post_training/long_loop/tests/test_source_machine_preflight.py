# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

from marin.rl.skyrl import skyrl_cpu_command


def test_source_admission_checks_real_machine_bindings_without_inference_or_sandboxes(tmp_path):
    result = subprocess.run(
        skyrl_cpu_command(
            "python", "-m", "experiments.post_training.long_loop.tests.source_machine_probe", str(tmp_path)
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert "thin-actor" in observed["unbound_error"]
    assert [row["task_id"] for row in observed["receipts"]] == ["plain", "thin-actor"]
    assert observed["receipts"][1]["task_machine"]["backend"] == "daytona"
    assert observed["receipts"][0]["combined_input_features"]["task_spec"]["dtype"] == "large_string"
    assert observed["incompatible_schema_rejected"] is True

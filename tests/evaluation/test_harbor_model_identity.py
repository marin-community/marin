# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preflight policy identity is independent of model and serving names."""

import json
import tempfile
from pathlib import Path

from marin.evaluation.harbor.driver_config import preflight_harbor_configs


def test_preflight_policy_identity_ignores_model_and_serving_names(monkeypatch):
    monkeypatch.setenv("DAYTONA_TARGET", "test-target")
    with tempfile.TemporaryDirectory(prefix=".harbor-identity-", dir=Path(__file__).parents[2]) as directory:
        tmp_path = Path(directory)
        task = tmp_path / "dataset/task"
        (task / "environment").mkdir(parents=True)
        (task / "tests").mkdir()
        (task / "task.toml").write_text('[task]\nname = "test/model-identity"\n')
        (task / "instruction.md").write_text("Complete the task.\n")
        (task / "tests/test.sh").write_text("exit 0\n")
        paths = []
        for model_name, model_alias in [
            ("hosted_vllm/org/original", "hosted_vllm/old"),
            ("hosted_vllm/org/other", "hosted_vllm/new"),
        ]:
            path = tmp_path / f"policy-{len(paths)}.json"
            path.write_text(
                json.dumps(
                    {
                        "environment": {"type": "daytona"},
                        "agents": [{"name": "terminus-2", "model_name": model_name, "model_alias": model_alias}],
                        "datasets": [{"path": "dataset"}],
                    }
                )
            )
            paths.append(path)
        configs = preflight_harbor_configs([(path, {}) for path in paths], judge_model=None)
        assert configs[0].digest == configs[1].digest

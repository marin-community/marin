# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate every released schema, native Pi limits, and fresh-wheel grader loading."""

import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from harbor.agents.installed.pi import Pi
from rigging.filesystem.buckets import filesystem_for
from rolloutengine.models import TaskRolloutSpec
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes

BASE = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/dataset-release-20261030"


def main():
    root = Path(tempfile.mkdtemp(prefix="verify-mini-release-"))
    fs, path = filesystem_for(BASE + "/release.tar.gz")
    archive = root / "release.tar.gz"
    archive.write_bytes(fs.cat(path))
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    release = root / "release"
    mini = pq.read_table(release / "mini/data/validation.parquet").to_pylist()
    micro = pq.read_table(release / "micro/data/validation.parquet").to_pylist()
    assert len(mini) == 840 and len(micro) == 210 and mini[:210] == micro
    assert len({r["task_id"] for r in mini}) == 840 and len({r["benchmark"] for r in mini}) == 21
    for row in mini:
        task = TaskSpec.model_validate_json(row["task_spec"])
        rollout = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
        assert rollout.task == task
        assert rollout.limits.max_total_tokens is None
        if row["agent"] == "pi":
            assert rollout.harness == "harbor_trial" and "harbor" in task.tags
        else:
            assert rollout.limits.max_turns == 1
    agent_config = yaml.safe_load((release / "runtime/harbor-agent.yaml").read_text())
    agent = Pi(
        model_name="hosted_vllm/mini-validation",
        logs_dir=root / "pi",
        api_base="http://127.0.0.1:1/v1",
        **agent_config["kwargs"],
    )
    assert agent._model_limits.context_window == 65536 and agent._model_limits.max_tokens == 16384
    assert agent_config["kwargs"]["thinking"] == "high" and agent_config["kwargs"]["version"] == "0.87.0"
    # A fresh environment proves the published wheels and their frozen requirements load together.
    grader = root / "grader-env"
    subprocess.run(["uv", "venv", "--python", "3.12", str(grader)], check=True)
    packages = json.loads((release / "runtime/grading-requirements.json").read_text())
    subprocess.run(
        ["uv", "pip", "install", "--no-config", "--python", str(grader / "bin/python"), *packages], check=True
    )
    controls = []
    for benchmark in ["NUPA", "gsm8k-0shot"]:
        row = next(row for row in mini if row["benchmark"] == benchmark)
        task = TaskSpec.model_validate_json(row["task_spec"])
        private = root / ("private-" + benchmark)
        private.mkdir()
        for resource in task.resources.verifier:
            target = private / resource.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(resource_bytes(resource))
        config = json.loads((private / "config.json").read_text())
        source = config["spec"]
        answer = (
            str(source["target"])
            if benchmark == "NUPA"
            else "\\boxed{" + source["doc"]["answer"].rsplit("#### ", 1)[-1] + "}"
        )
        script = """
import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from grade import grade_response
config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
reference = asyncio.run(grade_response(config, sys.argv[2]))
empty = asyncio.run(grade_response(config, ""))
assert reference == 1 and empty == 0, (reference, empty)
print(json.dumps({"reference": reference, "empty": empty}))
"""

        result = subprocess.run(
            [str(grader / "bin/python"), "-c", script, str(private), answer],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(private)},
        )
        controls.append({"benchmark": benchmark, "fresh_wheel_result": result.stdout.strip()})
    result = {
        "unique_mini": 840,
        "unique_micro": 210,
        "nested": True,
        "schema_rows": 840,
        "pi_context_window": 65536,
        "pi_output_tokens": 16384,
        "pi_version": "0.87.0",
        "thinking": "high",
        "fresh_wheel_controls": controls,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    data = json.dumps(result, indent=2).encode()
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / "verification.json").write_bytes(data)
    fs, path = filesystem_for(BASE + "/verification.json")
    fs.pipe_file(path, data)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

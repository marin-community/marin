# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read back the public release and verify source selection and runtime schemas."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download
from rolloutengine.models import TaskRolloutSpec
from taskcompendium.models import TaskSpec


def main():
    receipts = json.loads(os.environ["MINI_RELEASE_REVISIONS"])
    revisions = {row["repo"]: row["revision"] for row in receipts}
    datasets = {}
    counts = {}
    for name, expected in [("Marin-Mini-Eval-840", 840), ("Marin-Micro-Eval-210", 210)]:
        repo = "open-athena/" + name
        rows = load_dataset(repo, revision=revisions[repo], split="validation")
        expanded = load_dataset(repo, "rollouts", revision=revisions[repo], split="validation")
        assert len(rows) == expected
        assert len(expanded) == (1626 if expected == 840 else 450)
        datasets[expected] = list(rows)
        counts[name] = {"unique": len(rows), "rollouts": len(expanded)}
        for row in rows:
            task = TaskSpec.model_validate_json(row["task_spec"])
            rollout = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
            assert task == rollout.task
            assert rollout.limits.max_total_tokens is None
            if row["agent"] == "pi":
                assert rollout.harness == "harbor_trial"
    assert datasets[840][:210] == datasets[210]
    repo = "open-athena/Marin-Mini-Eval-Policy-Index"
    root = Path(
        snapshot_download(
            repo,
            repo_type="dataset",
            revision=revisions[repo],
            allow_patterns=["index/*", "code/reproduce_release_policy.py", "runtime/harbor-agent.yaml"],
        )
    )
    rows = [json.loads(line) for line in (root / "index/tasks.jsonl").read_text().splitlines()]
    assert len(rows) == 840
    agent = yaml.safe_load((root / "runtime/harbor-agent.yaml").read_text())
    assert agent["kwargs"]["model_info"]["max_input_tokens"] == 65536
    assert agent["kwargs"]["model_info"]["max_output_tokens"] == 16384
    assert agent["kwargs"]["thinking"] == "high"
    for budget in [210, 840]:
        output = Path(tempfile.mkdtemp())
        subprocess.run(
            [
                sys.executable,
                str(root / "code/reproduce_release_policy.py"),
                "--index",
                str(root),
                "--budget",
                str(budget),
                "--output",
                str(output),
            ],
            check=True,
        )
        selected = sorted(rows, key=lambda row: row["selection_order"])[:budget]
        for benchmark in {row["benchmark"] for row in selected}:
            group = [row for row in selected if row["benchmark"] == benchmark]
            source = group[0]["policy"]
            config = yaml.safe_load((output / (benchmark + ".yaml")).read_text())
            if source["kind"] == "harbor":
                original = yaml.safe_load(source["original_config"])
                task_names = config["datasets"][0].pop("task_names")
                assert task_names == sorted(row["policy"]["task_name"] for row in group)
            else:
                original = yaml.safe_load(source["original_parent"]["config_text"])
                task = config["tasks"][0]
                keys = config["task_options"][task].pop("selected_source_keys")
                assert keys == sorted({key for row in group for key in row["policy"]["source_keys"]})
                if not config["task_options"][task]:
                    del config["task_options"][task]
                if not config["task_options"] and "task_options" not in original:
                    del config["task_options"]
            assert config == original, benchmark
    receipt = {
        "public_readback": "passed",
        "counts": counts,
        "nested": True,
        "schemas": "passed",
        "reproduction_selection_only": "passed",
        "revisions": receipts,
    }
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / "public-readback.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

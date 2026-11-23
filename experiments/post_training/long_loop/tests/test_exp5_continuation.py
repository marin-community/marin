# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from marin.execution.lazy import StepContext
from marin.experiment.cli import graph_handles
from marin.rl.skyrl import ArtifactHfModel
from rigging.filesystem.cluster_config import StoreType
from rigging.filesystem.s3_compat import s3_credentials

from experiments.post_training.long_loop.exp5 import certified_sft_inputs, main
from experiments.post_training.long_loop.run_spec import FilePin


@pytest.fixture
def certification(tmp_path):
    root = tmp_path / "sft"
    checkpoint = root / "checkpoints/step-527"
    hf = root / "hf/step-527"
    hf.mkdir(parents=True)
    receipts = root / "recovery/finalization-527"
    receipts.mkdir(parents=True)

    def write(path, document):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(document).encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    config_sha = write(hf / "config.json", {"model_type": "grug_moe", "max_position_embeddings": 262144})
    completion = root / "telemetry/completion.json"
    completion_sha = write(completion, {"checkpoint": str(checkpoint), "step": 527, "completed_updates": 528})
    native_sha = write(
        receipts / "native-checkpoint.json",
        {"checkpoint": str(checkpoint), "step": 527, "is_temporary": False, "verdict": "full"},
    )
    hf_sha = write(
        receipts / "hf-export.json",
        {
            "native_checkpoint": str(checkpoint),
            "source": str(hf),
            "step": 527,
            "verdict": "full_hash_and_index",
            "files": [{"path": "config.json", "sha256": config_sha}],
        },
    )
    seal = receipts / "seal.json"
    seal_sha = write(
        seal,
        {
            "status": "COMPLETE",
            "step": 527,
            "completed_updates": 528,
            "native_checkpoint": str(checkpoint),
            "hf_export": str(hf),
            "completion_receipt": str(completion),
            "completion_sha256": completion_sha,
            "native_receipt_sha256": native_sha,
            "hf_receipt_sha256": hf_sha,
        },
    )
    return FilePin(str(seal), seal_sha), root, hf


def test_certified_continuation_resolves_saved_hf_without_training_or_copy(certification, tmp_path):
    pin, root, hf = certification
    completed = certified_sft_inputs(pin, name="fixture/exp5", version="2026.11.22.114")
    steps = graph_handles([completed.trained, completed.handoff])
    assert len(steps) == 2
    assert all(step.adopt_source == str(root) for step in steps)
    context = StepContext.for_run(str(tmp_path / "rl"), str(tmp_path), deps=steps)
    locator = ArtifactHfModel(
        completed.handoff, "fixture/tokenizer", "a" * 40, relative_path=completed.model_relative_path
    ).resolve(context)
    assert locator.uri == str(hf)


@pytest.mark.parametrize("changed_file", ["completion", "native", "hf", "config"])
def test_changed_certification_payload_refuses_model_adoption(certification, changed_file):
    pin, root, hf = certification
    paths = {
        "completion": root / "telemetry/completion.json",
        "native": root / "recovery/finalization-527/native-checkpoint.json",
        "hf": root / "recovery/finalization-527/hf-export.json",
        "config": hf / "config.json",
    }
    paths[changed_file].write_text("{}")
    with pytest.raises(ValueError, match="Pinned file changed"):
        certified_sft_inputs(pin, name="fixture/exp5", version="2026.11.22.114")


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_exp5_rl_no_run_binds_its_published_pools_without_an_exp4_dependency(tmp_path):
    if s3_credentials(StoreType.COREWEAVE) is None:
        pytest.skip("CoreWeave credentials are required for the published Exp5 preflight")
    workspace = Path(__file__).resolve().parents[4]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()
    result = CliRunner().invoke(
        main,
        [
            "--round",
            str(Path(__file__).parents[1] / "rounds/exp5-setup.yaml"),
            "--source-revision",
            revision,
            "--version",
            "2026.11.22.133",
            "--stage",
            "rl",
            "--completed-sft-record",
            "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp5/long-loop-exp5/round5/sft/"
            "2026.11.21.113/recovery/finalization-527-20261122/seal.json",
            "--completed-sft-sha256",
            "935f34393e6cd26844dd72c0ce439a3a1e1d4336aed7adae50cfe5b3ce52b790",
            "--validation-output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    resolved = json.loads((tmp_path / "resolved-plan.json").read_text())
    assert resolved["datasets"]["pools"]["uri"] == (
        "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp5/long-loop-exp5/rl-pools/2026.11.22.1"
    )
    assert resolved["curriculum"]["identity"] == "long-loop/exp5-rl@2026.11.20:995e3844"
    assert resolved["continuation"]["sft_replayed"] is False
    assert resolved["exp5_admission"]["launch_ready"]
    graph = json.loads((tmp_path / "graph.json").read_text())
    [rl] = [entry for entry in graph if entry["status"].startswith("strictly composed")]
    assert rl["initializer_config_sha256"] == "bf3da5aaf9ae83084fd704d53dea5eefa93bea5b4b8d300eff5c18f617fd878b"
    launches = [yaml.safe_load(path.read_text()) for path in tmp_path.glob("*/launch.yaml")]
    [launch] = launches
    assert launch["skyrl"]["trainer"]["rollout_buffer"]["consumed_route_quotas"] == {"agentic": 13, "nonagentic": 51}
    [best] = [entry for entry in launch["skyrl"]["trainer"]["callbacks"] if entry["type"] == "best_checkpoint"]
    assert best["expected_tasks"] == 200
    assert best["minimum_scored_tasks"] == 180

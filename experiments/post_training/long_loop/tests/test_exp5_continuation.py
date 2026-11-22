# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
from marin.execution.lazy import StepContext
from marin.experiment.cli import graph_handles
from marin.rl.skyrl import ArtifactHfModel

from experiments.post_training.long_loop.exp5 import certified_sft_inputs
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

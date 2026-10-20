# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
from marin.execution.lazy import StepContext

from experiments.post_training.agentic_sweep.arms import PARENTS, ArmEpoch, pipeline
from experiments.post_training.agentic_sweep.train import MODEL, REVISION, TOKENIZER


def manifest(parent_name, arm):
    parent = PARENTS[parent_name]
    identity = {"name": parent_name, "model": parent.model, "revision": parent.revision}
    return {
        "arm": arm,
        "parent_model": identity,
        "tokenizer_source": {"model": MODEL, "revision": REVISION, "note": "Shared tokenizer, distinct parent"},
        "relay_student": identity if arm == "relay" else None,
        "max_sequence_tokens": 32768,
        "pad_token_id": 128001,
    }


def graph(rows, parent_name, plateau=False):
    return pipeline(
        rows,
        "a" * 64,
        "/production/rows",
        parent_name,
        "20261020",
        1e-6,
        [ArmEpoch(f"epoch-{epoch}", 16, 1, 128, 256) for epoch in range(3)],
        plateau,
    )


def test_parent_and_plateau_outputs_are_disjoint_and_use_the_selected_weights():
    outputs, tracker_ids, fingerprints = set(), set(), set()
    for parent_name in PARENTS:
        for method in ("plain", "rs-multi", "relay", "recovery", "plateau"):
            arm = "plain" if method == "plateau" else method
            artifacts = graph(manifest(parent_name, arm), parent_name, method == "plateau")
            trained = artifacts["train"]
            output = trained.path("/outputs")
            pod = trained.build_config(StepContext.for_run(output, "/outputs", deps=trained.deps))
            config = pod.train_config
            assert config.initialize_from_hf == PARENTS[parent_name].uri
            assert config.data.tokenizer == TOKENIZER
            assert config.trainer.num_train_steps == 3
            assert (config.plateau_dropping is not None) == (method == "plateau")
            assert trained.name == f"checkpoints/karan/{method}-{parent_name}-v7-20261020-lr1e-06-3ep"
            assert trained.version == "2026.10.20"
            outputs.add(output)
            tracker_ids.add(config.trainer.id)
            fingerprints.add(trained.fingerprint())
    assert len(outputs) == len(tracker_ids) == len(fingerprints) == 10


def test_wrong_parent_tokenizer_or_relay_student_cannot_build_a_training_graph():
    rows = manifest("v0.1", "relay")
    with pytest.raises(ValueError, match="different parent"):
        graph(rows, "v0")
    changed_tokenizer = deepcopy(rows)
    changed_tokenizer["tokenizer_source"]["revision"] = "another-tokenizer"
    with pytest.raises(ValueError, match="tokenizer"):
        graph(changed_tokenizer, "v0.1")
    changed_student = deepcopy(rows)
    changed_student["relay_student"] = manifest("v0", "relay")["parent_model"]
    with pytest.raises(ValueError, match="relay student"):
        graph(changed_student, "v0.1")

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


def graph(rows, parent_name, plateau=False, data_variant="high", context_length=32768):
    return pipeline(
        rows,
        "a" * 64,
        "/production/rows",
        parent_name,
        "20261020",
        1e-6,
        [ArmEpoch(f"epoch-{epoch}", 16, 1, 128, 256) for epoch in range(3)],
        plateau,
        data_variant=data_variant,
        context_length=context_length,
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


@pytest.mark.parametrize("parent_name", tuple(PARENTS))
def test_low_plain_has_distinct_cache_checkpoint_and_tracker_identity(parent_name):
    rows = manifest(parent_name, "plain")
    high = graph(rows, parent_name)
    low = graph(rows, parent_name, data_variant="low")
    high_train, low_train = high["train"], low["train"]
    assert low_train.name == f"checkpoints/karan/plain-low-{parent_name}-v7-20261020-lr1e-06-3ep"
    assert low_train.path("/outputs") != high_train.path("/outputs")
    assert low_train.fingerprint() != high_train.fingerprint()
    assert {cache.path("/outputs") for cache in high["caches"]}.isdisjoint(
        cache.path("/outputs") for cache in low["caches"]
    )
    configurations = [
        trained.build_config(StepContext.for_run(trained.path("/outputs"), "/outputs", deps=trained.deps)).train_config
        for trained in (high_train, low_train)
    ]
    assert configurations[0].trainer.id != configurations[1].trainer.id
    assert configurations[0].initialize_from_hf == configurations[1].initialize_from_hf == PARENTS[parent_name].uri
    assert rows["arm"] == "plain"


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


def test_long_context_uses_matching_rows_and_cannot_resume_the_short_context_run():
    rows = manifest("v0.1", "plain")
    short = graph(rows, "v0.1")
    with pytest.raises(ValueError, match="manifest row cap"):
        graph(rows, "v0.1", context_length=40960)
    rows["max_sequence_tokens"] = 40960
    long = graph(rows, "v0.1", context_length=40960)
    configurations = [
        trained.build_config(StepContext.for_run(trained.path("/outputs"), "/outputs", deps=trained.deps)).train_config
        for trained in (short["train"], long["train"])
    ]
    assert [config.train_seq_len for config in configurations] == [32768, 40960]
    assert [config.adapter.sequence_length for config in configurations] == [32768, 40960]
    assert configurations[0].trainer.train_batch_size == configurations[1].trainer.train_batch_size == 16
    assert configurations[0].trainer.load_checkpoint_path != configurations[1].trainer.load_checkpoint_path
    assert {cache.path("/outputs") for cache in short["caches"]}.isdisjoint(
        cache.path("/outputs") for cache in long["caches"]
    )
    with pytest.raises(ValueError, match="manifest row cap"):
        graph(rows, "v0.1")

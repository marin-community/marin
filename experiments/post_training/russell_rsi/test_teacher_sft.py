# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace

import pytest
from rigging.runtime_bundle import RuntimeBundle

from experiments.post_training.russell_rsi.launch_teacher_sft import (
    TeacherCollectionConfig,
    require_teacher_condition,
    run_teacher_collection,
)


def test_teacher_condition_uses_frozen_dose_choice_and_blocks_inference_after_gain(tmp_path):
    parent = {"checkpoint_identity": "parent", "development": [25 / 32, 25 / 32], "retention": 1 / 3}
    four = {**parent, "checkpoint_identity": "four", "development": [24 / 32, 25 / 32]}
    eight = {**parent, "checkpoint_identity": "eight", "development": [26 / 32, 25 / 32]}
    path = tmp_path / "dose-selection.json"

    def save(selected):
        decision = {"parent": parent, "four": four, "eight": eight, "selected": selected}
        raw = json.dumps(decision).encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    config = TeacherCollectionConfig(
        selection={},
        dose_decision_uri=str(path),
        dose_decision_sha256=save(eight),
        bank_path="unread-bank",
        parent_path="unread-parent",
        parent_identity="parent",
        tokenizer_files={},
        runtime_bundle=RuntimeBundle("unread-manifest", "unused", "unread-archive", "unused"),
        relay_job="unresolved-relay",
        output_path=str(tmp_path / "teacher"),
    )
    with pytest.raises(ValueError, match="Dose improved the parent"):
        run_teacher_collection(config)
    assert not (tmp_path / "teacher").exists()

    tampered = replace(config, dose_decision_sha256=save(four))
    with pytest.raises(ValueError, match="frozen dose choice"):
        require_teacher_condition(tampered)

    eight["development"] = [24 / 32, 25 / 32]
    permitted = replace(config, dose_decision_sha256=save(four))
    assert require_teacher_condition(permitted)["selected"] == four

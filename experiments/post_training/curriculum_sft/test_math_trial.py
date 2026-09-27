# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The curriculum trial consumes prepared Parquet and evaluates Levanter's HF output."""

from marin.execution.lazy import materialized_config
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.curriculum_sft.math_trial import CURRICULUM_IDS, build_trial


def test_curriculum_sft_uses_prepared_chat_and_builtin_hf_save():
    trial = build_trial("2026.09.25")
    trained = trial["train"]
    train_config = materialized_config(trained, "s3://test-prefix").train_config

    assert len(trained.deps) == len(CURRICULUM_IDS)
    assert all(len(dep.deps) == 1 for dep in trained.deps)
    for capability_id, component in train_config.data.components.items():
        assert component.source.train_urls == [
            prefix_join(trained.deps[CURRICULUM_IDS.index(capability_id)].path("s3://test-prefix"), "*.parquet")
        ]
        assert component.format.pack is False

    assert train_config.hf_save_steps == train_config.trainer.num_train_steps
    assert train_config.hf_save_dtype.name == "bfloat16"
    assert trial["after"].deps == (trained,)
    after_config = materialized_config(trial["after"], "s3://test-prefix")
    assert after_config.model.location == prefix_join(trained.path("s3://test-prefix"), "hf")

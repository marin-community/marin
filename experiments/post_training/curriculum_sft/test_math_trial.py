# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The curriculum trial consumes prepared Parquet and evaluates Levanter's HF output."""

import pytest
from levanter.models.snowball import SnowballConfig
from marin.execution.lazy import materialized_config
from marin.experiment.namespacing import user_owned_name
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.curriculum_sft.math_trial import (
    CURRICULUM_IDS,
    HF_MODEL,
    HF_REVISION,
    S3_TRIAL_PREFIX,
    STEPS,
    build_trial,
)


@pytest.fixture(autouse=True)
def avoid_hf_config_fetch(monkeypatch):
    monkeypatch.setattr("marin.experiment.checkpoints.resolve_lm_config", lambda *_: SnowballConfig())


def test_curriculum_sft_uses_prepared_chat_and_builtin_hf_save():
    trial = build_trial("2026.09.25")
    trained = trial["train"]
    train_config = materialized_config(trained, "s3://test-prefix").train_config

    assert len(trained.deps) == len(CURRICULUM_IDS) + 1
    assert all(len(dep.deps) == 1 for dep in trained.deps[:-1])
    conversion = trained.deps[-1]
    conversion_config = materialized_config(conversion, "s3://test-prefix")
    assert conversion_config.tokenizer == f"hf://{HF_MODEL}@{HF_REVISION}"
    assert train_config.initialize_from_hf is False
    assert train_config.initialize_model_from_checkpoint_path == conversion.path("s3://test-prefix")
    assert train_config.pad_tokenizer_to_match_model is False
    for capability_id, component in train_config.data.components.items():
        assert component.source.train_urls == [
            prefix_join(trained.deps[CURRICULUM_IDS.index(capability_id)].path("s3://test-prefix"), "*.parquet")
        ]
        assert component.format.pack is False

    assert train_config.hf_save_steps == train_config.trainer.num_train_steps
    assert train_config.hf_save_dtype.name == "bfloat16"
    assert trial["after"].deps == (trained,)
    after_config = materialized_config(trial["after"], "s3://test-prefix")
    assert after_config.model.location == prefix_join(trained.path("s3://test-prefix"), f"hf/step-{STEPS - 1}")


def test_curriculum_sft_reuses_existing_generation_with_short_lived_outputs():
    trained = build_trial("2026.09.25.11")["train"]
    source_prefix = "s3://marin-us-east-02a/tmp/ttl=30d/curriculum-math-20260924"
    output_prefix = "s3://marin-us-east-02a/tmp/ttl=7d/curriculum-math-20260924"

    for capability_id, prepared in zip(CURRICULUM_IDS, trained.deps[:-1], strict=True):
        staged = prepared.deps[0]
        assert staged.path(S3_TRIAL_PREFIX) == prefix_join(
            source_prefix,
            user_owned_name(f"documents/curriculum-sft/{capability_id}/generated-chat/2026.09.24"),
        )
        assert prepared.path(S3_TRIAL_PREFIX).startswith(output_prefix)
    assert trained.path(S3_TRIAL_PREFIX).startswith(output_prefix)
    assert trained.deps[-1].path(S3_TRIAL_PREFIX).startswith(output_prefix)

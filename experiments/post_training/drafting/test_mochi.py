# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from marin.execution.lazy import ArtifactStep, artifact_identity

from experiments.post_training import dspark_sft
from experiments.post_training.drafting import mochi


def test_mochi_draft_sizes_share_prepared_data_and_teacher_activations():
    one, mix = dspark_sft._resolve_recipe(None, "mochi-agentic-50m", None, "/unused")
    two = replace(one, mochi_draft_size=mochi.DraftSize.TWO_LAYERS)
    teacher = dspark_sft._teacher_artifact(one.teacher)
    prepared_one, draft_one = dspark_sft.artifacts(one, teacher=teacher, mix=mix)
    prepared_two, draft_two = dspark_sft.artifacts(two, teacher=teacher, mix=mix)
    assert artifact_identity(prepared_one) == artifact_identity(prepared_two)
    assert artifact_identity(draft_one) != artifact_identity(draft_two)

    teacher_id = artifact_identity(teacher)
    one_recipe, two_recipe = one.training_recipe(), two.training_recipe()
    key = mochi.activation_cache_key(teacher_id, prepared_one, one_recipe, one.mochi_sample)
    assert key == mochi.activation_cache_key(teacher_id, prepared_two, two_recipe, two.mochi_sample)

    different_taps = replace(two_recipe, feature_taps=(2, 24, 46))
    assert key != mochi.activation_cache_key(teacher_id, prepared_two, different_taps, two.mochi_sample)
    another_mix = ArtifactStep.adopt("test/mix", "2026.11.24", "/unused/mix")
    prepared_other, _ = dspark_sft.artifacts(one, teacher=teacher, mix=another_mix)
    assert key != mochi.activation_cache_key(teacher_id, prepared_other, one_recipe, one.mochi_sample)

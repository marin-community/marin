# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import haliax as hax
import jax.random as jrandom
import numpy as np
from levanter.store.cache import write_levanter_cache
from marin.datakit.sft import SftTokenStore
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import ArtifactHfModel
from marin.training.training import LevanterCheckpoint

from experiments.post_training.bfcl_rl.collect import MODELS
from experiments.post_training.bfcl_rl.offline_optimize import offline_optimizer_step
from experiments.post_training.bfcl_rl.optimize import RECOVERY_CONTEXT


def test_offline_trainer_loads_curated_student_tokens_and_preserves_assistant_loss(tmp_path):
    root = str(tmp_path / "corpus")
    row = {
        "input_ids": np.asarray([1, 2, 3, 4, 5], dtype=np.int32),
        "assistant_masks": np.asarray([0, 0, 1, 1, 0], dtype=np.int32),
    }
    write_levanter_cache(iter([row]), f"{root}/student-store/train", metadata={})
    source = MODELS["student"]
    tokenizer = f"{source.model}@{source.revision}"
    store = SftTokenStore(
        path=root,
        cache_path=f"{root}/student-store/train",
        tokenizer=tokenizer,
        max_length=RECOVERY_CONTEXT,
        seed=42,
        sources={},
        packed_sequences=1,
    )
    write_record(
        ArtifactRecord(output_path=root, result_type=result_type_name(SftTokenStore), result=store.result_payload())
    )
    corpus = ArtifactStep.adopt("corpus", "2026.10.04", root, kind=SftTokenStore)
    student = ArtifactStep.adopt("student", "2026.10.04", str(tmp_path / "student"), kind=LevanterCheckpoint)
    policy = ArtifactHfModel(student, source.model, source.revision, relative_path="hf/step-57")
    with build_context(BuildContext(VersionCodex("2026.10.04"))):
        step = offline_optimizer_step(corpus, policy, num_train_steps=2)
    config = step.build_config(StepContext.for_run(str(tmp_path / "output"), str(tmp_path), deps=step.deps))
    data = replace(config.train_config.data, tokenizer="passthrough", vocab_size=16, shuffle=False)
    position = hax.Axis("position", RECOVERY_CONTEXT)
    assert data.validation_sets(position) == {}
    example = data.train_set(position, config.train_config.trainer.batch_schedule, key=jrandom.PRNGKey(0))
    example = example.as_sync_dataset()[0]
    np.testing.assert_array_equal(np.asarray(example.tokens.array)[:5], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(np.flatnonzero(np.asarray(example.loss_weight.array)), [1, 2])

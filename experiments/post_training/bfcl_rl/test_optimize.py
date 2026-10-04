# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from math import prod

import haliax as hax
import jax.random as jrandom
import numpy as np
import pytest
from levanter.main.train_dpo import _build_dpo_dataset, _build_validation_specs
from levanter.store.cache import write_levanter_cache
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, StepContext
from marin.training.training import LevanterCheckpoint

from experiments.post_training.bfcl_rl.collect import MODELS
from experiments.post_training.bfcl_rl.optimize import RECOVERY_CONTEXT, RecoveryOptimization, recovery_optimizer_step
from experiments.post_training.bfcl_rl.recovery_data import RecoveryPreferenceCache


def test_recovery_train_only_cache_loads_without_complement_validation(tmp_path, monkeypatch):
    # The live trainer attempted a validation ledger although recovery writes only train.
    root = str(tmp_path / "preferences")
    row = {
        "chosen_input_ids": np.asarray([1, 2, 3], dtype=np.int32),
        "chosen_assistant_masks": np.asarray([0, 1, 1], dtype=np.int32),
        "rejected_input_ids": np.asarray([1, 4, 5], dtype=np.int32),
        "rejected_assistant_masks": np.asarray([0, 1, 1], dtype=np.int32),
    }
    write_levanter_cache(iter([row]), f"{root}/train", metadata={})
    source = MODELS["student"]
    value = RecoveryPreferenceCache(
        path=root,
        num_preferences=1,
        tokenizer_uri=source.model,
        tokenizer_revision=source.revision,
        max_length=RECOVERY_CONTEXT,
        selection_manifest_uri=f"{root}/selection.json",
    )
    write_record(
        ArtifactRecord(
            output_path=root, result_type=result_type_name(RecoveryPreferenceCache), result=value.result_payload()
        )
    )
    cache = ArtifactStep.adopt("preferences", "2026.10.03", root, kind=RecoveryPreferenceCache)
    with build_context(BuildContext(VersionCodex("2026.10.03"))):
        step = recovery_optimizer_step(
            cache, selection_name="full", optimization=RecoveryOptimization(1, 16, 0.1, 8, 8, 4, 0.75)
        )
    # The initial model is remote; this test exercises only local preference data.
    monkeypatch.setattr(LevanterCheckpoint, "raw_load", lambda path: LevanterCheckpoint(path=path))
    config = step.build_config(StepContext.for_run(str(tmp_path / "output"), str(tmp_path), deps=step.deps))
    # Cached IDs bypass rendering; a local tokenizer/template keeps the read offline.
    component = config.train_config.data.components["bfcl_complement"]
    data = replace(
        config.train_config.data,
        tokenizer="passthrough",
        vocab_size=16,
        shuffle=False,
        components={
            "bfcl_complement": replace(
                component,
                format=replace(component.format, chat_template="{% generation %}{{ messages }}{% endgeneration %}"),
            )
        },
    )
    pos = hax.Axis("position", 8)
    assert _build_validation_specs(data, pos) == {}
    example = _build_dpo_dataset(data, pos, key=jrandom.PRNGKey(0)).as_sync_dataset()[0]
    np.testing.assert_array_equal(np.asarray(example.chosen.tokens.array)[:3], [1, 2, 3])
    np.testing.assert_array_equal(np.asarray(example.rejected.tokens.array)[:3], [1, 4, 5])


def test_recovery_mesh_fits_eight_gpu_nodes_and_preserves_batch_parallelism():
    # The live 64-GPU recovery run failed because preflight assumed one slice.
    cache = ArtifactStep.adopt("preferences", "2026.10.03.16", "preferences", kind=RecoveryPreferenceCache)
    with build_context(BuildContext(VersionCodex("2026.10.03.18"))):
        step = recovery_optimizer_step(
            cache, selection_name="full", optimization=RecoveryOptimization(1, 16, 0.1, 8, 8, 4, 0.75)
        )
    config = step.build_config(StepContext.for_fingerprint(deps=step.deps))
    mesh = config.train_config.trainer.mesh
    ici, dcn = mesh.axis_shapes(config.resources.chip_count(), config.resources.replicas)
    assert prod(ici.values()) == 8
    assert prod(dcn.values()) == 8
    assert ici["expert"] == 8
    assert dcn["context"] == 4
    batch_axes = mesh.resolved_compute_mapping["batch"]
    width = prod(ici.get(axis, 1) * dcn.get(axis, 1) for axis in batch_axes)
    assert width == config.train_config.trainer.train_batch_size == 16

    with build_context(BuildContext(VersionCodex("2026.10.03.18"))):
        with pytest.raises(ValueError, match="ICI product"):
            recovery_optimizer_step(
                cache, selection_name="full", optimization=RecoveryOptimization(1, 16, 0.1, 8, 16, 4, 0.75)
            )

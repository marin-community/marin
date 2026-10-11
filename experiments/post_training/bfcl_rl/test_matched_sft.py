# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import haliax as hax
import numpy as np
import pytest
from levanter.data.text.datasets import DatasetComponent
from levanter.data.text.preference import PreferenceChatLmDatasetFormat, PreferenceLmDataConfig
from levanter.main.train_dpo import _build_dpo_dataset, _derive_training_keys
from levanter.schedule import BatchSchedule
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import ArtifactHfModel
from marin.training.training import LevanterCheckpoint

from experiments.post_training.bfcl_rl.collect import MODELS
from experiments.post_training.bfcl_rl.data import PARTITION_MANIFEST_SHA256, BFCLPartition, TaskIdentity
from experiments.post_training.bfcl_rl.matched_sft_data import matched_sft_data_config, project_chosen_exposure
from experiments.post_training.bfcl_rl.matched_sft_optimize import matched_sft_optimizer_step
from experiments.post_training.bfcl_rl.optimize import RecoveryOptimization, recovery_optimizer_step
from experiments.post_training.bfcl_rl.recovery_data import RecoveryPreferenceCache, write_recovery_cache


@pytest.fixture
def source_cache(tmp_path):
    partition = BFCLPartition(
        "dataset-revision",
        tuple(TaskIdentity(f"task-{i}", f"source-{i}", f"digest-{i}") for i in range(3)),
        (TaskIdentity("holdout", "holdout-id", "holdout-digest"),),
    )
    rows = [
        {
            "chosen_input_ids": [1, 2, 3 + i, 8, 9, 10, 11, 12],
            "chosen_assistant_masks": [0, 0, 1, 1, 0, 0, 1, 1],
            "rejected_input_ids": [1, 2, 4 + i, 9],
            "rejected_assistant_masks": [0, 0, 1, 1],
        }
        for i in range(3)
    ]
    manifest = {
        "dataset_commit": partition.dataset_commit,
        "partition_manifest_sha256": PARTITION_MANIFEST_SHA256,
        "student_tokenizer": "passthrough@revision",
        "max_length": 16,
        "preferences": [
            {
                role: {"task_source_id": task.source_id, "task_digest": task.digest, "outcome": outcome}
                for role, outcome in (("chosen", "correct"), ("rejected", "incorrect"))
            }
            for task in partition.complement
        ],
    }
    path = tmp_path / "source"
    write_recovery_cache(rows, manifest, str(path))
    source = RecoveryPreferenceCache(
        path=str(path),
        num_preferences=3,
        tokenizer_uri="passthrough",
        tokenizer_revision="revision",
        max_length=16,
        selection_manifest_uri=str(path / "selection.json"),
    )
    return source, partition, rows, manifest


def test_chosen_projection_matches_real_dpo_reader_order_tokens_masks_and_attention(tmp_path, source_cache):
    source, partition, rows, _ = source_cache
    projected = project_chosen_exposure(source, partition, seed=42, presentations=8, output_path=str(tmp_path / "sft"))
    position = hax.Axis("position", 16)
    control = PreferenceLmDataConfig(
        tokenizer="passthrough",
        vocab_size=128,
        auto_build_caches=False,
        shuffle=True,
        components={
            "bfcl_complement": DatasetComponent(
                cache_dir=source.path + "/train",
                flat_cache=True,
                format=PreferenceChatLmDatasetFormat(
                    chat_template=MARIN_CHAT_TEMPLATE, pack=False, slice_strategy="raise"
                ),
            )
        },
    )
    data_key, *_ = _derive_training_keys(42)
    expected = _build_dpo_dataset(control, position, key=data_key).as_sync_dataset()
    data = matched_sft_data_config(projected.cache_path, "passthrough")
    data = replace(data, vocab_size=128)
    actual = data.train_set(position, BatchSchedule(4), key=data_key).as_sync_dataset()
    for index in range(8):
        left, right = expected[index].chosen, actual[index]
        np.testing.assert_array_equal(left.tokens.array, right.tokens.array)
        np.testing.assert_array_equal(left.loss_weight.array, right.loss_weight.array)
        key_position = hax.Axis("key_position", 16)
        np.testing.assert_array_equal(
            left.attn_mask.materialize(position, key_position).array,
            right.attn_mask.materialize(position, key_position).array,
        )
        np.testing.assert_array_equal(np.flatnonzero(right.loss_weight.array), [1, 2, 5, 6])
        row = rows[projected.row_indices[index]]
        np.testing.assert_array_equal(right.tokens.array[:8], row["chosen_input_ids"])
    assert projected.presentations == 8
    selection = json.loads((tmp_path / "sft/selection.json").read_text())
    assert [pair["chosen"]["task_source_id"] for pair in selection["preferences"]] == [
        partition.complement[index].source_id for index in projected.row_indices
    ]


def test_chosen_projection_rejects_holdout_with_valid_source_ledger(tmp_path, source_cache):
    source, partition, rows, manifest = source_cache
    manifest["preferences"][0]["chosen"].update(task_source_id="holdout-id", task_digest="holdout-digest")
    path = tmp_path / "holdout-source"
    write_recovery_cache(rows, manifest, str(path))
    source = source.model_copy(update={"path": str(path), "selection_manifest_uri": str(path / "selection.json")})
    output = tmp_path / "blocked"
    with pytest.raises(ValueError, match="outside the BFCL complement"):
        project_chosen_exposure(source, partition, seed=42, presentations=8, output_path=str(output))
    assert not output.exists()


def test_matched_optimizer_rejects_projection_from_a_different_control(tmp_path, source_cache):
    source, partition, _, _ = source_cache
    projected = project_chosen_exposure(source, partition, seed=42, presentations=16, output_path=str(tmp_path / "sft"))
    model = MODELS["student"]
    source = source.model_copy(
        update={"tokenizer_uri": model.model, "tokenizer_revision": model.revision, "max_length": 40960}
    )
    projected = projected.model_copy(
        update={
            "tokenizer": f"{model.model}@{model.revision}",
            "max_length": 40960,
            "source_cache_path": "different-control",
        }
    )
    for artifact in (source, projected):
        write_record(
            ArtifactRecord(
                output_path=artifact.path, result_type=result_type_name(type(artifact)), result=artifact.result_payload()
            )
        )
    cache = ArtifactStep.adopt("preferences", "2026.10.08", source.path, kind=RecoveryPreferenceCache)
    projection = ArtifactStep.adopt("projection", "2026.10.08", projected.path, kind=type(projected))
    policy = ArtifactHfModel(
        ArtifactStep.adopt("policy", "2026.10.08", str(tmp_path / "policy"), kind=LevanterCheckpoint),
        model.model,
        model.revision,
        relative_path="hf/step-57",
    )
    with build_context(BuildContext(VersionCodex("2026.10.08"))):
        control = recovery_optimizer_step(
            cache,
            initial_policy=policy,
            selection_name="native",
            optimization=RecoveryOptimization(1, 16, 0.1, 4e-6, 1, 16, 8, 16, 0.90),
        )
        step = matched_sft_optimizer_step(control, projection)
    context = StepContext.for_run(str(tmp_path / "output"), str(tmp_path), deps=step.deps)
    with pytest.raises(ValueError, match="does not derive from the configured DPO cache"):
        step.build_config(context)
    projected = projected.model_copy(update={"source_cache_path": source.path})
    write_record(
        ArtifactRecord(
            output_path=projected.path, result_type=result_type_name(type(projected)), result=projected.result_payload()
        )
    )
    context = StepContext.for_run(str(tmp_path / "output"), str(tmp_path), deps=step.deps)
    config = step.build_config(context)
    assert config.train_config.initialize_from_hf == str(tmp_path / "policy/hf/step-57")
    assert config.resources.chip_count() == 128

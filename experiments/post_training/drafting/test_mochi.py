# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
from datasets import Dataset
from marin.execution.lazy import ArtifactStep, artifact_identity, resolve
from marin.training.drafting.activation_store import activation_path, get_or_build_activation_cache
from marin.training.drafting.dspark_data import ARROW_DATASET_DIR, PREPARED_DIR, SPLITS_FILE, PreparedSplits
from marin.training.drafting.dspark_distillation import DSparkExport, TrainedDraft
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration
from safetensors.numpy import save_file

from experiments.post_training.drafting import dspark_sft, mochi
from vendor.skyrl.marinskyrl.model_manifest import HF_WEIGHT_INDEX_FILENAME, MODEL_MANIFEST_FILENAME, ModelManifest


def test_mochi_publishes_both_draft_sizes_from_one_activation_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    teacher_path, prepared_path = tmp_path / "teacher", tmp_path / "prepared"
    teacher_path.mkdir()
    teacher = ArtifactStep.adopt("test/teacher", "2026.11.24", str(teacher_path))
    prepared = ArtifactStep.adopt("test/prepared", "2026.11.24", str(prepared_path))
    dataset = Dataset.from_dict({"input_ids": [[1, 2], [3]]})
    data_path = prepared_path / PREPARED_DIR
    dataset.save_to_disk(str(data_path / ARROW_DATASET_DIR))
    expected_split = PreparedSplits.from_counts(1, 1)
    (data_path / SPLITS_FILE).write_text(json.dumps(asdict(expected_split)))

    config, _ = dspark_sft._resolve_recipe(None, "mochi-agentic-50m", None, str(tmp_path))
    recipe = config.training_recipe()
    key = mochi.activation_cache_key(artifact_identity(teacher), prepared, recipe, mochi.Sample.ALL)
    monkeypatch.setattr(mochi, "_activation_cache_root", lambda key, _prefix: StoragePath(str(tmp_path / key)))
    root = StoragePath(str(tmp_path / key))

    def fit(*, scratch, hidden_states, split, **_):
        assert split == expected_split
        assert activation_path(hidden_states, 0).read_bytes() == b"row 0"
        assert Dataset.load_from_disk(str(scratch / "data" / ARROW_DATASET_DIR))[0] == dataset[0]
        result = DSparkExport(
            steps=3, validation={}, trainer_packages={}, speculators_revision="", vllm_source_commit=""
        )
        checkpoint = scratch / "checkpoints" / "0"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}")
        (checkpoint / "training_state.json").write_text("{}")
        save_file({"weight": np.ones(1)}, str(checkpoint / "model.safetensors"))
        return TrainedDraft(checkpoint, ("config.json", "model.safetensors"), result)

    monkeypatch.setattr(mochi, "train_cached_draft", fit)

    get_or_build_activation_cache(
        root,
        reusable_for=Duration.from_hours(48),
        populate=lambda path: activation_path(path, 0).write_bytes(b"row 0"),
    )
    for version, size in (("2026.11.24", mochi.DraftSize.ONE_LAYER), ("2026.11.25", mochi.DraftSize.TWO_LAYERS)):
        recipe = replace(config, mochi_draft_size=size).training_recipe()
        draft = mochi.checkpoint_step(teacher, "", prepared, recipe, version, mochi.Sample.ALL)
        result = resolve(replace(draft, output_path_prefix=f"memory://{tmp_path.name}"))
        export = StoragePath(result.path) / result.hf_subdirectory
        manifest = ModelManifest.model_validate_json((export / MODEL_MANIFEST_FILENAME).read_text())
        assert result.source_identity == manifest.identity
        assert manifest.tokenizer_mode == "policy"
        assert {entry.path for entry in manifest.files} == {
            "config.json",
            "model.safetensors",
            HF_WEIGHT_INDEX_FILENAME,
        }
        assert not (export / "training_state.json").exists()
        index = json.loads((export / HF_WEIGHT_INDEX_FILENAME).read_text())
        assert index["weight_map"] == {"weight": "model.safetensors"}
        for entry in manifest.files:
            assert hashlib.sha256((export / entry.path).read_bytes()).hexdigest() == entry.sha256

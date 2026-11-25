# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict, replace

from datasets import Dataset
from marin.execution.lazy import ArtifactStep, artifact_identity, resolve
from marin.training.drafting.activation_store import activation_path, get_or_build_activation_cache
from marin.training.drafting.dspark_data import ARROW_DATASET_DIR, PREPARED_DIR, SPLITS_FILE, PreparedSplits
from marin.training.drafting.dspark_distillation import DSparkExport
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.post_training import dspark_sft
from experiments.post_training.drafting import mochi


def test_mochi_fits_both_draft_sizes_from_one_activation_cache(tmp_path, monkeypatch):
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
        return DSparkExport(steps=3, validation={}, trainer_packages={}, speculators_revision="", vllm_source_commit="")

    monkeypatch.setattr(mochi, "train_cached_and_export", fit)

    get_or_build_activation_cache(
        root,
        reusable_for=Duration.from_hours(48),
        populate=lambda path: activation_path(path, 0).write_bytes(b"row 0"),
    )
    for version, size in (("2026.11.24", mochi.DraftSize.ONE_LAYER), ("2026.11.25", mochi.DraftSize.TWO_LAYERS)):
        recipe = replace(config, mochi_draft_size=size).training_recipe()
        draft = mochi.checkpoint_step(teacher, "", prepared, recipe, version, mochi.Sample.ALL)
        resolve(draft)

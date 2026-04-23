# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
import types
from pathlib import Path

from marin.evaluation.eval_dataset_cache import (
    CacheManifest,
    HF_CACHE_LAYOUT_VERSION,
    load_eval_datasets_from_gcs,
    save_eval_datasets_to_gcs,
)
from marin.evaluation.evaluation_config import EvalTaskConfig


def test_save_eval_datasets_to_gcs_uploads_full_hf_cache_root(monkeypatch, tmp_path):
    cache_root = tmp_path / "local-hf-cache"
    gcs_path = tmp_path / "gcs-cache"

    def fake_load_dataset(*, cache_dir, **_kwargs):
        Path(cache_dir, "datasets-marker.txt").write_text("dataset\n")
        Path(os.environ["HF_HUB_CACHE"], "hub-marker.txt").write_text("hub\n")
        Path(os.environ["HF_MODULES_CACHE"], "modules-marker.txt").write_text("modules\n")
        return object()

    monkeypatch.setattr(
        "marin.evaluation.eval_dataset_cache.extract_datasets_from_tasks",
        lambda eval_tasks, log=None: {("cais/mmlu", "all")},
    )
    monkeypatch.setattr(
        "marin.evaluation.eval_dataset_cache.warm_task_metadata_cache",
        lambda eval_tasks, log=None: None,
    )
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    out = save_eval_datasets_to_gcs(
        eval_tasks=[EvalTaskConfig(name="mmlu", num_fewshot=5)],
        gcs_path=str(gcs_path),
        local_cache_dir=str(cache_root),
    )

    assert out == str(gcs_path)
    assert (gcs_path / "datasets" / "datasets-marker.txt").exists()
    assert (gcs_path / "hub" / "hub-marker.txt").exists()
    assert (gcs_path / "modules" / "modules-marker.txt").exists()

    manifest = CacheManifest.from_dict(json.loads((gcs_path / ".eval_datasets_manifest.json").read_text()))
    assert manifest.supports_full_offline_task_loading() is True
    assert manifest.cache_layout_version == HF_CACHE_LAYOUT_VERSION


def test_load_eval_datasets_from_gcs_syncs_full_cache_root(monkeypatch, tmp_path):
    source = tmp_path / "source-cache"
    source.mkdir()
    (source / "datasets").mkdir()
    (source / "hub").mkdir()
    (source / "modules").mkdir()
    (source / "datasets" / "datasets-marker.txt").write_text("dataset\n")
    (source / "hub" / "hub-marker.txt").write_text("hub\n")
    (source / "modules" / "modules-marker.txt").write_text("modules\n")

    manifest = CacheManifest(
        task_names=["mmlu"],
        cached_datasets=[("cais/mmlu", "all")],
        failed_datasets=[],
    )
    (source / ".eval_datasets_manifest.json").write_text(json.dumps(manifest.to_dict()))

    target_root = tmp_path / "worker-hf-home"
    monkeypatch.setenv("HF_HOME", str(target_root))

    loaded_manifest = load_eval_datasets_from_gcs(str(source))

    assert loaded_manifest is not None
    assert loaded_manifest.supports_full_offline_task_loading() is True
    assert (target_root / "datasets" / "datasets-marker.txt").exists()
    assert (target_root / "hub" / "hub-marker.txt").exists()
    assert (target_root / "modules" / "modules-marker.txt").exists()

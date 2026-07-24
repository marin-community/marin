# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from marin.evaluation.harbor_dataset import materialize_harbor_dataset


def test_materialize_harbor_dataset_downloads_hf_revision_as_local_tasks(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / ".gitattributes").write_text("*.gz filter=lfs")
    (snapshot / "task-one").mkdir()
    calls: list[dict] = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("marin.evaluation.harbor_dataset.snapshot_download", download)

    path = materialize_harbor_dataset(
        "hf://DCAgent2/terminal_bench_2",
        "main",
        tmp_path / "workdir",
    )

    assert path == Path(snapshot)
    assert calls == [
        {
            "repo_id": "DCAgent2/terminal_bench_2",
            "repo_type": "dataset",
            "revision": "main",
            "local_dir": str(tmp_path / "workdir" / "hf_dataset"),
            "cache_dir": str(tmp_path / "workdir" / "hf_cache"),
            "token": False,
        }
    ]
    assert not (snapshot / ".gitattributes").exists()

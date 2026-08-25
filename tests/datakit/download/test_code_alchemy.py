# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import marin.datakit.download.blob_id_download as blob_download
import marin.datakit.download.code_alchemy as code_alchemy
from marin.datakit.download.huggingface import HfRepoFile, HfRepoListing


def test_lists_all_shards_into_schema_safe_subset_partitions(monkeypatch, tmp_path: Path):
    cfg = code_alchemy.CodeAlchemyDownloadConfig(output_path=str(tmp_path))
    source_root = "datasets/open-alchemy/code-alchemy"
    source_paths = [
        f"{source_root}@{cfg.revision}/code-dev/train-00001-of-00252.parquet",
        f"{source_root}@{cfg.revision}/code-qa/train-00000-of-00031.parquet",
    ]
    listing = HfRepoListing(
        source_root=source_root,
        files={
            path: HfRepoFile(size=1_000_000_000, xet_hash=f"hash-{index}") for index, path in enumerate(source_paths)
        },
    )
    list_calls = []
    metadata_calls = []

    class Metadata:
        location = "https://cdn.example.test/code-dev.parquet"

    def list_files(**kwargs):
        list_calls.append(kwargs)
        return listing

    def resolve_file(url: str, **kwargs):
        metadata_calls.append((url, kwargs))
        return Metadata()

    monkeypatch.setattr(blob_download, "list_hf_repo_files", list_files)
    monkeypatch.setattr(blob_download, "get_hf_file_metadata", resolve_file)
    monkeypatch.setattr(blob_download, "get_token", lambda: "hf-test-token")

    tasks = code_alchemy.list_code_alchemy_tasks(cfg)

    assert list_calls == [
        {
            "hf_dataset_id": "open-alchemy/code-alchemy",
            "revision": code_alchemy.HF_REVISION,
            "hf_urls_glob": ["*/*.parquet"],
            "token": "hf-test-token",
        }
    ]
    assert [task.relative_source_path for task in tasks] == [
        "code-dev/train-00001-of-00252.parquet",
        "code-qa/train-00000-of-00031.parquet",
    ]
    assert [task.output_path for task in tasks] == [
        str(tmp_path / "data" / "subset=code-dev"),
        str(tmp_path / "data" / "subset=code-qa"),
    ]
    assert all(task.dataset_id == code_alchemy.HF_DATASET_ID for task in tasks)
    assert all(task.revision == code_alchemy.HF_REVISION for task in tasks)
    assert all(not task.require_token for task in tasks)
    assert metadata_calls == [
        (
            (
                "https://huggingface.co/datasets/open-alchemy/code-alchemy/"
                f"resolve/{code_alchemy.HF_REVISION}/code-dev/train-00001-of-00252.parquet"
            ),
            {"token": "hf-test-token", "retry_on_errors": True},
        )
    ]

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from experiments.domain_phase_mix import delphi_tpp40_evaluation_identity as identity


@dataclass
class FakeTreeFilesystem:
    objects: dict[str, tuple[int, str]]

    def find(self, root: str) -> list[str]:
        return sorted(path for path in self.objects if path.startswith(root.rstrip("/") + "/"))

    def info(self, path: str) -> dict[str, int | str]:
        size, crc32c = self.objects[path]
        return {"size": size, "crc32c": crc32c}


def test_tree_payload_identity_can_exclude_executor_bookkeeping(monkeypatch: pytest.MonkeyPatch) -> None:
    filesystem = FakeTreeFilesystem(
        {
            "bucket/result/.executor_info": (20, "executor-v1"),
            "bucket/result/results.json": (100, "result-v1"),
        }
    )
    monkeypatch.setattr(identity.fsspec, "get_fs_token_paths", lambda _: (filesystem, None, ["bucket/result"]))

    before = identity.tree_payload_identity(
        "gs://bucket/result",
        excluded_relative_paths=(".executor_info",),
    )
    filesystem.objects["bucket/result/.executor_info"] = (20, "executor-v2")
    after_bookkeeping_change = identity.tree_payload_identity(
        "gs://bucket/result",
        excluded_relative_paths=(".executor_info",),
    )
    filesystem.objects["bucket/result/results.json"] = (101, "result-v2")
    after_result_change = identity.tree_payload_identity(
        "gs://bucket/result",
        excluded_relative_paths=(".executor_info",),
    )

    assert before == after_bookkeeping_change
    assert before["payload_sha256"] != after_result_change["payload_sha256"]

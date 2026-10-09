# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Packed Harbor filenames and identities shared by exporters and artifact consumers."""

import hashlib
import io
import json
import tarfile
from dataclasses import dataclass, field

TASKS_FILENAME = "tasks.parquet"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class HarborSourceMetadata:
    name: str
    atlas_id: str
    family: str


@dataclass
class VerifierPayloadIdentity:
    """Identify emitted verifier files, recipes and dispatch configuration without building them."""

    _payloads: set[str] = field(default_factory=set, init=False)

    def add(self, task_binary: bytes) -> None:
        files = []
        with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:*") as archive:
            for member in archive:
                if not member.isfile() or not (member.name.startswith("tests/") or member.name == "task.toml"):
                    continue
                content = archive.extractfile(member)
                assert content is not None
                files.append((member.name, member.mode, hashlib.sha256(content.read()).hexdigest()))
        self._payloads.add(hashlib.sha256(json.dumps(sorted(files)).encode()).hexdigest())

    @property
    def ref(self) -> str:
        return "sha256:" + hashlib.sha256(json.dumps(sorted(self._payloads)).encode()).hexdigest()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The dependency-free wire record shared with the isolated Atlas process."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenAtlasRequest:
    source_id: str
    name: str
    version: str
    repo: str
    revision: str | None
    files: list[str]
    output_path: str
    mode: str
    input_root: str | None
    input_files: dict[str, str]
    auxiliary: dict[str, str]

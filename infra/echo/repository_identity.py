# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable identities for repository files in Echo's shared corpus."""

from dataclasses import dataclass

import repository_files
import search_config


@dataclass(frozen=True)
class RepositoryFileReference:
    target: search_config.RepositoryTarget
    path: str

    @property
    def result_id(self) -> str:
        return f"file:{self.target.repository}@{self.target.branch}:{self.path}"

    @property
    def route_value(self) -> str:
        return self.result_id.removeprefix("file:")


def repository_file_reference(
    target: search_config.RepositoryTarget,
    path: str,
) -> RepositoryFileReference:
    normalized = repository_files.repository_path(path)
    if normalized is None:
        raise ValueError("repository file path must be a safe relative POSIX path")
    return RepositoryFileReference(target, str(normalized))


def parse_repository_file_id(value: str) -> RepositoryFileReference:
    if not value.startswith("file:"):
        raise ValueError("repository file ID must start with file:")
    detail = value.removeprefix("file:")
    for target in search_config.REPOSITORY_TARGETS:
        prefix = f"{target.repository}@{target.branch}:"
        if detail.startswith(prefix):
            return repository_file_reference(target, detail.removeprefix(prefix))
    raise ValueError("repository file ID must name a configured repository and branch")


def configured_repository_target(repository: str) -> search_config.RepositoryTarget:
    for target in search_config.REPOSITORY_TARGETS:
        if target.repository == repository:
            return target
    choices = ", ".join(target.repository for target in search_config.REPOSITORY_TARGETS)
    raise ValueError(f"unknown repository {repository!r}; choose one of: {choices}")

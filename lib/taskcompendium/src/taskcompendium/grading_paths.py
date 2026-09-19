# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map explicit agent paths into an exported candidate workspace."""

from pathlib import PurePosixPath

from taskcompendium.models import relative_path

EXTERNAL_DIRECTORY = "__external__"


def submission_relative(path: str, workdir: str = "/app", additional_directories: tuple[str, ...] = ()) -> str:
    candidate = PurePosixPath(path)
    if candidate.is_absolute():
        try:
            path = candidate.relative_to(workdir).as_posix()
        except ValueError as error:
            for root in additional_directories:
                if candidate == PurePosixPath(root) or candidate.is_relative_to(root):
                    relative_path(candidate.as_posix().lstrip("/"))
                    return f"{EXTERNAL_DIRECTORY}/{candidate.as_posix().lstrip('/')}"
            raise ValueError(f"Submission must be inside a declared filesystem root: {path}") from error
    if PurePosixPath(path).parts[:1] == (EXTERNAL_DIRECTORY,):
        raise ValueError("Submission uses a reserved snapshot directory")
    if path != ".":
        relative_path(path)
    return path

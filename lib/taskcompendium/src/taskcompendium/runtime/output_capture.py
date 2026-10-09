# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select bounded workspace files without granting access to private grading mounts."""

import fnmatch
from collections.abc import Mapping
from pathlib import PurePosixPath

from rigging.filesystem.path_validation import validate_relative_file_path

from taskcompendium.models import OutputDirectory

PRIVATE_ROOTS = (PurePosixPath("/tests"), PurePosixPath("/logs/verifier"), PurePosixPath("/solution"))
CAPTURE_METADATA_BYTES = 1_048_576
DIRECTORY_CAPTURE_PROBE = (
    "import os; d = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW); "
    "s = os.scandir(d); s.close(); os.close(d)"
)

# Runs inside the existing Shellbox machine. Directory descriptors prevent a
# background process from redirecting a selected path through a symlink between
# discovery and reading. scandir's depth-first order matches unsorted find -P.
DIRECTORY_CAPTURE_SCRIPT = """import base64, fnmatch, json, os, stat, sys
selection = json.loads(sys.argv[1])
per_file_limit = int(sys.argv[2])
root = selection['root']
directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

def visit(directory, prefix, files, total, metadata_bytes):
    with os.scandir(directory) as entries:
        for entry in entries:
            relative = prefix + entry.name
            if entry.is_dir(follow_symlinks=False):
                child = os.open(entry.name, directory_flags, dir_fd=directory)
                try:
                    total, metadata_bytes = visit(child, relative + '/', files, total, metadata_bytes)
                finally:
                    os.close(child)
            elif entry.is_file(follow_symlinks=False) and any(
                fnmatch.fnmatchcase(relative, pattern) for pattern in selection['patterns']
            ):
                if len(files) >= selection['max_files']:
                    raise RuntimeError('Output directory exceeds file count budget')
                descriptor = os.open(entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                                     dir_fd=directory)
                with os.fdopen(descriptor, 'rb') as stream:
                    info = os.fstat(stream.fileno())
                    limit = min(per_file_limit, selection['max_bytes'] - total)
                    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                        raise RuntimeError('Output directory contains an unbounded or non-regular file')
                    data = stream.read(limit + 1)
                if len(data) > limit:
                    raise RuntimeError('Output directory exceeds byte budget')
                path = root + '/' + relative
                metadata_bytes += len(json.dumps(path, ensure_ascii=True)) + 8
                if metadata_bytes > int(sys.argv[3]):
                    raise RuntimeError('Output directory filenames exceed capture metadata budget')
                files[path] = base64.b64encode(data).decode('ascii')
                total += len(data)
    return total, metadata_bytes

files = {}
directory = os.open('/', directory_flags)
try:
    try:
        for component in root.split('/')[1:]:
            child = os.open(component, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = child
    except FileNotFoundError:
        print('{}')
        sys.exit(0)
    visit(directory, '', files, 0, 0)
finally:
    os.close(directory)
print(json.dumps(files, ensure_ascii=True))
"""


def validate_output_directories(selections: tuple[OutputDirectory, ...], workspace: str) -> None:
    """Require each capture root to stay inside the declared workspace and outside private mounts."""
    base = PurePosixPath(workspace)
    if not base.is_absolute() or ".." in base.parts:
        raise ValueError("Capture requires an absolute workspace")
    for selection in selections:
        root = PurePosixPath(selection.root)
        if not root.is_relative_to(base) or any(
            root.is_relative_to(private) or private.is_relative_to(root) for private in PRIVATE_ROOTS
        ):
            raise ValueError(f"Output directory overlaps private mounts or escapes workspace: {selection.root}")


def selected_directory_files(selection: OutputDirectory, files: Mapping[str, bytes]) -> dict[str, bytes]:
    """Recheck submission membership and budgets before transferring files to the grader machine."""
    selected = {}
    total = 0
    root = PurePosixPath(selection.root)
    for path, data in files.items():
        candidate = PurePosixPath(path)
        if candidate == root or not candidate.is_relative_to(root):
            continue
        relative = candidate.relative_to(root).as_posix()
        if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in selection.patterns):
            continue
        validate_relative_file_path(path[1:])
        if any(candidate.is_relative_to(private) for private in PRIVATE_ROOTS):
            raise ValueError(f"Submission overlaps private grading files: {path}")
        selected[path] = data
        total += len(data)
        if len(selected) > selection.max_files or total > selection.max_bytes:
            raise RuntimeError(f"Output directory capture exceeds its budget: {selection.root}")
    return selected

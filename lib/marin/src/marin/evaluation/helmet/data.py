# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import posixpath
import tarfile
import tempfile
from dataclasses import dataclass

import fsspec

from marin.utils import fsspec_copy_path_into_dir

_SUCCESS_FILE = "_SUCCESS"
_DATA_DIRNAME = "data"
_METADATA_FILENAME = "metadata.json"


def _join_fs_path(path_a: str, path_b: str) -> str:
    if "://" in path_a:
        return posixpath.join(path_a, path_b)
    return os.path.join(path_a, path_b)


def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    abs_path = os.path.abspath(path)
    for member in tar.getmembers():
        target = os.path.abspath(os.path.join(path, member.name))
        if not target.startswith(abs_path + os.sep) and target != abs_path:
            raise RuntimeError(f"Refusing to extract path traversal member: {member.name}")
    tar.extractall(path=path)


@dataclass(frozen=True)
class HelmetDataDownloadConfig:
    output_path: str
    """Executor-provided output path (must be under `gcsfuse_mount/`)."""

    repo_id: str
    revision: str
    resolved_sha: str

    filename: str = "data.tar.gz"


def _clear_partial_output(*, fs_out: fsspec.AbstractFileSystem, out_root: str) -> None:
    """Remove only the artifacts we own, leaving executor sentinel files intact."""
    # We only remove our own payload + markers; executor sentinel files (e.g. `.executor_status`) remain.
    data_dir = _join_fs_path(out_root, _DATA_DIRNAME)
    if fs_out.exists(data_dir):
        fs_out.rm(data_dir, recursive=True)

    for filename in (_METADATA_FILENAME, _SUCCESS_FILE):
        path = _join_fs_path(out_root, filename)
        if fs_out.exists(path):
            fs_out.rm(path)


def _hf_url_for_repo_file(*, repo_id: str, revision: str | None, filename: str) -> str:
    # `hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<path/in/repo>`
    # For HELMET data, the repo is a dataset by default.
    if repo_id.startswith(("datasets/", "models/", "spaces/")):
        repo_path = repo_id
    else:
        repo_path = f"datasets/{repo_id}"

    url = f"hf://{repo_path}"
    if revision:
        url += f"@{revision}"
    url += f"/{filename}"
    return url


def download_helmet_data(config: HelmetDataDownloadConfig) -> None:
    """Download and extract HELMET data into the step output directory.

    For `gs://.../gcsfuse_mount/...` outputs, we write directly to GCS; TPU jobs will read via the gcsfuse mount.
    """
    fs_out, out_root = fsspec.core.url_to_fs(config.output_path)

    success_path = _join_fs_path(out_root, _SUCCESS_FILE)
    data_out_dir = _join_fs_path(out_root, _DATA_DIRNAME)
    metadata_path = _join_fs_path(out_root, _METADATA_FILENAME)

    fs_out.makedirs(out_root, exist_ok=True)

    if fs_out.exists(success_path):
        # Completed previously; confirm the payload directory exists.
        if fs_out.exists(data_out_dir):
            return
        fs_out.rm(success_path)

    # Avoid silently reusing partial outputs, but don't delete executor sentinel files in the root.
    _clear_partial_output(fs_out=fs_out, out_root=out_root)

    with tempfile.TemporaryDirectory(prefix="helmet_data_") as tmpdir:
        tar_dir = os.path.join(tmpdir, "tar")
        os.makedirs(tar_dir, exist_ok=True)

        tar_url = _hf_url_for_repo_file(repo_id=config.repo_id, revision=config.revision, filename=config.filename)
        fsspec_copy_path_into_dir(src_path=tar_url, dst_path=tar_dir)

        tar_path = os.path.join(tar_dir, config.filename)
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Expected tarball at {tar_path} after copying from {tar_url}")

        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            _safe_extract_tar(tar, extract_dir)

        extracted_data_dir = os.path.join(extract_dir, _DATA_DIRNAME)
        if not os.path.isdir(extracted_data_dir):
            raise FileNotFoundError(
                f"Expected HELMET archive to contain '{_DATA_DIRNAME}/' under {extract_dir}; "
                f"found: {sorted(os.listdir(extract_dir))}"
            )

        # Copy extracted `data/` to the step output path (gs://... or local).
        fsspec_copy_path_into_dir(src_path=extracted_data_dir, dst_path=data_out_dir, fs_out=fs_out)

    with fs_out.open(metadata_path, "w") as f:
        json.dump(
            {
                "repo_id": config.repo_id,
                "revision": config.revision,
                "resolved_sha": config.resolved_sha,
                "filename": config.filename,
                "source_url": _hf_url_for_repo_file(
                    repo_id=config.repo_id,
                    revision=config.revision,
                    filename=config.filename,
                ),
            },
            f,
            indent=2,
        )

    with fs_out.open(success_path, "w") as f:
        f.write("")

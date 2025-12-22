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
import shutil
import tarfile
import tempfile
from dataclasses import dataclass

from huggingface_hub import hf_hub_download

from marin.evaluation.utils import is_remote_path

_SUCCESS_FILE = "_SUCCESS"
_DATA_DIRNAME = "data"
_METADATA_FILENAME = "metadata.json"


def _local_path_for_gcsfuse_mount(output_path: str, *, local_mount_root: str = "/opt/gcsfuse") -> str:
    """Map an executor output path under `gcsfuse_mount/` to the local gcsfuse mount.

    Example:
      `gs://bucket/prefix/gcsfuse_mount/helmet-data-<sha>` -> `/opt/gcsfuse/helmet-data-<sha>`
    """
    marker = "gcsfuse_mount/"
    if marker not in output_path:
        raise ValueError(f"Expected output_path under {marker}, got: {output_path}")

    relative = output_path.split(marker, 1)[1].lstrip("/")
    return os.path.join(local_mount_root, relative)


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
    local_mount_root: str = "/opt/gcsfuse"

    hf_cache_dir: str = "/tmp/helmet-hf-cache"


def _clear_partial_output(local_out_dir: str) -> None:
    """Remove only the artifacts we own, leaving executor sentinel files intact."""
    data_dir = os.path.join(local_out_dir, _DATA_DIRNAME)
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    for filename in (_METADATA_FILENAME, _SUCCESS_FILE):
        path = os.path.join(local_out_dir, filename)
        if os.path.exists(path):
            os.remove(path)


def download_helmet_data(config: HelmetDataDownloadConfig) -> None:
    """Download and extract HELMET data into the gcsfuse-backed output directory."""
    if not is_remote_path(config.output_path):
        # Local runs can still use the same directory structure; no mapping needed.
        local_out_dir = config.output_path
    else:
        local_out_dir = _local_path_for_gcsfuse_mount(config.output_path, local_mount_root=config.local_mount_root)

    success_path = os.path.join(local_out_dir, _SUCCESS_FILE)
    os.makedirs(local_out_dir, exist_ok=True)

    data_out_dir = os.path.join(local_out_dir, _DATA_DIRNAME)
    if os.path.exists(success_path):
        # Completed previously; confirm the payload directory exists.
        if os.path.exists(data_out_dir):
            return
        os.remove(success_path)

    # Avoid silently reusing partial outputs, but don't delete executor sentinel files in the root.
    _clear_partial_output(local_out_dir)

    with tempfile.TemporaryDirectory(prefix="helmet_data_") as tmpdir:
        tar_path = hf_hub_download(
            repo_id=config.repo_id,
            repo_type="dataset",
            filename=config.filename,
            revision=config.revision,
            cache_dir=config.hf_cache_dir,
        )

        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            _safe_extract_tar(tar, extract_dir)

        extracted_data_dir = os.path.join(extract_dir, _DATA_DIRNAME)
        if not os.path.isdir(extracted_data_dir):
            raise FileNotFoundError(
                f"Expected HELMET archive to contain '{_DATA_DIRNAME}/' at {extracted_data_dir}; "
                f"found: {sorted(os.listdir(extract_dir))}"
            )
        shutil.copytree(extracted_data_dir, data_out_dir, dirs_exist_ok=True)

    with open(os.path.join(local_out_dir, _METADATA_FILENAME), "w") as f:
        json.dump(
            {
                "repo_id": config.repo_id,
                "revision": config.revision,
                "resolved_sha": config.resolved_sha,
                "filename": config.filename,
            },
            f,
            indent=2,
        )

    with open(success_path, "w") as f:
        f.write("")

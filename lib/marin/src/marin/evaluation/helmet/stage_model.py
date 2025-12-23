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
from dataclasses import dataclass

import fsspec

from marin.utils import fsspec_copy_path_into_dir

STAGED_MODEL_DIRNAME = "model"


@dataclass(frozen=True)
class StageModelToGcsfuseConfig:
    input_path: str
    output_path: str


def _looks_like_hf_model_id(path: str) -> bool:
    if "://" in path:
        return False
    parts = path.split("/")
    return len(parts) == 2 and all(part.strip() for part in parts)


def stage_model_to_gcsfuse(config: StageModelToGcsfuseConfig) -> None:
    """Copy a model directory to a gcsfuse-backed output directory.

    This is used to avoid repeated downloads onto local TPU disks for large checkpoints:
    once staged into `gs://$PREFIX/gcsfuse_mount/...`, TPU jobs can read via `/opt/gcsfuse_mount/...`.
    """
    fs_out, out_root = fsspec.core.url_to_fs(config.output_path)
    success_path = os.path.join(out_root, "_SUCCESS")
    staged_root = os.path.join(out_root, STAGED_MODEL_DIRNAME)

    if fs_out.exists(success_path):
        # A previous run completed successfully. Sanity check the staged subdir exists.
        if fs_out.exists(staged_root):
            return
        fs_out.rm(success_path)

    fs_out.makedirs(out_root, exist_ok=True)
    if fs_out.exists(staged_root):
        fs_out.rm(staged_root, recursive=True)
    fs_out.makedirs(staged_root, exist_ok=True)

    src_path = config.input_path
    # Convenience: allow passing HF model ids like `org/repo` for staging.
    # We only treat it as an HF id if it doesn't exist as a local path.
    if _looks_like_hf_model_id(src_path) and not os.path.exists(src_path):
        src_path = f"hf://{src_path}"

    fsspec_copy_path_into_dir(src_path=src_path, fs_out=fs_out, dst_path=staged_root)

    with fs_out.open(os.path.join(out_root, "metadata.json"), "w") as f:
        json.dump(
            {
                "input_path": config.input_path,
                "staged_subdir": STAGED_MODEL_DIRNAME,
            },
            f,
            indent=2,
        )
    with fs_out.open(success_path, "w") as f:
        f.write("")

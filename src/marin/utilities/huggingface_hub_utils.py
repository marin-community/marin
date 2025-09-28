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

"""
huggingface_hub_utils.py

Helpful functions for facilitating downloads/verification of datasets/artifacts hosted on the HuggingFace Hub.
"""

import os

import fsspec
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import GatedRepoError


def get_hf_dataset_urls(hf_dataset_id: str, revision: str, hf_url_globs: list[str]) -> list[str]:
    """Walk through Dataset Repo using the `hf://` fsspec built-ins."""
    # get the token from the environment
    hf_token = os.environ.get("HF_TOKEN")
    fs = fsspec.filesystem("hf", token=hf_token)

    # Check if Dataset is Public or Gated
    try:
        fs.info(f"hf://datasets/{hf_dataset_id}", revision=revision)
    except GatedRepoError as err:
        raise NotImplementedError(f"Unable to automatically download gated dataset `{hf_dataset_id}`") from err

    try:
        base_dir = f"hf://datasets/{hf_dataset_id}"
        if not hf_url_globs:
            # We get all the files using find
            files = fs.find(base_dir, revision=revision)
        else:
            files = []
            # Get list of files directly from HfFileSystem matching the pattern
            for hf_url_glob in hf_url_globs:
                pattern = os.path.join(base_dir, hf_url_glob)
                files += fs.glob(pattern, revision=revision)

        url_list = []
        for fpath in files:
            # Resolve to HF Path =>> grab URL
            resolved_fpath = fs.resolve_path(fpath)
            url_list.append(
                hf_hub_url(
                    resolved_fpath.repo_id,
                    resolved_fpath.path_in_repo,
                    revision=resolved_fpath.revision,
                    repo_type=resolved_fpath.repo_type,
                )
            )
    except Exception as err:
        raise ValueError(f"Unable to download dataset `{hf_dataset_id}`") from err

    return url_list

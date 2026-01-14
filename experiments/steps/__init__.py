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
Step wrappers for marin library functions.

This module contains step definitions that wrap library processing functions
from lib/marin. The library functions work with concrete types (strings),
while these step wrappers handle dependency resolution and executor integration.

Usage:
    from experiments.steps import upload_dir_to_hf, upload_path_to_hf

    # Upload a step's output:
    step = upload_dir_to_hf(name="my-upload", input_step=my_previous_step, repo_id="user/repo")

    # Upload a raw GCS/local path:
    step = upload_path_to_hf(name="my-upload", input_path="gs://bucket/path", repo_id="user/repo")

    # Convert a checkpoint to HuggingFace format:
    step = convert_checkpoint_to_hf_step(name="hf/my-model", checkpoint_path="gs://...", ...)

    # Download a pretokenized cache from HuggingFace:
    step = download_pretokenized_cache(output_cache_path_name="my-cache", hf_repo_id="user/repo", tokenizer="...")
"""

from experiments.steps.download_pretokenized import download_pretokenized_cache
from experiments.steps.hf_upload import upload_dir_to_hf, upload_path_to_hf
from experiments.steps.levanter_checkpoint import (
    convert_checkpoint_from_step,
    convert_checkpoint_to_hf_step,
)
from experiments.steps.log_probs import default_lm_log_probs, log_probs_from_path, log_probs_step
from experiments.steps.slice_cache import slice_cache
from experiments.steps.tokenize import (
    tokenize_from_paths,
    tokenize_from_step,
    tokenize_hf_dataset,
)

__all__ = [
    "convert_checkpoint_from_step",
    "convert_checkpoint_to_hf_step",
    "default_lm_log_probs",
    "download_pretokenized_cache",
    "log_probs_from_path",
    "log_probs_step",
    "slice_cache",
    "tokenize_from_paths",
    "tokenize_from_step",
    "tokenize_hf_dataset",
    "upload_dir_to_hf",
    "upload_path_to_hf",
]

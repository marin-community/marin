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
Step wrappers for uploading to Hugging Face Hub.

This module provides step definitions that wrap the library functions in
marin.export.hf_upload.
"""

from marin.execution import ExecutorStep, StepContext, step
from marin.export.hf_upload import UploadToHfConfig, upload_to_hf


@step(name="metadata/hf_uploads/{name}", fn=upload_to_hf)
def upload_dir_to_hf(
    ctx: StepContext,
    name: str,
    input_step: ExecutorStep,
    repo_id: str,
    repo_type: str = "dataset",
    token: str | None = None,
    private: bool = False,
    revision: str | None = None,
    commit_batch_size: str = "1GiB",
    **upload_kwargs: str,
):
    """
    Create a step that uploads a step's output to a Hugging Face repo.

    Args:
        ctx: Step context (automatically provided by @step decorator)
        name: Name for this upload step (used in output path)
        input_step: The step whose output to upload
        repo_id: The repo id to upload to (e.g. "username/repo_name")
        repo_type: The type of repo ("dataset", "model", etc.)
        token: Auth token (if not provided, uses default)
        private: Whether to create a private repo if it doesn't exist
        revision: Branch to upload to (if not provided, uses default)
        commit_batch_size: Maximum size of files to batch in a single commit
        **upload_kwargs: Additional kwargs for huggingface_hub.upload_folder
    """
    return UploadToHfConfig(
        input_path=ctx.require(input_step),
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        revision=revision,
        private=private,
        commit_batch_size=commit_batch_size,
        upload_kwargs=upload_kwargs,
    )


@step(name="metadata/hf_uploads/{name}", fn=upload_to_hf)
def upload_path_to_hf(
    ctx: StepContext,
    name: str,
    input_path: str,
    repo_id: str,
    repo_type: str = "dataset",
    token: str | None = None,
    private: bool = False,
    revision: str | None = None,
    commit_batch_size: str = "1GiB",
    **upload_kwargs: str,
):
    """
    Create a step that uploads a raw path (GCS, local) to a Hugging Face repo.

    Use this for uploading existing paths that aren't step outputs.
    For uploading step outputs, use upload_dir_to_hf instead.

    Args:
        ctx: Step context (automatically provided by @step decorator)
        name: Name for this upload step (used in output path)
        input_path: The GCS/local path to upload
        repo_id: The repo id to upload to (e.g. "username/repo_name")
        repo_type: The type of repo ("dataset", "model", etc.)
        token: Auth token (if not provided, uses default)
        private: Whether to create a private repo if it doesn't exist
        revision: Branch to upload to (if not provided, uses default)
        commit_batch_size: Maximum size of files to batch in a single commit
        **upload_kwargs: Additional kwargs for huggingface_hub.upload_folder
    """
    return UploadToHfConfig(
        input_path=input_path,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        revision=revision,
        private=private,
        commit_batch_size=commit_batch_size,
        upload_kwargs=upload_kwargs,
    )

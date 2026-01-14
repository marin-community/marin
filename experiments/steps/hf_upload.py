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
marin.export.hf_upload. It handles StepRef resolution for executor integration.
"""

from urllib.parse import urlparse

from marin.execution import ExecutorStep, StepContext, StepRef, step
from marin.export.hf_upload import UploadToHfConfig, upload_to_hf


def upload_dir_to_hf(
    input_path: str | StepRef | ExecutorStep,
    repo_id: str,
    repo_type: str = "dataset",
    token: str | None = None,
    certificate_path: str | None = None,
    private: bool = False,
    revision: str | None = None,
    commit_batch_size: str = "1GiB",
    **upload_kwargs: str,
) -> ExecutorStep:
    """
    Create a step that uploads a directory to a Hugging Face repo.

    This is a step wrapper that handles StepRef resolution. The actual upload
    logic is in marin.export.hf_upload.upload_to_hf.

    For local paths, it will use the huggingface_hub.upload_folder function.
    For GCS (or other fsspec paths), it will stream the files using
    preupload_lfs_files and/or upload_folder.

    Args:
        input_path: Path to upload. Can be:
            - A string path (local or GCS)
            - A StepRef from another step
            - An ExecutorStep (will be converted to StepRef)
        repo_id: The repo id to upload to (e.g. "username/repo_name")
        repo_type: The type of repo to upload to (e.g. "dataset", "model", etc.)
        token: The token to use for authentication (if not provided, uses default)
        certificate_path: Where to store the upload certificate (for idempotency).
            If not provided, a reasonable default will be used.
        private: Whether to create a private repo if it doesn't exist
        revision: The branch to upload to (if not provided, uses default branch)
        commit_batch_size: Maximum size of files to batch in a single commit
        **upload_kwargs: Additional kwargs passed to huggingface_hub.upload_folder

    Returns:
        ExecutorStep that performs the upload
    """
    # Convert ExecutorStep to StepRef
    if isinstance(input_path, ExecutorStep):
        input_path = StepRef(_step=input_path)

    # Generate certificate path if not provided
    if not certificate_path:
        if isinstance(input_path, StepRef):
            certificate_path = f"metadata/hf_uploads/{input_path._subpath or 'output'}"
        else:
            # This will drop the scheme (e.g., 'gs') and keep the path
            parsed = urlparse(input_path)
            path = parsed.path.lstrip("/")
            certificate_path = f"metadata/hf_uploads/{path}"

    @step(name=certificate_path, fn=upload_to_hf)
    def _upload(ctx: StepContext):
        # Resolve input_path if it's a StepRef
        if isinstance(input_path, StepRef):
            actual_input = ctx.require(input_path)
        else:
            actual_input = input_path

        return UploadToHfConfig(
            input_path=actual_input,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            revision=revision,
            private=private,
            commit_batch_size=commit_batch_size,
            upload_kwargs=upload_kwargs,
        )

    return _upload()

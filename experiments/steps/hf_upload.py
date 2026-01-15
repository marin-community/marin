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
Step definitions for uploading to Hugging Face Hub.
"""

from marin.execution import StepRef, deferred, output, step
from marin.export.hf_upload import UploadToHfConfig
from marin.export.hf_upload import upload_to_hf as _upload_to_hf

# Mark the library function as deferred
upload_to_hf = deferred(_upload_to_hf)


@step(name="metadata/hf_uploads/{name}")
def upload_to_hf_step(
    name: str,
    input_path: StepRef,
    repo_id: str,
    repo_type: str = "dataset",
    token: str | None = None,
    private: bool = False,
    revision: str | None = None,
    commit_batch_size: str = "1GiB",
    **upload_kwargs: str,
):
    """
    Upload data to a Hugging Face repo.

    Args:
        name: Name for this upload step (used in output path)
        input_path: StepRef to upload (from calling another step, or StepRef.hardcoded())
        repo_id: The repo id to upload to (e.g. "username/repo_name")
        repo_type: The type of repo ("dataset", "model", etc.)
        token: Auth token (if not provided, uses default)
        private: Whether to create a private repo if it doesn't exist
        revision: Branch to upload to (if not provided, uses default)
        commit_batch_size: Maximum size of files to batch in a single commit
        **upload_kwargs: Additional kwargs for huggingface_hub.upload_folder

    Usage:
        # Upload a step's output
        tokenized = tokenize_dataset()
        upload = upload_to_hf_step(
            name="fineweb_tokenized",
            input_path=tokenized,
            repo_id="my-org/fineweb-tokenized",
        )

        # Upload a hardcoded path
        upload = upload_to_hf_step(
            name="manual_upload",
            input_path=StepRef.hardcoded("gs://bucket/path"),
            repo_id="my-org/my-data",
        )
    """
    return upload_to_hf(
        UploadToHfConfig(
            input_path=input_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            revision=revision,
            private=private,
            commit_batch_size=commit_batch_size,
            upload_kwargs=upload_kwargs,
        )
    )

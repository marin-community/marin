# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create new Hugging Face transfer destinations with private visibility."""

from huggingface_hub import bucket_info, create_bucket, create_repo, parse_hf_uri, repo_exists
from huggingface_hub.errors import BucketNotFoundError, HfUriError

from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import TransferError


def prepare_hf_destination(url: str) -> None:
    """Create a missing HF bucket or repository privately before writing into it.

    Existing destinations retain their visibility.
    """
    if StoragePath(url).scheme != "hf":
        return

    try:
        uri = parse_hf_uri(url)
    except HfUriError as error:
        raise TransferError(f"Invalid HF destination: {error}") from error
    if uri.is_bucket:
        try:
            bucket_info(uri.id)
        except BucketNotFoundError:
            create_bucket(uri.id, private=True, exist_ok=True)
        return

    if not repo_exists(uri.id, repo_type=uri.type):
        create_repo(uri.id, repo_type=uri.type, private=True, exist_ok=True)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create new Hugging Face transfer destinations with private visibility."""

from huggingface_hub import bucket_info, create_bucket, create_repo, repo_exists
from huggingface_hub.errors import BucketNotFoundError

from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import TransferError


def ensure_private_hf_destination(url: str) -> None:
    """Create a missing HF bucket or repository privately before writing into it.

    Existing destinations retain their visibility.
    """
    path = StoragePath(url)
    if path.scheme != "hf":
        return

    if path.netloc in {"buckets", "datasets", "models", "spaces"}:
        if len(path.segments) < 2:
            raise TransferError(f"HF destination must name an owner and bucket or repository: {url}")
        owner, name = path.segments[:2]
        kind = path.netloc
    else:
        if not path.netloc or not path.segments:
            raise TransferError(f"HF destination must name an owner and repository: {url}")
        owner, name = path.netloc, path.segments[0]
        kind = "models"

    if kind != "buckets":
        name = name.split("@", 1)[0]
    identifier = f"{owner}/{name}"
    if kind == "buckets":
        try:
            bucket_info(identifier)
        except BucketNotFoundError:
            create_bucket(identifier, private=True, exist_ok=True)
        return

    repo_type = {"models": "model", "datasets": "dataset", "spaces": "space"}[kind]
    if not repo_exists(identifier, repo_type=repo_type):
        create_repo(identifier, repo_type=repo_type, private=True, exist_ok=True)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import io
import logging
import os
import tempfile
from dataclasses import dataclass

import fsspec
import humanfriendly
from fsspec.implementations.local import LocalFileSystem
from huggingface_hub import CommitOperationAdd, create_commit, upload_folder
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from rigging.filesystem import StoragePath
from rigging.timing import ExponentialBackoff, retry_with_backoff
from tqdm_loggable.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class UploadToHfConfig:
    input_path: str
    repo_id: str
    repo_type: str = "dataset"
    token: str | None = None
    revision: str | None = None

    upload_kwargs: dict[str, str] = dataclasses.field(default_factory=dict)
    """Will be passed to huggingface_hub.upload_folder"""
    private: bool = False
    commit_batch_size: str = "1GiB"
    small_file_limit: str = "5 MiB"
    """
    The size limit for small files. Files larger than this will be uploaded using lfs.
    """


def _actually_upload_to_hf(config: UploadToHfConfig):
    # Check if the repo exists
    api = HfApi()
    try:
        api.repo_info(config.repo_id, repo_type=config.repo_type)
    except RepositoryNotFoundError:
        # Create the repo if it doesn't exist
        api.create_repo(
            repo_id=config.repo_id,
            repo_type=config.repo_type,
            token=config.token,
            private=config.private,
        )

    logger.info(f"Uploading {config.input_path} to {config.repo_id}")

    # Upload the folder. For local paths, we want to upload the folder directly.
    # For fsspec paths, we want to stream the files using create_commit
    fs = fsspec.core.get_fs_token_paths(config.input_path, mode="rb")[0]

    if isinstance(fs, LocalFileSystem):
        # Local path, use upload_folder
        upload_folder(
            repo_id=config.repo_id,
            folder_path=config.input_path,
            repo_type=config.repo_type,
            token=config.token,
            revision=config.revision,
            # commit_batch_size=config.commit_batch_size,
            **config.upload_kwargs,
        )
    else:
        all_paths = [str(m) for m in StoragePath(os.path.join(config.input_path, "**")).glob()]
        tiny_files = []
        large_files: dict[str, int] = {}  # path -> size

        small_file_size = humanfriendly.parse_size(config.small_file_limit)

        for path in all_paths:
            info = fs.info(path)
            if info["type"] == "directory":
                continue

            # skip executor metadata files
            if path.endswith(".executor_info") or path.endswith(".executor_status"):
                continue

            size_ = info["size"]
            if size_ < small_file_size:
                tiny_files.append(path)
            else:
                large_files[path] = size_

        max_size_bytes = humanfriendly.parse_size(config.commit_batch_size)
        base_message = f"Commiting these files to the repo from {config.input_path}:\n"

        # Upload the large files using preupload_lfs_files
        if large_files:
            batch = []
            batch_bytes = 0
            commit_message = base_message
            total_bytes = sum(large_files.values())

            pbar = tqdm(total=total_bytes, desc="Uploading large files", unit="byte")

            for path, size_ in large_files.items():
                fileobj = fs.open(path, "rb")
                path_in_repo = os.path.relpath(path, config.input_path)
                logger.info(f"Uploading {path} to {config.repo_id}/{path_in_repo}")

                # HF is very picky about the type of fileobj we pass in, so we need to
                if not isinstance(fileobj, io.BufferedIOBase):
                    fileobj = _wrap_in_buffered_base(fileobj)

                batch.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=fileobj))
                batch_bytes += size_
                commit_message += f"- {path_in_repo}\n"

                if batch_bytes > max_size_bytes:
                    retrying_create_commit(
                        config.repo_id,
                        operations=batch,
                        commit_message=base_message,
                        token=config.token,
                        commit_description=commit_message,
                        repo_type=config.repo_type,
                        revision=config.revision,
                    )
                    pbar.update(batch_bytes)
                    batch = []
                    batch_bytes = 0
                    commit_message = base_message

            if batch:
                retrying_create_commit(
                    config.repo_id,
                    operations=batch,
                    commit_message=base_message,
                    token=config.token,
                    commit_description=commit_message,
                    repo_type=config.repo_type,
                    revision=config.revision,
                )
                pbar.update(batch_bytes)

        # Upload the small files using upload_folder
        if tiny_files:
            logger.info(f"Uploading {len(tiny_files)} small files to {config.repo_id}")
            with tempfile.TemporaryDirectory() as tmpdir:
                for path in tiny_files:
                    path_in_repo = os.path.relpath(path, config.input_path)
                    fs.get(path, os.path.join(tmpdir, path_in_repo))

                retrying_upload_folder(
                    folder_path=tmpdir,
                    repo_id=config.repo_id,
                    repo_type=config.repo_type,
                    token=config.token,
                    revision=config.revision,
                    commit_message=f"Uploading small files from {config.input_path}",
                    **config.upload_kwargs,
                )


def retrying_upload_folder(*args, **kwargs):
    return retry_with_backoff(
        lambda: upload_folder(*args, **kwargs),
        max_attempts=3,
        backoff=ExponentialBackoff(initial=2.0, maximum=30.0, factor=2.0),
        operation="upload_folder",
    )


def retrying_create_commit(*args, **kwargs):
    return retry_with_backoff(
        lambda: create_commit(*args, **kwargs),
        max_attempts=3,
        backoff=ExponentialBackoff(initial=2.0, maximum=30.0, factor=2.0),
        operation="create_commit",
    )


def _wrap_in_buffered_base(fileobj):
    """
    Wraps a file-like object in a BufferedIOBase object.
    This is necessary because HF's upload_folder function expects a BufferedIOBase object.
    """
    if isinstance(fileobj, io.BufferedIOBase):
        return fileobj
    else:
        return io.BufferedReader(fileobj)


if __name__ == "__main__":
    # dummy test

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test.txt"), "w") as f:
            f.write("Hello, world!")

        _actually_upload_to_hf(
            UploadToHfConfig(
                tmpdir,
                repo_id="dlwh/test_uploading_local",
                repo_type="dataset",
            )
        )

    # also test memory fs
    StoragePath("memory://foo/bar/test.txt").write_text("Hello, world!!!!!\nadad :-)")

    _actually_upload_to_hf(
        UploadToHfConfig(
            "memory://foo/", repo_id="dlwh/test_uploading_fsspec", repo_type="dataset", small_file_limit="0B"
        )
    )

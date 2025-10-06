# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import glob
import os

import braceexpand
import fsspec
from fsspec.asyn import AsyncFileSystem


def exists(url, **kwargs) -> bool:
    """Check if a file exists on a remote filesystem."""
    fs, path = fsspec.core.url_to_fs(url, **kwargs)
    return fs.exists(path)


def mkdirs(path):
    """Create a directory and any necessary parent directories."""
    fs, path = fsspec.core.url_to_fs(path)
    fs.makedirs(path, exist_ok=True)


def expand_glob(url):
    """
    Yield every URL produced by brace and glob expansion.

    Examples
    --------
    >>> list(expand_glob("s3://bucket/{2023,2024}/*.json"))
    ['s3://bucket/2023/a.json', 's3://bucket/2024/b.json', ...]
    """
    for candidate in braceexpand.braceexpand(url):
        fs, path = fsspec.core.url_to_fs(candidate)

        if glob.has_magic(path):
            proto = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
            for p in fs.glob(path):
                yield f"{proto}://{p}" if proto else p
        else:
            yield candidate


def remove(url, *, recursive=False, **kwargs):
    """Remove a file from a remote filesystem."""
    # TODO: better to use a STS deletion policy or job for this one.
    fs, path = fsspec.core.url_to_fs(url, **kwargs)

    fs.rm(path, recursive=recursive)


async def async_remove(url, *, recursive=False, **kwargs):
    """Remove a file from a remote filesystem."""
    fs, path = fsspec.core.url_to_fs(url, **kwargs)

    if isinstance(fs, AsyncFileSystem):
        return await fs._rm(path, recursive=recursive)
    else:
        fs.rm(path, recursive=recursive)


def join_path(lhs, rhs):
    """
    Join parts of a path together. Similar to plain old os.path.join except when there is a protocol in the rhs, it
    is treated as an absolute path. However, the lhs protocol and rhs protocol must match if the rhs has one.

    """

    lhs_protocol, lhs_rest = fsspec.core.split_protocol(lhs)
    rhs_protocol, rhs_rest = fsspec.core.split_protocol(rhs)

    if rhs_protocol is not None and lhs_protocol is not None and lhs_protocol != rhs_protocol:
        raise ValueError(f"Cannot join paths with different protocols: {lhs} and {rhs}")

    if rhs_protocol is not None:
        return rhs
    else:
        return os.path.join(lhs, rhs)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compression formats for file previews."""

_COMPRESSION_BY_SUFFIX = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".xz": "xz",
    ".lzma": "lzma",
}


def compression_for(name: str) -> str | None:
    """Return the fsspec compression name for a supported file suffix."""
    lower_name = name.lower()
    for suffix, compression in _COMPRESSION_BY_SUFFIX.items():
        if lower_name.endswith(suffix):
            return compression
    return None


def uncompressed_name(name: str) -> str:
    """Remove one supported compression suffix from a file name."""
    lower_name = name.lower()
    for suffix in _COMPRESSION_BY_SUFFIX:
        if lower_name.endswith(suffix):
            return name[: -len(suffix)]
    return name

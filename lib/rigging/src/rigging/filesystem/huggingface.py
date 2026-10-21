# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit filesystem schemes for Hugging Face buckets and repositories."""

from typing import Any

from fsspec.implementations.dirfs import DirFileSystem
from huggingface_hub import HfFileSystem

HF_SCHEMES = {"hf-bucket": "buckets", "hf-model": "models", "hf-dataset": "datasets"}
HF_PROTOCOLS = frozenset({"hf", *HF_SCHEMES})


def hf_url(url: str) -> str:
    """Translate an explicit storage scheme to the Hub's native URL syntax."""
    scheme, _, path = url.partition("://")
    if scheme in HF_SCHEMES:
        return f"hf://{HF_SCHEMES[scheme]}/{path}"
    return url


class HfBucketFileSystem(DirFileSystem):
    """Expose HF buckets as ``hf-bucket://owner/bucket/path``."""

    protocol = "hf-bucket"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(path="buckets", fs=HfFileSystem(**kwargs))


class HfModelFileSystem(HfFileSystem):
    """Expose HF model repositories as ``hf-model://owner/model/path``."""

    protocol = "hf-model"


class HfDatasetFileSystem(DirFileSystem):
    """Expose HF dataset repositories as ``hf-dataset://owner/dataset/path``."""

    protocol = "hf-dataset"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(path="datasets", fs=HfFileSystem(**kwargs))

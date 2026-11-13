"""In-memory Hugging Face model snapshots for Iris tests."""

import hashlib

import fsspec
import numpy as np
from safetensors.numpy import save

from cloud.iris.hf_model_cache import HuggingFaceSnapshot, HuggingFaceSnapshotFile


def memory_model_snapshot(root: str) -> HuggingFaceSnapshot:
    filesystem = fsspec.filesystem("memory")
    payloads = {
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "model.safetensors": save({"weight": np.arange(4, dtype=np.float32)}),
    }
    files = []
    for path, payload in payloads.items():
        with filesystem.open(f"{root}/{path}", "wb") as destination:
            destination.write(payload)
        files.append(HuggingFaceSnapshotFile(path=path, size=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    return HuggingFaceSnapshot(filesystem=filesystem, root=root, files=tuple(files))

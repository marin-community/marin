# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit export layout, tokenizer/template identity and frozen router biases with range reads."""

import argparse
import hashlib
import json

import ml_dtypes
import numpy as np
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.cruxeval_transfer.constants import PARENT_DESTINATION

MAX_HEADER_BYTES = 10_000_000
ROUTER_BIAS_SUFFIX = ".mlp.router.bias"
TENSOR_DTYPES = {"BF16": ml_dtypes.bfloat16, "F32": np.float32}


def read_header(path: StoragePath) -> tuple[int, dict]:
    filesystem, key = url_to_fs(str(path))
    length = int.from_bytes(filesystem.cat_file(key, start=0, end=8), "little")
    assert 0 < length < MAX_HEADER_BYTES
    return 8 + length, json.loads(filesystem.cat_file(key, start=8, end=8 + length))


def read_tensor(path: StoragePath, name: str, headers: dict) -> np.ndarray:
    if path not in headers:
        headers[path] = read_header(path)
    base, header = headers[path]
    entry = header[name]
    start, end = entry["data_offsets"]
    filesystem, key = url_to_fs(str(path))
    raw = filesystem.cat_file(key, start=base + start, end=base + end)
    return np.frombuffer(raw, dtype=TENSOR_DTYPES[entry["dtype"]]).reshape(entry["shape"])


def verify_export(export: StoragePath, parent: StoragePath) -> dict:
    """Verify metadata, shard bounds and router biases, without reading all model weights."""
    metadata = {
        name: json.loads((export / name).read_text())
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "model.safetensors.index.json",
        )
    }
    assert metadata["config.json"]["model_type"] == "grug_moe"
    template = (export / "chat_template.jinja").read_bytes()
    assert template == (parent / "chat_template.jinja").read_bytes()
    assert metadata["tokenizer.json"] == json.loads((parent / "tokenizer.json").read_text())
    weight_map = metadata["model.safetensors.index.json"]["weight_map"]
    total_bytes = 0
    observed = {}
    headers = {}
    for shard in sorted(set(weight_map.values())):
        path = export / shard
        base, header = read_header(path)
        headers[path] = (base, header)
        tensors = {name: entry for name, entry in header.items() if name != "__metadata__"}
        assert set(tensors) == {name for name, filename in weight_map.items() if filename == shard}
        offset = 0
        for start, end in sorted(entry["data_offsets"] for entry in tensors.values()):
            assert start == offset and end >= start
            offset = end
        assert path.size() == base + offset
        total_bytes += offset
        observed.update({name: entry["dtype"] for name, entry in tensors.items()})
    assert total_bytes == metadata["model.safetensors.index.json"]["metadata"]["total_size"]
    parent_index = json.loads((parent / "model.safetensors.index.json").read_text())["weight_map"]
    names = {name for name in parent_index if name.endswith(ROUTER_BIAS_SUFFIX)}
    assert names and names == {name for name in weight_map if name.endswith(ROUTER_BIAS_SUFFIX)}
    for name in sorted(names):
        before = read_tensor(parent / parent_index[name], name, headers)
        after = read_tensor(export / weight_map[name], name, headers)
        np.testing.assert_array_equal(before.astype(after.dtype), after)
    return {
        "prefix": str(export),
        "shards": len(set(weight_map.values())),
        "tensors": len(observed),
        "tensor_bytes": total_bytes,
        "dtypes": sorted(set(observed.values())),
        "template_sha256": hashlib.sha256(template).hexdigest(),
        "tokenizer_matches_parent": True,
        "generation_config": metadata["generation_config.json"],
        "router_bias_tensors": len(names),
        "router_bias_equal_after_export_dtype_cast": True,
        "verification": (
            "Shard headers, bounds, tokenizer/template and router biases; no full weight checksum or inference."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--parent", default=f"s3://{PARENT_DESTINATION}")
    args = parser.parse_args()
    report = verify_export(StoragePath(args.export), StoragePath(args.parent))
    print("EXPORT_VERIFIED " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read Snowball config and safetensors headers in east without weight payloads."""

import hashlib
import json
import struct

import fsspec

PREFIX = "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b-sft-s2-thinking/step-630/hf-bf16-vllm"


def main():
    fs, root = fsspec.core.url_to_fs(PREFIX)
    config_bytes = fs.cat_file(root + "/config.json")
    index_bytes = fs.cat_file(root + "/model.safetensors.index.json")
    index = json.loads(index_bytes)
    tensors, headers = {}, {}
    for filename in sorted(set(index["weight_map"].values())):
        assert "/" not in filename and filename.endswith(".safetensors")
        path = root + "/" + filename
        length_bytes = fs.cat_file(path, start=0, end=8)
        assert len(length_bytes) == 8
        size = struct.unpack("<Q", length_bytes)[0]
        assert 0 < size <= 1024 * 1024, "unexpected header size"
        raw = fs.cat_file(path, start=8, end=8 + size)
        assert len(raw) == size
        headers[filename] = {"bytes": size, "sha256": hashlib.sha256(raw).hexdigest()}
        for name, entry in json.loads(raw).items():
            if name == "__metadata__":
                continue
            assert name not in tensors and index["weight_map"][name] == filename
            start, end = entry["data_offsets"]
            assert 0 <= start < end
            tensors[name] = {"shape": entry["shape"], "dtype": entry["dtype"], "bytes": end - start}
    assert set(tensors) == set(index["weight_map"])
    assert sum(row["bytes"] for row in tensors.values()) == index["metadata"]["total_size"]
    result = {
        "prefix": PREFIX,
        "config": json.loads(config_bytes),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "metadata": index["metadata"],
        "headers": headers,
        "tensors": tensors,
        "tensor_count": len(tensors),
        "over_1GiB": {name: row for name, row in tensors.items() if row["bytes"] > 2**30},
        "payload_bytes_read": 0,
    }
    print("SNOWBALL_CHECKPOINT_HEADERS_PASS", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep Snowball's exported FP32 routers when the other weights use BF16."""

import argparse
import hashlib
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import equinox as eqx
import fsspec
import jax
import numpy as np
from levanter.compat.hf_checkpoints import _convert_to_hf_url
from levanter.models.snowball import SnowballLMHeadModel
from rigging.filesystem.s3_compat import configure_coreweave_s3


def extract(weights: str, revision: str, output: Path) -> None:
    """Read only router byte ranges from the pinned safetensors snapshot."""
    source = weights if "://" in weights else _convert_to_hf_url(weights, revision)
    fs, prefix = fsspec.core.url_to_fs(source)
    index = json.loads(fs.cat_file(prefix + "/model.safetensors.index.json"))
    names = sorted(name for name in index["weight_map"] if ".mlp.router." in name)

    def read(name):
        path = prefix + "/" + index["weight_map"][name]
        length = struct.unpack("<Q", cast(bytes, fs.cat_file(path, start=0, end=8)))[0]
        header = json.loads(fs.cat_file(path, start=8, end=8 + length))
        entry = header[name]
        assert entry["dtype"] == "F32", (name, entry["dtype"])
        start, end = entry["data_offsets"]
        raw = fs.cat_file(path, start=8 + length + start, end=8 + length + end)
        return name, np.frombuffer(cast(bytes, raw), dtype="<f4").reshape(entry["shape"]).copy()

    with ThreadPoolExecutor(4) as pool:
        arrays = dict(pool.map(read, names))
    assert arrays, "Snapshot has no Snowball router tensors"
    np.savez_compressed(output, **arrays, source_weights=weights, source_revision=revision)


def restore(
    model: SnowballLMHeadModel, path: Path, weights: str, revision: str
) -> SnowballLMHeadModel:
    """Restore authoritative router values without rounding them through BF16."""
    replacements = []
    with np.load(path) as archive:
        assert archive["source_weights"].item() == weights
        assert archive["source_revision"].item() == revision
        for index, block in enumerate(model.transformer.blocks):
            for suffix, original in (("weight", block.mlp.router), ("bias", block.mlp.router_bias)):
                value = archive[f"model.layers.{index}.mlp.router.{suffix}"]
                if suffix == "weight":
                    value = value.T
                assert value.dtype == np.float32 and value.shape == original.shape
                replacements.append(
                    jax.make_array_from_callback(
                        value.shape, original.sharding, lambda indices, array=value: array[indices], dtype=np.float32
                    )
                )
    return eqx.tree_at(
        lambda item: tuple(
            array for block in item.transformer.blocks for array in (block.mlp.router, block.mlp.router_bias)
        ),
        model,
        tuple(replacements),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    extract(args.weights, args.revision, args.output)
    print(json.dumps({"path": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()

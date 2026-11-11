# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare policy tensor names and shapes in θ_R's first and final checkpoints."""

import argparse
import hashlib
import json
import pickle

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def tensor_schema(uri: str) -> dict[str, list[int]]:
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "rb") as source:
        metadata = pickle.load(source)
    return {
        name: list(value.size)
        for name, value in metadata.state_dict_metadata.items()
        if isinstance(value, TensorStorageMetadata) and not name.startswith(("optimizer.", "chained_"))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    root = args.checkpoint_root.rstrip("/")
    first_uri = f"{root}/global_step_10/policy/.metadata"
    final_uri = f"{root}/global_step_125/policy/.metadata"
    first = tensor_schema(first_uri)
    final = tensor_schema(final_uri)
    if not first or first != final:
        missing = set(final) - set(first)
        extra = set(first) - set(final)
        mismatched = [name for name in first.keys() & final.keys() if first[name] != final[name]]
        raise ValueError(
            f"Checkpoint schema mismatch: missing={len(missing)} extra={len(extra)} shape={len(mismatched)}"
        )
    encoded = json.dumps(first, sort_keys=True, separators=(",", ":")).encode()
    print(
        json.dumps(
            {
                "status": "matched",
                "tensor_count": len(first),
                "schema_sha256": hashlib.sha256(encoded).hexdigest(),
                "first_metadata_uri": first_uri,
                "final_metadata_uri": final_uri,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

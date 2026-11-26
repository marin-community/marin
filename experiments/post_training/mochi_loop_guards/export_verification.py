# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a regional converted model through SkyRL's completion and shard checks.

Run directly in the SkyRL CPU environment. Disable read-ahead so shard verification
reads headers without fetching weight payloads:

    FSSPEC_S3_DEFAULT_CACHE_TYPE=none uv run --project vendor/skyrl --extra cpu \\
        python -m experiments.post_training.mochi_loop_guards.export_verification \\
        --model s3://.../policy --output /path/to/verification.json
"""

import argparse
import json

from marinskyrl.model_files import MODEL_MANIFEST_FILENAME
from marinskyrl.model_manifest import HF_WEIGHT_INDEX_FILENAME
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from skyrl_train.hf_model_io import verify_completed_hf_model_export


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Completed regional HF-format model directory")
    parser.add_argument("--output", required=True, help="Small verification receipt to write")
    args = parser.parse_args(argv)
    configure_coreweave_s3()
    verify_completed_hf_model_export(args.model)
    root = StoragePath(args.model)
    manifest = json.loads((root / MODEL_MANIFEST_FILENAME).read_text())
    index = json.loads((root / HF_WEIGHT_INDEX_FILENAME).read_text())
    receipt = {
        "model": args.model,
        "complete": True,
        "manifest_identity": manifest["identity"],
        "manifest_files": len(manifest["files"]),
        "indexed_shards": len(set(index["weight_map"].values())),
        "indexed_tensors": len(index["weight_map"]),
        "verification": "Native completion markers, manifest files, indexed shard headers and recorded sizes",
        "weight_payloads_rehashed": False,
    }
    StoragePath(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

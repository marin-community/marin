# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage the certified Snowball base HF copy into the evaluation region.

Deletion Tuning's Micro210 rows need an HF-format base checkpoint in the
evaluation region (the micro-eval launcher rejects models outside it). The
certified copy lives in the training region (08a), produced for the Exp3
initializer with per-file SHA-256s and a verified-copy manifest; this job
copies it into the eval region with the same verified-copy discipline, then
pins the sidecar digests and safetensors shard headers the launcher's
ModelBinding consumes.

Run inside Iris on the eval cluster:

    uv run python -m experiments.deletion_tuning.stage_base_hf
"""

import hashlib
import json
import logging
import struct

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.fsutil.verified_copy import verified_copy_prefix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stage_base_hf")

SOURCE_RECEIPT_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp3/evidence/initializer-hf/2026.11.15.32.source.json"
)
DESTINATION = "s3://marin-us-east-02a/marin/users/benfeuer/snowball-deletion-tuning/inputs/base-hf/2026.11.18.1"
STATUS_PREFIX = "s3://marin-us-east-02a/marin/users/benfeuer/snowball-deletion-tuning/inputs/base-hf/2026.11.18.1.copy-status"
WORKERS = 16

SIDECARS = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
)


def hash_shard_header(fs, key: str) -> tuple[int, str]:
    """Return (bytes, header sha256) for one safetensors file, reading only the header."""
    with fs.open(key, "rb", block_size=0) as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > 64 * 1024 * 1024:
            raise ValueError(f"safetensors header exceeds the metadata-read limit: {key}")
        header = stream.read(size)
    if len(header) != size:
        raise ValueError(f"short header read: {key}")
    return size, hashlib.sha256(header).hexdigest()


def main() -> None:
    configure_coreweave_s3()
    receipt = json.loads(StoragePath(SOURCE_RECEIPT_URI).read_bytes())
    source = receipt["model_ref"]
    if receipt["repo"] != "open-athena/snowball-67b-a2b-base-262k-qk175-skew8":
        raise ValueError(f"unexpected source repo: {receipt['repo']}")
    if receipt["revision"] != "058ecaf27b9e4f37219df221a51e7d490d58ec3d":
        raise ValueError(f"unexpected source revision: {receipt['revision']}")
    logger.info("verified copy %s -> %s", source, DESTINATION)
    result = verified_copy_prefix(source, DESTINATION, status_url=STATUS_PREFIX, workers=WORKERS)
    logger.info("copied %s files (%s bytes); manifest %s", result.total_files, result.total_bytes, result.manifest_url)

    destination = StoragePath(DESTINATION)
    entries = [str(entry) for entry in destination.ls()]
    shard_pins = []
    sidecar_pins = {}
    fs, _ = filesystem_for(DESTINATION)
    for entry in entries:
        name = entry.rsplit("/", 1)[-1]
        key = prefix_join(DESTINATION, name)
        info = fs.info(key)
        if name in SIDECARS:
            sidecar_pins[name] = hashlib.sha256(StoragePath(key).read_bytes()).hexdigest()
        elif name.endswith(".safetensors"):
            size, header_sha = hash_shard_header(fs, key)
            shard_pins.append({"path": name, "bytes": info["size"], "header_sha256": header_sha})
    shard_pins.sort(key=lambda pin: pin["path"])
    if len(sidecar_pins) != len(SIDECARS):
        raise ValueError(f"missing sidecars: {sorted(set(SIDECARS) - set(sidecar_pins))}")
    if not shard_pins:
        raise ValueError("no safetensors shards found at the destination")

    binding_inputs = {
        "hf_uri": DESTINATION,
        "repo": receipt["repo"],
        "revision": receipt["revision"],
        "source_receipt": SOURCE_RECEIPT_URI,
        "source_model_ref": source,
        "copy_manifest": result.manifest_url,
        "total_bytes": result.total_bytes,
        "sidecar_sha256": sidecar_pins,
        "shards": shard_pins,
    }
    receipt_uri = prefix_join(DESTINATION, "model-binding-inputs.json")
    StoragePath(receipt_uri).write_text(json.dumps(binding_inputs, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %s (%d shards)", receipt_uri, len(shard_pins))


if __name__ == "__main__":
    main()

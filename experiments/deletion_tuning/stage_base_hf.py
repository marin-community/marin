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

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.fsutil.verified_copy import verified_copy_prefix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stage_base_hf")

SOURCE_RECEIPT_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp3/evidence/initializer-hf/2026.11.15.32.source.json"
)
DESTINATION = "s3://marin-us-east-02a/marin/users/benfeuer/snowball-deletion-tuning/inputs/base-hf/2026.11.18.1"
STATUS_PREFIX = (
    "s3://marin-us-east-02a/marin/users/benfeuer/snowball-deletion-tuning/inputs/base-hf/2026.11.18.1.copy-status"
)
WORKERS = 16

SIDECARS = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
)

# The base repo @ 058ecaf carries no generation_config.json (verified against the
# HF tree); the export convention fills one from the model's IDs and standard
# sampling defaults. The scored eval sends explicit sampling parameters per
# request, so these defaults never reach the rollouts; the file exists to keep
# the launcher's binding-completeness check honest.
SYNTHESIZED_GENERATION_CONFIG = {
    "bos_token_id": 128000,
    "eos_token_id": [128001, 128009],
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
}


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


def main(dry_run: bool = False) -> None:
    configure_coreweave_s3()
    receipt = json.loads(StoragePath(SOURCE_RECEIPT_URI).read_bytes())
    if receipt["repo"] != "open-athena/snowball-67b-a2b-base-262k-qk175-skew8":
        raise ValueError(f"unexpected source repo: {receipt['repo']}")
    if receipt["revision"] != "058ecaf27b9e4f37219df221a51e7d490d58ec3d":
        raise ValueError(f"unexpected source revision: {receipt['revision']}")
    source = receipt["model_ref"]
    total_bytes = receipt["total_bytes"]
    files = receipt.get("files") or []
    logger.info(
        "dry-run" if dry_run else "verified copy",
    )
    logger.info(
        "source %s (%s @ %s, %s bytes, %d pinned files) -> %s",
        source,
        receipt["repo"],
        receipt["revision"],
        f"{total_bytes:,}",
        len(files),
        DESTINATION,
    )
    source_fs, _ = filesystem_for(source)
    names = sorted(entry.rsplit("/", 1)[-1] for entry in source_fs.ls(source))
    sidecars_found = [name for name in names if name in SIDECARS]
    shards_found = [name for name in names if name.endswith(".safetensors")]
    missing_sidecars = sorted(set(SIDECARS) - set(sidecars_found))
    logger.info(
        "source listing: %d sidecars, %d shards, %d files total; missing sidecars: %s",
        len(sidecars_found),
        len(shards_found),
        len(names),
        missing_sidecars,
    )
    if missing_sidecars != ["generation_config.json"]:
        raise ValueError(f"unexpected source sidecar set: missing {missing_sidecars}")
    if not shards_found:
        raise ValueError("source copy has no safetensors shards")
    if dry_run:
        logger.info("dry-run complete: no copy performed")
        return
    logger.info("verified copy %s -> %s", source, DESTINATION)
    result = verified_copy_prefix(source, DESTINATION, status_url=STATUS_PREFIX, workers=WORKERS)
    logger.info("copied %s files (%s bytes); manifest %s", result.total_files, result.total_bytes, result.manifest_url)
    if missing_sidecars:
        synthesized = {}
        for name in missing_sidecars:
            if name != "generation_config.json":
                raise ValueError(f"cannot synthesize sidecar {name}")
            payload = json.dumps(SYNTHESIZED_GENERATION_CONFIG, indent=2, sort_keys=True) + "\n"
            StoragePath(prefix_join(DESTINATION, name)).write_text(payload)
            synthesized[name] = hashlib.sha256(payload.encode()).hexdigest()
        logger.info("synthesized sidecars: %s", sorted(synthesized))

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
            _header_size, header_sha = hash_shard_header(fs, key)
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
        "synthesized_sidecars": sorted(synthesized) if missing_sidecars else [],
        "synthesis_note": (
            "generation_config.json synthesized per the export convention (bos 128000, eos union"
            " {128001, 128009}, do_sample/temperature/top_p fill-ins); the base repo @ 058ecaf has"
            " none. The scored eval sends explicit sampling parameters per request."
            if missing_sidecars
            else None
        ),
    }
    receipt_uri = prefix_join(DESTINATION, "model-binding-inputs.json")
    StoragePath(receipt_uri).write_text(json.dumps(binding_inputs, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %s (%d shards)", receipt_uri, len(shard_pins))


if __name__ == "__main__":
    import sys

    main(dry_run="--dry-run" in sys.argv)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror the two V7 comp-prog final shard directories across CoreWeave regions.

Copies every object under the inventory's comp-prog shard directories from the
source regional prefix to the target regional prefix at the same relative path,
streaming each object once and recording its size and sha256. Verifies object
counts per shard and writes a mirror manifest that the cohort merge pins.
"""

import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned

logger = logging.getLogger(__name__)

COMP_PROG_SOURCES = (
    "nemotron_sft_v3/competitive_programming_v2/competitive_coding_cpp",
    "nemotron_sft_v3/competitive_programming_v2/competitive_coding_python",
)


def copy_object(source_uri: str, target_uri: str) -> dict:
    digest = hashlib.sha256()
    size = 0
    with StoragePath(source_uri).open("rb") as reader, StoragePath(target_uri).open("wb") as writer:
        while chunk := reader.read(8 << 20):
            digest.update(chunk)
            size += len(chunk)
            writer.write(chunk)
    return {"path": target_uri, "sha256": digest.hexdigest(), "bytes": size}


def mirror_shard(shard_dir: str, source_root: str, target_root: str) -> dict:
    shard = StoragePath(shard_dir)
    entries = []
    for dirpath, _dirnames, filenames in shard.walk():
        for name in filenames:
            entries.append(str(dirpath / name))
    entries.sort()
    if not entries:
        raise ValueError(f"Source shard has no objects: {shard_dir}")
    relative = str(shard.relative_to(StoragePath(source_root)))
    if not relative:
        raise ValueError(f"Shard lies outside the source root: {shard_dir}")
    objects = []
    for source_uri in entries:
        target_uri = str(StoragePath(target_root) / relative / str(StoragePath(source_uri).relative_to(shard)))
        objects.append(copy_object(source_uri, target_uri))
    written = [object for object in objects if (StoragePath(object["path"])).isfile()]
    if len(written) != len(entries):
        raise ValueError(f"Mirror object count mismatch for {shard_dir}: {len(written)} of {len(entries)}")
    total = sum(object["bytes"] for object in objects)
    return {
        "shard": shard_dir,
        "mirror": str(StoragePath(target_root) / relative),
        "objects": len(objects),
        "bytes": total,
        "sha256s": {object["path"]: object["sha256"] for object in objects},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    inventory = json.loads(read_pinned(args.inventory, args.inventory_sha256))
    shard_dirs = []
    for source in inventory["sources"]:
        if source["source"] not in COMP_PROG_SOURCES:
            continue
        for shard in source["shards"]:
            shard_dirs.append(str(StoragePath(shard["cache_uri"]).parent))
    if len(shard_dirs) != 101:
        raise ValueError(f"Expected 101 comp-prog shard directories, found {len(shard_dirs)}")

    logger.info("Mirroring %d shards from %s to %s", len(shard_dirs), args.source_root, args.target_root)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda shard: mirror_shard(shard, args.source_root, args.target_root), shard_dirs))
    total_bytes = sum(result["bytes"] for result in results)
    total_objects = sum(result["objects"] for result in results)
    manifest = {
        "source_root": args.source_root,
        "target_root": args.target_root,
        "inventory_sha256": args.inventory_sha256,
        "shards": sorted(results, key=lambda result: result["shard"]),
        "total_shards": len(results),
        "total_objects": total_objects,
        "total_bytes": total_bytes,
    }
    blob = json.dumps(manifest, sort_keys=True).encode()
    output_path = output / "mirror-manifest.json"
    output_path.write_bytes(blob)
    destination = StoragePath(args.destination)
    (destination / "mirror-manifest.json").write_bytes(blob)
    receipt = {
        "mirror_manifest_sha256": hashlib.sha256(blob).hexdigest(),
        "mirror_manifest_uri": str(destination / "mirror-manifest.json"),
        "total_shards": len(results),
        "total_objects": total_objects,
        "total_bytes": total_bytes,
        "inventory_sha256": args.inventory_sha256,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    (destination / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

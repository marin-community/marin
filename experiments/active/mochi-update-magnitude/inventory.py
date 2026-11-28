# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dump checkpoint inventories for the Mochi update-magnitude measurement.

For every Megatron checkpoint: parse the DCP ``.metadata`` (no tensor IO) and
write a compact family preview plus, where the layout is confident, the
key mapping the measurement needs. For Levanter checkpoints: the manifest is
already compact; record shapes/chunks for the record.

Usage (in region):
  python experiments/active/mochi-update-magnitude/inventory.py --out s3://.../inventory
"""

from __future__ import annotations

import argparse
import json
import logging
import posixpath
import time

import fsspec
from megatron_map import build_mapping, preview
from readers import LevanterReader, MegatronInventory, configure_object_storage
from registry import CHECKPOINTS, LEVANTER, MEGATRON, OUTPUT_PREFIX

logger = logging.getLogger("mochi-update-magnitude")


def write_json(fs_root: str, name: str, payload) -> str:
    path = posixpath.join(fs_root, name)
    fs, _ = fsspec.url_to_fs(path)
    fs.makedirs(fs_root, exist_ok=True)
    with fs.open(path, "wb") as handle:
        handle.write(json.dumps(payload, indent=1, default=str).encode())
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=posixpath.join(OUTPUT_PREFIX, "inventory"))
    parser.add_argument("--only", nargs="*", default=None, help="checkpoint ids")
    args = parser.parse_args()

    configure_object_storage()

    registry.resolve_live_pairs()  # pin whatever consecutive pair the live rows still hold
    selected = CHECKPOINTS
    if args.only:
        selected = {cid: CHECKPOINTS[cid] for cid in args.only}

    levanter_manifests = {}
    for checkpoint in selected.values():
        try:
            _inventory_one(checkpoint, args.out, levanter_manifests)
        except FileNotFoundError as error:
            logger.warning("checkpoint missing (pruned or rolled): %s (%s)", checkpoint.id, error)
        except Exception as error:
            logger.warning("inventory failed for %s: %s", checkpoint.id, error)
    if levanter_manifests:
        path = write_json(args.out, "levanter-manifests.json", levanter_manifests)
        logger.info("wrote %s", path)


def _inventory_one(checkpoint, args_out, levanter_manifests):
    started = time.monotonic()
    if checkpoint.kind == LEVANTER:
        reader = LevanterReader(checkpoint)
        manifest = reader._manifest()
        levanter_manifests[checkpoint.id] = {
            "uri": checkpoint.uri,
            "arrays": [
                {"path": a["path"], "shape": a["shape"], "dtype": a["dtype"], "chunk_shape": a["chunk_shape"]}
                for a in manifest["arrays"]
            ],
        }
        logger.info("levanter manifest %s (%.1fs)", checkpoint.id, time.monotonic() - started)
    elif checkpoint.kind == MEGATRON:
        inventory = MegatronInventory(checkpoint)
        entries = inventory.entries()
        mapping, notes = build_mapping(entries)
        preview_payload = {
            "uri": checkpoint.uri,
            "key_count": len(entries),
            "preview": preview(entries),
            "mapping_notes": notes,
        }
        name = f"{checkpoint.id}-preview.json"
        path = write_json(args.out, name, preview_payload)
        mapping_path = write_json(args.out, f"{checkpoint.id}-mapping.json", mapping)
        logger.info(
            "megatron inventory %s: %d keys -> %s, mapping %s (%.1fs)",
            checkpoint.id,
            len(entries),
            path,
            mapping_path,
            time.monotonic() - started,
        )


if __name__ == "__main__":
    main()

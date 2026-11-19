# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage archived browser evidence in its storage region without changing Atlas admission."""

import argparse
import gzip
import hashlib
import inspect
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from infra.marina.apps.rl_data_catalog import load_reviews, load_rollouts, review_mapping
from infra.marina.apps.rl_data_catalog.review_mapping import original_batch_mapping
from infra.marina.apps.rl_data_catalog.role_projection import ManifestReference, manifest_bytes
from infra.marina.apps.rl_data_catalog.server import review_content
from infra.marina.apps.rl_data_catalog.server.restrictions import is_private


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", nargs=2, required=True, metavar=("URI", "SHA256"))
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    manifest = json.loads(manifest_bytes(ManifestReference(uri=args.manifest[0], sha256=args.manifest[1])))
    modules = {
        "load_reviews": load_reviews,
        "load_rollouts": load_rollouts,
        "review_mapping": review_mapping,
        "review_content": review_content,
    }
    for name, expected in manifest["code_hashes"].items():
        if hashlib.sha256(Path(inspect.getfile(modules[name])).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Regional evidence worker loaded another {name} implementation")
    selected = [entry for index, entry in enumerate(manifest["entries"]) if index % args.shards == args.shard]
    receipts = []
    root = StoragePath(args.output_root)
    for entry in selected:
        source = entry["source"]
        if is_private(source):
            receipts.append({"source_id": source["id"], "status": "private_always", "content_staged": False})
            continue
        if manifest["kind"] == "qc":
            reference = ManifestReference.model_validate(entry["artifact"])
            mapping_reference = None
            identity = reference.sha256
            if entry.get("map_original_batches"):
                artifact = json.loads(manifest_bytes(reference))["result"]
                records = ManifestReference(uri=artifact["records_path"], sha256=artifact["records_sha256"])
                rows = pq.read_table(io.BytesIO(manifest_bytes(records))).to_pylist()
                prior = manifest.get("prior_mapping_root")
                prior_path = StoragePath(prior) / f"{identity}.mapping.json" if prior else None
                if prior_path is not None and prior_path.exists():
                    prior_raw = prior_path.read_bytes()
                    if prior_path.read_bytes() != prior_raw:
                        raise ValueError("Prior immutable mapping changed during readback")
                    mapping = json.loads(prior_raw)
                else:
                    mapping = original_batch_mapping(rows, artifact["batch_receipts"], 8)
                    mapping.update(artifact=entry["artifact"], records=records.model_dump(mode="json"))
                mapping_raw = (json.dumps(mapping, sort_keys=True) + "\n").encode()
                mapping_path = root / f"{identity}.mapping.json"
                if mapping_path.exists():
                    if mapping_path.read_bytes() != mapping_raw:
                        raise ValueError("Choose a new mapping output prefix: existing seal differs")
                else:
                    mapping_path.write_bytes(mapping_raw)
                if mapping_path.read_bytes() != mapping_raw:
                    raise ValueError("Original-batch mapping readback failed")
                mapping_reference = ManifestReference(
                    uri=str(mapping_path), sha256=hashlib.sha256(mapping_raw).hexdigest()
                )
                print(
                    json.dumps(
                        {
                            "source_id": source["id"],
                            "mapping": mapping_reference.model_dump(mode="json"),
                            "mapped": mapping["mapped"],
                            "not_mapped": mapping["not_mapped"],
                        }
                    ),
                    flush=True,
                )
            staged = load_reviews.stage_content_reviews(reference, source, 8, mapping_reference)
        else:
            metadata, rows = load_rollouts.stage_rollouts(entry["measurement"], source, 8)
            staged = {"measurement": entry["measurement"], "metadata": metadata, "rows": rows}
            identity = entry["measurement"]["measurement_id"]
        payload = gzip.compress(
            json.dumps(staged, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(), mtime=0
        )
        path = root / f"{identity}.json.gz"
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"Regional output already names different evidence: {path}")
        else:
            path.write_bytes(payload)
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashlib.sha256(payload).hexdigest():
            raise ValueError(f"Regional evidence readback failed: {path}")
        receipt = {
            "source_id": source["id"],
            "identity": identity,
            "status": "complete",
            "bytes": len(payload),
            "reference": {"uri": str(path), "sha256": hashlib.sha256(payload).hexdigest()},
            "records": len(staged["items"] if manifest["kind"] == "qc" else staged["rows"]),
        }
        receipts.append(receipt)
        print(json.dumps({key: value for key, value in receipt.items() if key != "reference"}), flush=True)
    result = {
        "manifest": {"uri": args.manifest[0], "sha256": args.manifest[1]},
        "kind": manifest["kind"],
        "snapshot_id": manifest["snapshot_id"],
        "load_id": manifest["load_id"],
        "shard": args.shard,
        "shards": args.shards,
        "receipts": receipts,
    }
    path = root / f"index-{args.shard:02}.json"
    payload = (json.dumps(result, indent=2) + "\n").encode()
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(payload)
    if path.read_bytes() != payload:
        raise ValueError("Regional index readback failed")
    print(json.dumps({"complete": True, "index": str(path), "sha256": hashlib.sha256(payload).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the conserved SFT initializer's regional verified copy against its HF inventory.

Promoted from the ad-hoc ``regional-conservation-check.py`` worker script (its
``submit-conservation-check.py`` wrapper becomes the generic CPU submit path):

    … cpu-submit --cpu 8 --memory 32GB --disk 64GB -- \\
      python -m experiments.post_training.mochi_loop_guards conservation-check \\
      --source s3://marin-us-east-08a/marin/models/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1 \\
      --inventory s3://…/conservation/sft-400407-HF-inventory.json \\
      --output s3://…/conservation/sft-400407-verification-receipt.json

The registered contract for this checkpoint is fixed: 193 weight shards, all present in
the saved ``model.safetensors.index.json`` weight map.
"""

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

REGISTERED_WEIGHT_SHARDS = 193
MANIFEST_NAME = ".marinskyrl-model-manifest.json"
INDEX_NAME = "model.safetensors.index.json"


def verified_files(
    hf_inventory: Mapping,
    manifest: Mapping,
    listing: Mapping[str, Mapping],
    read_source_bytes: Callable[[str], bytes],
) -> list[dict]:
    """Compare the HF repository inventory against the regional manifest and file listing.

    Every registered file must match in size and SHA256; LFS-registered hashes stand in
    for rehashing the large weight shards. Raises on any mismatch.
    """
    manifest_files = {row["path"]: row for row in manifest["files"]}
    results = []
    for item in hf_inventory["siblings"]:
        name = item["path"]
        registered = manifest_files.get(name)
        if registered is None:
            results.append({"path": name, "HF_repository_metadata_only": True})
            continue
        assert name in listing, name
        assert listing[name]["size"] == registered["size"] == item["size"], name
        if item["lfs"]:
            assert item["lfs"]["sha256"] == registered["sha256"], name
        else:
            assert hashlib.sha256(read_source_bytes(name)).hexdigest() == registered["sha256"], name
        results.append(
            {
                "path": name,
                "size": item["size"],
                "sha256": registered["sha256"],
                "size_match": True,
                "hash_match": True,
                "hash_basis": (
                    "retained verified-copy/source manifest and HF LFS SHA256"
                    if name.endswith(".safetensors")
                    else "regional content read SHA256"
                ),
                "regional_ETag": listing[name].get("ETag"),
            }
        )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Regional verified-copy prefix of the initializer")
    parser.add_argument("--inventory", required=True, help="HF repository inventory JSON for the initializer")
    parser.add_argument("--output", required=True, help="Verification receipt to write")
    args = parser.parse_args(argv)
    configure_coreweave_s3()
    source = StoragePath(args.source)
    hf = json.loads(StoragePath(args.inventory).read_text())
    manifest = json.loads((source / MANIFEST_NAME).read_text())
    fs, prefix = filesystem_for(str(source))
    listing = {entry["name"].removeprefix(prefix + "/"): entry for entry in fs.ls(prefix, detail=True)}
    results = verified_files(hf, manifest, listing, lambda name: (source / name).read_bytes())
    assert sum(row["path"].endswith(".safetensors") for row in results) == REGISTERED_WEIGHT_SHARDS
    index = json.loads((source / INDEX_NAME).read_text())
    assert set(index["weight_map"].values()) == {row["path"] for row in results if row["path"].endswith(".safetensors")}
    receipt = {
        "complete": True,
        "HF_private": hf["private"],
        "HF_commit": hf["sha"],
        "source": str(source),
        "regional_manifest_identity": manifest["identity"],
        "common_files_verified": sum("hash_match" in row for row in results),
        "weight_shards_verified": REGISTERED_WEIGHT_SHARDS,
        "all_index_shards_present": True,
        "HF_repository_only_files": [row["path"] for row in results if row.get("HF_repository_metadata_only")],
        "source_weights_rehashed_this_run": False,
        "hash_basis": (
            "Existing regional verified-copy SHA256 inventory compared with HF LFS SHA256; current regional "
            "file sizes verified. Small model files rehashed directly."
        ),
        "files": results,
    }
    StoragePath(args.output).write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "files"}), flush=True)


if __name__ == "__main__":
    main()

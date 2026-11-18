# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge the V7 comp-prog final cohort into the V6 atlas-applied inventory.

The V7 candidate differs from V6 only in the two comp-prog sources. The other
forty-eight sources reuse the V6 atlas-applied final caches verbatim; this
merge records that reuse by carrying their entries byte-identically and
replacing only the two comp-prog entries with the V7 apply output.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned

logger = logging.getLogger(__name__)

COMP_PROG_SOURCES = (
    "nemotron_sft_v3/competitive_programming_v2/competitive_coding_cpp",
    "nemotron_sft_v3/competitive_programming_v2/competitive_coding_python",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-applied-inventory", required=True)
    parser.add_argument("--v6-applied-inventory-sha256", required=True)
    parser.add_argument("--v7-applied-inventory", required=True)
    parser.add_argument("--v7-applied-inventory-sha256", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--comp-prog-source-root", required=True)
    parser.add_argument("--comp-prog-target-root", required=True)
    parser.add_argument("--mirror-manifest", required=True)
    parser.add_argument("--mirror-manifest-sha256", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    v6 = json.loads(read_pinned(args.v6_applied_inventory, args.v6_applied_inventory_sha256))
    v7 = json.loads(read_pinned(args.v7_applied_inventory, args.v7_applied_inventory_sha256))
    v7_entries = {source["source"]: source for source in v7["sources"]}
    if set(v7_entries) != set(COMP_PROG_SOURCES):
        raise ValueError(f"The V7 apply inventory covers {sorted(v7_entries)}")
    if v6["tokenizer_sha256"] != v7["tokenizer_sha256"]:
        raise ValueError("The V6 and V7 inventories use different tokenizers")

    mirror = json.loads(read_pinned(args.mirror_manifest, args.mirror_manifest_sha256))
    if mirror["inventory_sha256"] != args.v7_applied_inventory_sha256:
        raise ValueError("The mirror manifest does not cover this apply inventory")
    if mirror["target_root"] != args.comp_prog_target_root or mirror["source_root"] != args.comp_prog_source_root:
        raise ValueError("The mirror manifest roots do not match the rewrite")

    def rewritten(uri: str) -> str:
        if not uri.startswith(args.comp_prog_source_root + "/"):
            raise ValueError(f"Comp-prog shard URI lies outside the source root: {uri}")
        return args.comp_prog_target_root + uri[len(args.comp_prog_source_root) :]

    def rewritten_entry(source: dict) -> dict:
        return {
            **source,
            "shards": [{**shard, "cache_uri": rewritten(shard["cache_uri"])} for shard in source["shards"]],
        }

    sources = []
    replaced = set()
    for source in v6["sources"]:
        if source["source"] in v7_entries:
            sources.append(rewritten_entry(v7_entries[source["source"]]))
            replaced.add(source["source"])
        else:
            sources.append(source)
    if replaced != set(COMP_PROG_SOURCES):
        raise ValueError(
            f"The V6 inventory is missing the comp-prog entries: {sorted(set(COMP_PROG_SOURCES) - replaced)}"
        )
    for name in COMP_PROG_SOURCES:
        v6_entry = next(source for source in v6["sources"] if source["source"] == name)
        if v6_entry["source_revision"] != v7_entries[name]["source_revision"]:
            raise ValueError(f"Source revision changed for {name}")

    cohort = {
        **v6,
        "sources": sources,
        "transforms": [
            *[row for row in v6["transforms"] if row.get("source") not in set(COMP_PROG_SOURCES)],
            *v7["transforms"],
        ],
        "v7_comp_prog_reuse": {
            "note": "The forty-eight unchanged sources reuse the V6 atlas-applied final caches verbatim",
            "v6_applied_inventory_sha256": args.v6_applied_inventory_sha256,
            "v7_applied_inventory_sha256": args.v7_applied_inventory_sha256,
        },
        "v7_comp_prog_mirror": {
            "note": "The two comp-prog sources are served from the 08a mirror of the 02a post-Atlas caches",
            "source_root": args.comp_prog_source_root,
            "target_root": args.comp_prog_target_root,
            "mirror_manifest_sha256": args.mirror_manifest_sha256,
            "mirror_manifest_uri": args.mirror_manifest,
            "mirror_total_shards": mirror["total_shards"],
            "mirror_total_objects": mirror["total_objects"],
            "mirror_total_bytes": mirror["total_bytes"],
        },
    }
    blob = json.dumps(cohort, sort_keys=True, indent=2).encode()
    destination = StoragePath(args.destination)
    (destination / "cohort-inventory.json").write_bytes(blob)
    (output / "cohort-inventory.json").write_bytes(blob)
    receipt = {
        "cohort_sha256": hashlib.sha256(blob).hexdigest(),
        "sources": len(sources),
        "replaced": sorted(replaced),
        "rows": {name: v7_entries[name]["counts"]["output_documents"] for name in COMP_PROG_SOURCES},
        "tokens": {name: v7_entries[name]["counts"]["output_tokens"] for name in COMP_PROG_SOURCES},
        "comp_prog_target_root": args.comp_prog_target_root,
        "mirror_manifest_sha256": args.mirror_manifest_sha256,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    (destination / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

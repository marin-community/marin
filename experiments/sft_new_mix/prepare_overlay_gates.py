# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join pinned payloads to complete source shards, then apply the actual-call gate."""

import argparse
import gzip
import hashlib
import json
import logging
from collections import Counter, defaultdict

import ijson
from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.exp3_runtime import configure_storage, implementation_identity, read_pinned
from experiments.sft_new_mix.finalize_shards import FINALIZE_IMPLEMENTATION_FILES, execute_plan
from experiments.sft_new_mix.tool_overlay import export_key


def pinned_json(pin: dict) -> dict:
    return json.loads(read_pinned(pin["uri"], pin["sha256"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipe-sha256", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    recipe = pinned_json({"uri": args.recipe, "sha256": args.recipe_sha256})
    inventory = pinned_json(recipe["inventory"])
    scripted_plan = pinned_json(recipe["scripted_plan"])
    descriptor = pinned_json(recipe["admission_descriptor"])
    selected_sources = set(recipe["sources"])
    sources = {source["source"]: source for source in inventory["sources"] if source["source"] in selected_sources}
    if set(sources) != selected_sources or any(not source["complete"] for source in sources.values()):
        raise ValueError("Overlay gates require every selected source to be complete")
    if any(source["loss_mode"] != "chat_assistant_only" or source["teacher_only"] for source in sources.values()):
        raise ValueError("This replacement cohort requires ordinary chat supervision")
    input_jobs = {job["input_path"]: job for job in scripted_plan["jobs"]}
    for name, source in sources.items():
        if source["expected_shards"] != sum(job["source"] == name for job in scripted_plan["jobs"]):
            raise ValueError("Selected source does not cover the frozen scripted input population")
    allowed = set()
    for part in descriptor["parts"] if selected_sources.intersection(descriptor["by_source"]) else []:
        with StoragePath(part["manifest"]).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != part["sha256"]:
                raise ValueError("Canonical admission part digest changed")
        with StoragePath(part["manifest"]).open("rb") as stream, gzip.GzipFile(fileobj=stream) as decoded:
            for line in decoded:
                row = json.loads(line)
                if row["source"] in selected_sources:
                    allowed.add(export_key(row))
    excluded = {export_key(row) for row in descriptor["excluded_exports"]}
    replacements = {}
    counts = Counter()
    manifest_pins = []
    for manifest in recipe["tool_manifests"]:
        # These immutable producer outputs acquire their byte pin in this preparation.
        with StoragePath(manifest["uri"]).open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        if manifest["pin_mode"] == "expected":
            if sha256 != manifest["sha256"]:
                raise ValueError("Pinned payload manifest digest changed")
        elif manifest["pin_mode"] != "freeze_on_prepare":
            raise ValueError("Payload manifest requires an explicit pin mode")
        manifest_pins.append({"uri": manifest["uri"], "sha256": sha256})
        read_count = 0
        with StoragePath(manifest["uri"]).open("rb") as stream:
            for row in ijson.items(stream, manifest["record_path"], use_float=True):
                read_count += 1
                if row["source"] not in selected_sources:
                    continue
                if export_key(row) in excluded:
                    counts["excluded_export"] += 1
                    continue
                if row["priority"] == 2:
                    if export_key(row) not in allowed:
                        raise ValueError("Deterministic replacement is absent from the canonical allowlist")
                elif row["priority"] != 1 or row["snapshot_sha256"] != recipe["historical_snapshot_sha256"]:
                    raise ValueError("Historical replacement lacks its independent frozen snapshot")
                key = (row["source_shard"], int(row["source_row"]))
                if key[0] not in input_jobs or input_jobs[key[0]]["source"] != row["source"]:
                    raise ValueError("Replacement belongs to a different frozen source population")
                if key in replacements:
                    previous = replacements[key]
                    if previous["input_ids_sha256"] != row["input_ids_sha256"]:
                        raise ValueError("Payload manifests disagree on the original source row")
                    if previous["priority"] == row["priority"]:
                        if previous["export_manifest"]["files"] != row["export_manifest"]["files"]:
                            raise ValueError("Equal-priority payload manifests conflict")
                        counts["identical_payload_duplicates"] += 1
                        continue
                    counts["historical_payload_superseded"] += 1
                    if previous["priority"] > row["priority"]:
                        continue
                replacements[key] = row
        if read_count != manifest["expected_records"]:
            raise ValueError("Payload manifest row count differs from its frozen selection")
    by_shard = defaultdict(list)
    counts["selected_replacements"] = len(replacements)
    for (shard, _), row in replacements.items():
        by_shard[shard].append(row)
    jobs = []
    for name, source in sorted(sources.items()):
        for shard in source["shards"]:
            jobs.append(
                {
                    "source": name,
                    "base_shard": shard,
                    "loss_mode": source["loss_mode"],
                    "teacher_only": source["teacher_only"],
                    "passed": input_jobs[shard["original_input_path"]]["passed"],
                    "output_prefix": str(
                        StoragePath(recipe["output_prefix"]) / "components" / hashlib.sha256(name.encode()).hexdigest()
                    ),
                    "implementation_sha256": implementation_identity(FINALIZE_IMPLEMENTATION_FILES),
                    "overlays": by_shard.pop(shard["original_input_path"], []),
                    "overlay_enabled": True,
                    "real_tools_enabled": True,
                    "drop_rows": {},
                    "enabled_drop_stages": [],
                    "tokenizer_uri": inventory["tokenizer_uri"],
                    "tokenizer_sha256": inventory["tokenizer_sha256"],
                    **recipe["filter_config"],
                }
            )
    counts["replacements_in_fully_filtered_shards"] = sum(map(len, by_shard.values()))
    plan = {"jobs": jobs, "output_prefix": recipe["output_prefix"], "scope": recipe["scope"]}
    root = StoragePath(recipe["output_prefix"])
    content = json.dumps(plan, sort_keys=True).encode()
    (root / "plan.json").write_bytes(content)
    (root / "preparation.json").write_text(
        json.dumps(
            {
                "recipe_uri": args.recipe,
                "recipe_sha256": args.recipe_sha256,
                "plan_sha256": hashlib.sha256(content).hexdigest(),
                "tool_manifest_pins": manifest_pins,
                "counts": dict(counts),
                "jobs": len(jobs),
            },
            sort_keys=True,
        )
    )
    execute_plan(plan, recipe["workers"])


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and verify the final admission delta entirely in the target region."""

import argparse
import gzip
import json
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from rigging.filesystem.storage_path import StoragePath

from experiments.sft_new_mix.exp3_runtime import configure_storage, implementation_identity, read_pinned
from experiments.sft_new_mix.job_manifests import stage_job_manifest
from experiments.sft_new_mix.prepare_overlay_gates import pinned_json
from experiments.sft_new_mix.run_tool_payloads import JOB_MANIFEST_WRITE_WORKERS, execute_references
from experiments.sft_new_mix.tool_overlay import VERIFIER_IMPLEMENTATION_FILES, export_key, select_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipe-sha256", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    recipe = pinned_json({"uri": args.recipe, "sha256": args.recipe_sha256})
    descriptor = pinned_json(recipe["descriptor"])
    previous = pinned_json(recipe["previous_descriptor"])
    if descriptor["complete_population"] is not True or descriptor["pending_repairs"]:
        raise ValueError("Final descriptor must certify a complete population without pending repairs")
    if descriptor["source_selection_sha256"] != previous["source_selection_sha256"]:
        raise ValueError("Descriptor delta changed the frozen source selection")
    scripted = pinned_json(recipe["scripted_plan"])
    inputs = {(job["source"], job["input_path"]) for job in scripted["jobs"]}
    excluded = {export_key(row) for row in descriptor["excluded_exports"]}
    old_parts = {part["sha256"] for part in previous["parts"]}
    parts = [part for part in descriptor["parts"] if part["sha256"] not in old_parts]
    if implementation_identity(VERIFIER_IMPLEMENTATION_FILES) != recipe["verifier_sha256"]:
        raise ValueError("Final payload verifier changed")
    counts = Counter()
    by_source = Counter()
    archives = set()
    identities = set()
    references = []
    for part in parts:
        content = read_pinned(part["manifest"], part["sha256"])
        records, evidence = select_records([(gzip.decompress(content), 2)], inputs, excluded)
        if evidence["counts"]["manifest_rows"] != part["verified_and_actually_modified"] or evidence["superseded"]:
            raise ValueError("Final admission part has an unexpected population")
        counts.update(evidence["counts"])
        grouped = defaultdict(list)
        for row in records:
            identity = (row["source_shard"], int(row["source_row"]))
            if identity in identities:
                raise ValueError("Final delta parts repeat an original source row")
            identities.add(identity)
            grouped[(row["archive_uri"], row["archive_sha256"])].append(row)
            by_source[row["source"]] += 1
        jobs = []
        for (uri, sha256), rows in grouped.items():
            if sha256 in archives:
                raise ValueError("Final delta parts repeat an archive")
            archives.add(sha256)
            jobs.append(
                {
                    **recipe["tokenizers"],
                    "archive_uri": uri,
                    "archive_sha256": sha256,
                    "records": rows,
                    "output_prefix": recipe["output_prefix"],
                    "implementation_sha256": recipe["verifier_sha256"],
                }
            )
        with ThreadPoolExecutor(max_workers=JOB_MANIFEST_WRITE_WORKERS) as pool:
            references.extend(pool.map(stage_job_manifest, jobs))
        logging.info("Staged %d admitted rows from %s", len(records), part["manifest"])
    root = StoragePath(recipe["output_prefix"])
    (root / "job-references.json").write_text(json.dumps(references, sort_keys=True))
    (root / "preparation.json").write_text(
        json.dumps(
            {
                "recipe_uri": args.recipe,
                "recipe_sha256": args.recipe_sha256,
                "counts": dict(counts),
                "by_source": dict(by_source),
                "parts": parts,
                "archives": len(archives),
            },
            sort_keys=True,
        )
    )
    execute_references(references, recipe, recipe["output_prefix"], recipe["workers"], recipe["coordinator_memory"])


if __name__ == "__main__":
    main()

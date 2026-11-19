# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count original environment-lock references in complete, region-local populations."""

import argparse
import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.calibration_inventory import packages_lock_exists
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.published_census import population_rows

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CensusInput:
    source_id: str
    population: SourceTextPopulation
    expected_count: int
    expected_scanned_sha256: str


def count_references(entry: CensusInput, output_root: str) -> dict:
    """Verify complete retained bytes before publishing aggregate reference counts."""
    digest = hashlib.sha256()
    seen = set()
    tasks_by_lock = Counter()
    roles_by_lock = {"actor": Counter(), "grader": Counter()}
    for row in population_rows(entry.population):
        digest.update(row["task_spec"].encode())
        task = json.loads(row["task_spec"])
        task_id = task["id"]
        if task_id in seen:
            raise ValueError(f"{entry.source_id}: duplicate retained task ID {task_id}")
        seen.add(task_id)
        references = {
            "actor": (task.get("environment_requirements") or {}).get("packages_lock"),
            "grader": ((task.get("grader") or {}).get("environment") or {}).get("packages_lock"),
        }
        tasks_by_lock.update({uri for uri in references.values() if uri is not None})
        for role, uri in references.items():
            if uri is not None:
                roles_by_lock[role][uri] += 1
    if len(seen) != entry.expected_count or digest.hexdigest() != entry.expected_scanned_sha256:
        raise ValueError(f"{entry.source_id}: retained count or original serialized population digest changed")
    result = {
        "source_id": entry.source_id,
        "population": entry.population.model_dump(mode="json"),
        "verified_retained_tasks": len(seen),
        "verified_scanned_task_specs_sha256": digest.hexdigest(),
        "count_basis": "complete_retained_population",
        "locks": [
            {
                "uri": uri,
                "tasks": count,
                "actor_tasks": roles_by_lock["actor"][uri],
                "grader_tasks": roles_by_lock["grader"][uri],
                "original_artifact_available": packages_lock_exists(uri),
            }
            for uri, count in sorted(tasks_by_lock.items())
        ],
    }
    target = StoragePath(output_root) / hashlib.sha256(entry.source_id.encode()).hexdigest() / "census.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("Verified %s: %d retained tasks, %d lock references", entry.source_id, len(seen), len(tasks_by_lock))
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--population-prefix", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    data = StoragePath(args.manifest).read_bytes()
    if hashlib.sha256(data).hexdigest() != args.manifest_sha256:
        raise ValueError("Frozen census manifest differs from its digest")
    entries = [
        CensusInput(
            source_id=row["source_id"],
            population=SourceTextPopulation.model_validate(row["population"]),
            expected_count=row["expected_count"],
            expected_scanned_sha256=row["expected_scanned_sha256"],
        )
        for row in json.loads(data)
    ]
    if not entries or len({entry.source_id for entry in entries}) != len(entries):
        raise ValueError("Census requires a nonempty set of distinct sources")
    if any(not entry.population.specs_output_dir.startswith(args.population_prefix) for entry in entries):
        raise ValueError("Refuse raw population reads outside the explicitly selected storage region")
    output = StoragePath(args.output_root)
    if output.exists():
        raise FileExistsError(output)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        sources = list(pool.map(partial(count_references, output_root=args.output_root), entries))
    (output / "census.json").write_text(
        json.dumps(
            {
                "manifest": args.manifest,
                "manifest_sha256": args.manifest_sha256,
                "population_prefix": args.population_prefix,
                "sources": sources,
                "verified_retained_tasks": sum(source["verified_retained_tasks"] for source in sources),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

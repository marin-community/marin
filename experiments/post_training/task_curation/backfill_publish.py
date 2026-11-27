# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish one backfilled population as a private open-athena release with a read-back receipt.

Issue #11976 step 3: upload the requirement-backfilled TaskSpecs as a NEW private dataset
revision (the published source revision is never touched) through the licensed
publication chokepoint: the card declares the parent pin's resolved upstream, every
byte is read back, and the new pin's receipt registers before the receipt is recorded.
HF credentials come from the job environment; the release stays private.
"""

import argparse
import hashlib
import json
import logging
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.license_publication import publish_licensed_population

LOGGER = logging.getLogger(__name__)

TASK_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("path", pa.string()),
        ("task_spec", pa.large_string()),
        ("task_sha256", pa.string()),
        ("runtime", pa.large_string()),
    ]
)


def read_population(population: str) -> list[dict]:
    rows = []
    shards = sorted(str(path) for path in (StoragePath(population) / "specs/*.parquet").glob())
    if not shards:
        raise ValueError(f"The backfilled population holds no spec shards: {population}")
    for shard in shards:
        with StoragePath(shard).open("rb") as stream:
            table = pq.read_table(stream, columns=["id", "task_spec", "runtime"])
        ids = table.column("id").to_pylist()
        specs = table.column("task_spec").to_pylist()
        runtimes = table.column("runtime").to_pylist()
        for identifier, spec, runtime in zip(ids, specs, runtimes, strict=True):
            rows.append(
                {
                    "id": identifier,
                    "path": "",
                    "task_spec": spec,
                    "task_sha256": hashlib.sha256(spec.encode()).hexdigest(),
                    "runtime": runtime or "",
                }
            )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"The backfilled population repeats a task id: {population}")
    return rows


def dataset_card(source_id: str, parent_digest: str, policy_revision: str, count: int) -> str:
    return (
        f"# {source_id} · requirement-backfilled release\n\n"
        f"Private Atlas TaskCompendium release; {count:,} requirement-backfilled tasks.\n\n"
        f"Parent population TaskSpec digest: `{parent_digest}`. Backfill policy revision: "
        f"`{policy_revision}` (rule revision 2).\n"
        "Carry-forward bindings ride the publication receipt: every backfilled task differs from "
        "its parent only in `environment_requirements.capabilities` (extended) and the "
        "`requirements.backfill.capabilities` evidence label — prompts, files, graders and "
        "solutions are byte-identical, so prior oracle and QC verdicts carry forward.\n\n"
        "Readback verifies the parquet hash and count. This release supplies no new oracle, "
        "execution or quality qualification; admission runs through the standard Atlas gates.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--population", required=True, help="The backfilled population artifact prefix")
    parser.add_argument("--repo", required=True, help="The private open-athena dataset repo to create")
    parser.add_argument("--output", required=True, help="The evidence prefix for the publication receipt")
    parser.add_argument("--policy-revision", required=True)
    parser.add_argument("--parent-task-specs-sha256", required=True)
    parser.add_argument("--parent-dataset", required=True, help="The dataset_id the population derives from")
    parser.add_argument("--parent-revision", required=True, help="The pinned revision the population derives from")
    parser.add_argument("--use", required=True, choices=["train", "eval"], help="The publication use")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    configure_coreweave_s3()

    rows = read_population(args.population)
    LOGGER.info("Read %d backfilled tasks from %s", len(rows), args.population)
    table = pa.Table.from_pylist(rows, schema=TASK_SCHEMA)

    with tempfile.TemporaryDirectory(prefix="backfill-release-") as scratch:
        root = Path(scratch)
        tasks_path = root / "tasks.parquet"
        with tasks_path.open("wb") as stream:
            pq.write_table(table, stream, compression="zstd")
        tasks_sha256 = hashlib.sha256(tasks_path.read_bytes()).hexdigest()
        body = dataset_card(args.source_id, args.parent_task_specs_sha256, args.policy_revision, len(rows))
        publication = publish_licensed_population(
            args.source_id,
            args.parent_dataset,
            args.parent_revision,
            args.repo,
            root,
            ("tasks.parquet",),
            body=body,
            pretty_name=f"{args.source_id} requirement-backfilled release",
            use=args.use,
        )

    revision = publication.revision
    LOGGER.info("Published %s at revision %s", args.repo, revision)

    receipt = {
        "source_id": args.source_id,
        "repo": args.repo,
        "revision": revision,
        "tasks_parquet_sha256": tasks_sha256,
        "task_count": len(rows),
        "population": args.population,
        "policy_revision": args.policy_revision,
        "parent_task_specs_sha256": args.parent_task_specs_sha256,
        "parent_dataset_id": args.parent_dataset,
        "parent_revision": args.parent_revision,
        "license_source": publication.license_source,
        "license_index_commit": publication.registry_commit,
        "license_check_verdict": publication.license_check["verdict"],
    }
    output = StoragePath(args.output)
    (output / "publication.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

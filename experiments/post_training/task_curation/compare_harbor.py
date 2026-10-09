# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a generated Harbor view with a pinned Task Trove release manifest."""

import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import click
import pyarrow.parquet as pq
from harbor_config.models.task.config import TaskConfig

from experiments.post_training.task_curation.harbor import TASKS_SCHEMA


def compare_harbor(
    tasks: Path, golden_manifest: Path, *, sources: tuple[str, ...], golden_revision: str
) -> dict[str, Any]:
    """Check the wire schema, every native Harbor config, identities, and per-source counts."""
    golden_bytes = golden_manifest.read_bytes()
    golden = json.loads(golden_bytes)
    unknown = set(sources) - golden["by_source"].keys()
    if unknown:
        raise ValueError(f"Sources absent from the golden release: {sorted(unknown)}")
    parquet = pq.ParquetFile(tasks)
    if not parquet.schema_arrow.equals(TASKS_SCHEMA, check_metadata=False):
        raise ValueError(f"Incompatible task parquet schema: {parquet.schema_arrow}")
    counts: Counter[str] = Counter(dict.fromkeys(sources, 0))
    identities: set[tuple[str, str]] = set()
    task_ids: set[str] = set()
    for batch in parquet.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            identity = (row["source"], row["path"])
            if row["source"] not in counts:
                raise ValueError(f"Unexpected generated source: {row['source']}")
            if identity in identities:
                raise ValueError(f"Duplicate source/path: {identity}")
            identities.add(identity)
            with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:*") as archive:
                members = {member.name: member for member in archive}
                required = {"instruction.md", "task.toml", "environment/Dockerfile", "tests/test.sh"}
                if not required <= members.keys():
                    raise ValueError(f"Missing Harbor files for {identity}: {required - members.keys()}")
                if any(name.startswith("solution/") for name in members):
                    raise ValueError(f"Oracle solution appears in task archive: {identity}")
                handle = archive.extractfile(members["task.toml"])
                assert handle is not None
                config = TaskConfig.model_validate_toml(handle.read().decode())
            if config.metadata["tasktrove_path"] != row["path"] or config.metadata["tasktrove_source"] != row["source"]:
                raise ValueError(f"Source identity changed in archive: {identity}")
            task_id = config.metadata["taskcompendium_id"]
            if task_id in task_ids:
                raise ValueError(f"Duplicate TaskSpec id: {task_id}")
            task_ids.add(task_id)
            counts[row["source"]] += 1
    comparisons = []
    for source, count in sorted(counts.items()):
        released = golden["by_source"][source]
        golden_count = released.get("converted", 0)
        comparisons.append(
            {
                "source": source,
                "generated": count,
                "golden": golden_count,
                "delta": count - golden_count,
                "golden_statuses": released,
            }
        )
    return {
        "schema_compatible": True,
        "harbor_configs_valid": True,
        "row_identities_unique": True,
        "oracle_solutions_separate": True,
        "rows": sum(counts.values()),
        "sources": comparisons,
        "golden_counts_match": bool(comparisons) and all(row["delta"] == 0 for row in comparisons),
        "golden_revision": golden_revision,
        "golden_manifest_sha256": hashlib.sha256(golden_bytes).hexdigest(),
        "runtime_verified": False,
        "comparison_scope": (
            "Release-manifest counts and generated archive structure; golden task binaries were not compared."
        ),
    }


@click.command(help=__doc__)
@click.option("--tasks", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--golden-manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option(
    "--source", "sources", multiple=True, required=True, help="Expected source config; include sources with no output."
)
@click.option("--golden-revision", required=True, help="Pinned release revision from which the manifest was downloaded.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(tasks: Path, golden_manifest: Path, sources: tuple[str, ...], golden_revision: str, output: Path) -> None:
    report = compare_harbor(tasks, golden_manifest, sources=sources, golden_revision=golden_revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    click.echo(json.dumps(report))


if __name__ == "__main__":
    main()

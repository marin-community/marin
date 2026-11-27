# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a new Polaris grader candidate without publishing or admitting it."""

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.long_loop.campaign_data import publish_parquet, retained_task
from experiments.post_training.long_loop.polaris_repair import (
    GRADER_ENVIRONMENT,
    POLICY,
    grader_files,
    with_final_text_grader,
)
from experiments.post_training.task_curation.images.build import EnvironmentArtifact, environment_artifact

RETAINED_REPO = "open-athena/polaris-project-polaris-dataset-53k-b25e0cb8-retained-20261116-b-v1"
RETAINED_REVISION = "9079512fe3d2d332b446b1b37deb9d2898f5461e"
RETAINED_ROWS = 53167
BATCH_ROWS = 64
PARENT_METADATA = (
    "manifest.json",
    "publication-provenance.json",
    "combined-drops.parquet",
    "duplicates.parquet",
    "README.md",
    "UPSTREAM_README.md",
)


@dataclass(frozen=True)
class RewriteReceipt:
    rows: int
    source_sha256: str
    tasks_sha256: str
    inventory_sha256: str


def rewrite_cohort(
    source: Path, output: Path, inventory: Path, built: EnvironmentArtifact, expected_rows: int
) -> RewriteReceipt:
    """Stream every retained row, replacing only grader content and its task hash.

    Completed local files appear only after all input hashes, unique identities and
    the expected population count pass. Runtime and unknown metadata columns keep
    their original Arrow types and values; no sampling or filtering is performed.
    """
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(f"Retained population has {parquet.metadata.num_rows} rows, expected {expected_rows}")
    if output.exists() or inventory.exists():
        raise FileExistsError("Polaris candidate files already exist")
    identities: set[str] = set()
    paths: set[str] = set()
    count = 0
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        tasks = Path(temporary) / "tasks.parquet"
        ledger = Path(temporary) / "identity-inventory.jsonl"
        with pq.ParquetWriter(tasks, parquet.schema_arrow, compression="zstd") as writer, ledger.open("w") as records:
            for batch in parquet.iter_batches(batch_size=BATCH_ROWS):
                transformed = []
                for row in batch.to_pylist():
                    task = retained_task(row, RETAINED_REPO)
                    if row["id"] != task.id or task.id in identities or row["path"] in paths:
                        raise ValueError(f"Changed or duplicated retained identity: {row['path']}/{task.id}")
                    repaired = with_final_text_grader(task, built)
                    encoded = repaired.model_dump_json()
                    digest = hashlib.sha256(encoded.encode()).hexdigest()
                    transformed.append({**row, "task_spec": encoded, "task_sha256": digest})
                    records.write(
                        json.dumps(
                            {
                                "id": task.id,
                                "path": row["path"],
                                "parent_task_sha256": row["task_sha256"],
                                "task_sha256": digest,
                                "runtime_sha256": (
                                    hashlib.sha256(json.dumps(row["runtime"], sort_keys=True).encode()).hexdigest()
                                ),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    identities.add(task.id)
                    paths.add(row["path"])
                    count += 1
                writer.write_table(pa.Table.from_pylist(transformed, schema=parquet.schema_arrow))
        if count != expected_rows:
            raise ValueError(f"Incomplete retained population: {count}/{expected_rows}")
        tasks.replace(output)
        ledger.replace(inventory)
    return RewriteReceipt(
        count,
        file_sha256(StoragePath(str(source))),
        file_sha256(StoragePath(str(output))),
        file_sha256(StoragePath(str(inventory))),
    )


class PolarisCandidate(Artifact):
    rows: int
    tasks_sha256: str
    inventory_sha256: str
    qualified: bool = False


@dataclass(frozen=True)
class CandidateRun:
    output_path: str
    environment_path: str
    source_repo: str
    source_revision: str
    expected_rows: int
    policy: str
    grader_files_sha256: dict[str, str]


def prepare_candidate(config: CandidateRun) -> PolarisCandidate:
    """Build a candidate from the pinned retained cohort; do not publish a release."""
    if {name: hashlib.sha256(data).hexdigest() for name, data in grader_files().items()} != config.grader_files_sha256:
        raise ValueError("Polaris grader code differs from the candidate identity")
    built = EnvironmentArtifact.raw_load(config.environment_path)
    source = Path(
        hf_hub_download(config.source_repo, "tasks.parquet", repo_type="dataset", revision=config.source_revision)
    )
    destination = StoragePath(config.output_path)
    destination.mkdirs(exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        tasks = Path(temporary) / "tasks.parquet"
        inventory = Path(temporary) / "identity-inventory.jsonl"
        receipt = rewrite_cohort(source, tasks, inventory, built, config.expected_rows)
        for name, path, expected in (
            ("tasks.parquet", tasks, receipt.tasks_sha256),
            ("identity-inventory.jsonl", inventory, receipt.inventory_sha256),
        ):
            if publish_parquet(destination, name, path) != expected:
                raise ValueError(f"Candidate readback changed: {name}")
    parent = {}
    for name in PARENT_METADATA:
        source_file = Path(
            hf_hub_download(config.source_repo, name, repo_type="dataset", revision=config.source_revision)
        )
        parent[name] = publish_parquet(destination, f"parent/{name}", source_file)
    manifest = {
        **asdict(receipt),
        "source_repo": config.source_repo,
        "source_revision": config.source_revision,
        "policy": config.policy,
        "grader_files_sha256": config.grader_files_sha256,
        "grader_environment": built.model_dump(mode="json"),
        "parent_metadata": parent,
        "qualification": "PENDING_GUEST_AND_INDEPENDENT_GRADING_REVIEW",
        "admitted": False,
        "removed_rows": 0,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return PolarisCandidate(
        path=config.output_path,
        rows=receipt.rows,
        tasks_sha256=receipt.tasks_sha256,
        inventory_sha256=receipt.inventory_sha256,
    )


def _candidate_run(environment: ArtifactStep[EnvironmentArtifact], ctx: StepContext) -> CandidateRun:
    return CandidateRun(
        ctx.output_path,
        ctx.artifact_path(environment),
        RETAINED_REPO,
        RETAINED_REVISION,
        RETAINED_ROWS,
        POLICY,
        {name: hashlib.sha256(data).hexdigest() for name, data in grader_files().items()},
    )


def candidate_step(version: str) -> ArtifactStep[PolarisCandidate]:
    """Bind the grader environment and retained source into a candidate-only artifact."""
    environment = environment_artifact(GRADER_ENVIRONMENT)
    return ArtifactStep(
        name="rl-data-campaign/polaris-grader-candidate",
        version=version,
        artifact_type=PolarisCandidate,
        run=prepare_candidate,
        build_config=partial(_candidate_run, environment),
        deps=(environment,),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    configure_coreweave_s3()
    step = candidate_step(args.version)
    if args.run:
        print(run(step, max_concurrent=1, force_run_failed=False)[0].model_dump_json(indent=2))
    else:
        print(
            json.dumps(
                {
                    "candidate": artifact_identity(step),
                    "environment": artifact_identity(step.deps[0]),
                    "qualified": False,
                    "admitted": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

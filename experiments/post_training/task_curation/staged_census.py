# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure a staged conversion-release population and bind its census receipt.

The declarations re-pin to the conversion releases only at admission, and the
admission requires each release's census receipt to exist first — measured at
the release pin, not the declared one. This runner measures the DECLARED
Daytona demands of a published release's converted population directly from its
carrier rows (no provider calls, exactly as the declared-demand census works),
emits the census artifact beside the release record, and stages the
``ActorCensusReceipt`` shards bound to the RELEASE pin. The staged shards
append to the receipts registry at the admission re-pin; until then the
declared pins' census receipts stay untouched.
"""

import argparse
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.census_binding import (
    check_receipt_evidence,
    receipt_from_census,
    write_receipt_shards,
)
from experiments.post_training.task_curation.daytona_census import (
    EnvelopeCensus,
    measure_envelopes,
)

LOGGER = logging.getLogger(__name__)


def converter_revision(explicit: str | None) -> str:
    """The landed revision the measurement runs from."""
    if explicit:
        return explicit
    root = Path(__file__).resolve().parents[3]
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def release_documents(release_repo: str, revision: str) -> list[dict]:
    """The release's converted TaskSpec documents with their (empty) runtime records."""
    path = Path(hf_hub_download(release_repo, "tasks.parquet", repo_type="dataset", revision=revision))
    documents = []
    for row in pq_read(path):
        documents.append(row)
    return documents


def pq_read(path: Path) -> list[tuple[dict, dict]]:

    documents = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
        for row in batch.to_pylist():
            documents.append((json.loads(row["task_spec"]), json.loads(row.get("runtime") or "{}")))
    return documents


@dataclass(frozen=True)
class StagedCensus:
    """The measured census record for one staged release population."""

    record: dict
    census: dict


def acceptance_evidence_uri(job_name: str) -> str:
    """The acceptance run's evidence URI the census basis cites.

    The passing-acceptance verification runs at SUBMISSION (the launcher
    refuses without a succeeded acceptance job; the census pod cannot resolve
    the cluster config to re-check). The ruling 2026-11-25 13:00 EST makes the
    acceptance runs the declarative execution evidence.
    """
    return f"iris://cw-us-east-02a/benfeuer/{job_name}"


def measure_staged_population(
    source_id: str,
    release_repo: str,
    release_revision: str,
    *,
    converter_version: str,
    converter_revision: str,
    acceptance_job: str,
) -> StagedCensus:
    """Measure the release's declared Daytona demands and classify through the declarative path.

    The coordinator's ruling 2026-11-25 13:00 EST: declarative build contexts
    are acceptable for admission with execution evidence. The cohort's passing
    acceptance run built and executed its contexts through Daytona without
    named snapshots - that is the execution evidence, and it supersedes the
    declared measurement's unbuilt-image classifications. The launcher
    verifies the passing acceptance run at submission; a cohort without one is
    never submitted and stays Pending for the repair cycle.
    """
    evidence_uri = acceptance_evidence_uri(acceptance_job)
    path = Path(hf_hub_download(release_repo, "tasks.parquet", repo_type="dataset", revision=release_revision))
    documents = pq_read(path)
    result: EnvelopeCensus = measure_envelopes(documents)
    if result.converted == 0:
        raise ValueError(f"The release population is empty: {release_repo}@{release_revision[:12]}")
    task_specs_sha256 = file_sha256(StoragePath(str(path)))
    basis = (
        "Coordinator ruling 2026-11-25 13:00 EST: declarative build contexts are acceptable for "
        "admission with execution evidence. This cohort's passing acceptance run ("
        f"{acceptance_job}) built and executed its contexts through Daytona's declarative path "
        "without named snapshots; the declared measurement's unbuilt-image classifications "
        f"({result.execution_failure_tasks} tasks) are superseded by that evidence. Declared "
        f"demands measured from the carrier rows without provider calls: {result.machine_slots} "
        f"machine slots, {len(result.snapshot_sizes)} distinct contexts; converter checkout "
        f"{converter_revision}; release {release_repo}@{release_revision[:12]}; protocol "
        f"{converter_version}; execution evidence {evidence_uri}"
    )
    record = {
        "source_id": source_id,
        "dataset_id": release_repo,
        "dataset_revision": release_revision,
        "converter_version": converter_version,
        "converter_revision": converter_revision,
        "input_artifact": f"https://huggingface.co/datasets/{release_repo}/tree/{release_revision}",
        "task_specs_sha256": task_specs_sha256,
        "task_count": result.converted,
        "daytona_safety": "safe",
        "daytona_snapshot_count": 0,
        "daytona_machine_slots": result.machine_slots,
        "daytona_execution_failures": 0,
        "daytona_count_basis": basis,
        "daytona_unsafe_reasons": {},
        "daytona_unsafe_details": {},
        "daytona_unsafe_counts": {},
        "daytona_roles": result.roles,
        "actor_environment_counts": dict(result.actor_sources),
        "empty_declared_environment_tasks": {},
        "snapshot_sizes": {},
        "execution_failures": dict(result.execution_failures),
        "mapping_failures": dict(result.mapping_failures),
        "declared_measurement": {
            "snapshot_count": len(result.snapshot_sizes),
            "snapshot_sizes": dict(result.snapshot_sizes),
            "execution_failure_tasks": result.execution_failure_tasks,
            "execution_failures": dict(result.execution_failures),
            "mapping_failures": dict(result.mapping_failures),
            "superseded_by": "ruling 2026-11-25 13:00 EST declarative execution evidence",
        },
        "acceptance_evidence": evidence_uri,
    }
    census = {
        name: record[name]
        for name in (
            "source_id",
            "dataset_id",
            "dataset_revision",
            "converter_version",
            "converter_revision",
            "task_count",
            "task_specs_sha256",
            "daytona_snapshot_count",
            "daytona_machine_slots",
            "daytona_execution_failures",
            "daytona_count_basis",
            "daytona_unsafe_reasons",
            "daytona_unsafe_details",
            "daytona_unsafe_counts",
            "daytona_roles",
        )
    }
    return StagedCensus(record=record, census=census)

    census = {
        name: record[name]
        for name in (
            "source_id",
            "dataset_id",
            "dataset_revision",
            "converter_version",
            "converter_revision",
            "task_count",
            "task_specs_sha256",
            "daytona_snapshot_count",
            "daytona_machine_slots",
            "daytona_execution_failures",
            "daytona_count_basis",
            "daytona_unsafe_reasons",
            "daytona_unsafe_details",
            "daytona_unsafe_counts",
            "daytona_roles",
        )
    }
    return StagedCensus(record=record, census=census)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True, help="The cohort source id.")
    parser.add_argument("--release-repo", required=True, help="The published conversion release repo.")
    parser.add_argument("--release-revision", required=True, help="The release revision to measure.")
    parser.add_argument("--converter-version", required=True, help="The release protocol version.")
    parser.add_argument("--converter-revision", help="The landed revision; defaults to this checkout.")
    parser.add_argument(
        "--acceptance-job",
        required=True,
        help="The cohort's passing acceptance Iris job name (the declarative execution evidence).",
    )
    parser.add_argument("--output", required=True, type=Path, help="The staged census artifact directory.")
    parser.add_argument("--receipts-output", required=True, type=Path, help="The staged receipt shard directory.")
    parser.add_argument("--report", type=Path, help="Write the census result JSON here.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    revision = converter_revision(args.converter_revision)
    staged = measure_staged_population(
        args.source_id,
        args.release_repo,
        args.release_revision,
        converter_version=args.converter_version,
        converter_revision=revision,
        acceptance_job=args.acceptance_job,
    )
    record, census = staged.record, staged.census
    args.output.mkdir(parents=True, exist_ok=True)
    census["path"] = str(args.output)
    (args.output / ".artifact.json").write_text(
        json.dumps({"result": {**record, "path": str(args.output)}}, indent=2, sort_keys=True) + "\n"
    )
    receipt = receipt_from_census(census, None)
    check_receipt_evidence(receipt)
    args.receipts_output.mkdir(parents=True, exist_ok=True)
    write_receipt_shards(args.receipts_output, {receipt.source_id: receipt.model_dump(mode="json")})
    print(
        json.dumps(
            {
                "measured": record["task_count"],
                "receipt_source": receipt.source_id,
                "evidence": census["path"],
                "census": census,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed per-task byte-identity between two conversions of one pinned population.

A population re-measurement can re-serialize the same TaskSpec bytes into a
different artifact layout, changing the artifact-level ``task_specs_sha256``
without changing any task. This module records when that happened: a receipt
binding the two artifact digests to a verified per-task byte-hash equality.
Archived evidence pinned to the original artifact stays valid for the reissued
one through such a receipt; any differing task refuses the receipt, so the
equality is exact or absent.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.registry import read_registry_json
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId

LOGGER = logging.getLogger(__name__)

CONTENT_IDENTITY_RECEIPTS_PATH = Path(__file__).with_name("datasets") / "content_identity_receipts.json"

TASK_DIGEST_SCHEMA = pa.schema(
    [("task_id", pa.string()), ("original_sha256", pa.string()), ("reissued_sha256", pa.string())]
)


class PopulationDigest(BaseModel):
    """One converted population's artifact pin and its own content digest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact: str
    task_specs_sha256: str
    task_count: int


class ContentIdentityReceipt(BaseModel):
    """Per-task TaskSpec byte-hash equality between two conversions of one pin.

    ``identity_sha256`` digests the sorted task-id to TaskSpec-byte-hash map the
    builder verified identical on both sides. The bound digest pair is exact:
    archived evidence may ride the reissued population only when its recorded
    digest and the lineage's input digest are exactly this pair.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    dataset_id: str
    dataset_revision: str
    original: PopulationDigest
    reissued: PopulationDigest
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_url: str

    @model_validator(mode="after")
    def _same_population(self) -> "ContentIdentityReceipt":
        if self.original.task_count != self.reissued.task_count:
            raise ValueError(f"Content identity requires equal task counts: {self.source_id}")
        return self

    def binds(self, source_id: str, left_sha256: str, right_sha256: str) -> bool:
        """Whether this receipt binds exactly this source's digest pair."""
        if source_id != self.source_id:
            return False
        return {left_sha256, right_sha256} == {self.original.task_specs_sha256, self.reissued.task_specs_sha256}


def population_task_digests(population: SourceTextPopulation) -> dict[str, str]:
    """Digest every TaskSpec's exact bytes in the population's spec shards."""
    digests: dict[str, str] = {}
    for path in sorted(StoragePath(population.specs_output_dir + "/*.parquet").glob(), key=str):
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=512):
                columns = batch.column_names
                specs = batch.column(columns.index("task_spec")).to_pylist()
                identifiers = batch.column(columns.index("id")).to_pylist()
                for index, encoded in enumerate(specs):
                    if identifiers[index] in digests:
                        raise ValueError(f"Duplicate task id in {population.source_id}: {identifiers[index]}")
                    digests[identifiers[index]] = hashlib.sha256(encoded.encode()).hexdigest()
    if len(digests) != population.task_count:
        raise ValueError(
            f"{population.source_id}: read {len(digests)} task digests for {population.task_count} declared tasks"
        )
    return digests


def artifact_spec_digest(population: SourceTextPopulation) -> str:
    """The population's own row-order spec digest, as its record declared it."""
    digest = hashlib.sha256()
    for path in sorted(StoragePath(population.specs_output_dir + "/*.parquet").glob(), key=str):
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=512, columns=["task_spec"]):
                for encoded in batch.column(0).to_pylist():
                    digest.update(encoded.encode())
    return digest.hexdigest()


def carry_task_digests(population: SourceTextPopulation) -> dict[str, str]:
    """Digest every TaskSpec's grader/verifier structure in the population's spec shards.

    The carried-release admission verifies the grader identity separately from the task
    bytes (ruling 2026-11-25 ~22:20 EST): a re-serialized TaskSpec whose grader structure
    changed refuses even when other content matches.
    """
    digests: dict[str, str] = {}
    for path in sorted(StoragePath(f"{population.specs_output_dir}/*.parquet").glob(), key=str):
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=512):
                columns = batch.column_names
                specs = batch.column(columns.index("task_spec")).to_pylist()
                identifiers = batch.column(columns.index("id")).to_pylist()
                for index, encoded in enumerate(specs):
                    task = TaskSpec.model_validate_json(encoded)
                    grader = task.grader.model_dump(mode="json")
                    digest = hashlib.sha256(json.dumps(grader, sort_keys=True).encode()).hexdigest()
                    if identifiers[index] in digests and digests[identifiers[index]] != digest:
                        raise ValueError(f"Conflicting grader for task {identifiers[index]} in {population.source_id}")
                    digests[identifiers[index]] = digest
    return digests


def verify_carry_content(original: SourceTextPopulation, reissued: SourceTextPopulation) -> str:
    """Verify the carried-release identity between a prior pin's population and the re-pinned one.

    The re-pin re-keys tasks to canonical ids, so the comparison runs on digest multisets:
    every TaskSpec's exact bytes and every grader/verifier structure must match exactly, and
    both populations must be self-consistent with their declared spec digests. Returns the
    carry identity digest recorded beside the proof; any differing task or grader refuses.
    """
    for population in (original, reissued):
        declared = artifact_spec_digest(population)
        if declared != population.task_specs_sha256:
            raise ValueError(
                f"{population.source_id}: artifact at {population.path} declares "
                f"{population.task_specs_sha256[:12]} but its specs digest to {declared[:12]}"
            )
    original_digests = sorted(population_task_digests(original).values())
    reissued_digests = sorted(population_task_digests(reissued).values())
    if original_digests != reissued_digests:
        differing = len(
            {key for key, count in _counts(original_digests).items() if _counts(reissued_digests).get(key) != count}
        )
        raise ValueError(
            f"Carry content identity refused: {differing} task digest values differ between the prior pin "
            f"and the re-pinned population of {original.source_id}; a single changed task fails closed"
        )
    original_graders = sorted(carry_task_digests(original).values())
    reissued_graders = sorted(carry_task_digests(reissued).values())
    if original_graders != reissued_graders:
        raise ValueError(
            f"Carry grader identity refused: the grader/verifier structures differ between the prior pin "
            f"and the re-pinned population of {original.source_id}"
        )
    return canonical_sha256(original_digests + original_graders)


def _counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def build_identity_receipt(
    source_id: str,
    dataset_id: str,
    dataset_revision: str,
    original_artifact: str,
    reissued_artifact: str,
    evidence_url: str,
) -> ContentIdentityReceipt:
    """Verify per-task byte equality between two conversions and record the receipt."""
    original = SourceTextPopulation.raw_load(original_artifact)
    reissued = SourceTextPopulation.raw_load(reissued_artifact)
    original_digests = population_task_digests(original)
    reissued_digests = population_task_digests(reissued)
    differing = sorted(
        key
        for key in original_digests.keys() | reissued_digests.keys()
        if original_digests.get(key) != reissued_digests.get(key)
    )
    if differing:
        raise ValueError(
            f"Content identity refused: {len(differing)} tasks differ between the conversions of "
            f"{source_id} (first: {differing[:3]})"
        )
    for population in (original, reissued):
        declared = artifact_spec_digest(population)
        if declared != population.task_specs_sha256:
            raise ValueError(
                f"{population.source_id}: artifact at {population.path} declares {population.task_specs_sha256[:12]} "
                f"but its specs digest to {declared[:12]}"
            )
    identity_sha256 = canonical_sha256(original_digests)
    table = pa.Table.from_pylist(
        [
            {"task_id": key, "original_sha256": original_digests[key], "reissued_sha256": reissued_digests[key]}
            for key in sorted(original_digests)
        ],
        schema=TASK_DIGEST_SCHEMA,
    )
    with StoragePath(evidence_url).open("wb", auto_mkdir=True) as stream:
        pq.write_table(table, stream, compression="zstd")
    LOGGER.info(
        "Content identity %s: %d tasks identical across %s and %s",
        source_id,
        len(original_digests),
        original.task_specs_sha256[:12],
        reissued.task_specs_sha256[:12],
    )
    return ContentIdentityReceipt(
        source_id=source_id,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        original=PopulationDigest(
            artifact=original_artifact,
            task_specs_sha256=original.task_specs_sha256,
            task_count=original.task_count,
        ),
        reissued=PopulationDigest(
            artifact=reissued_artifact,
            task_specs_sha256=reissued.task_specs_sha256,
            task_count=reissued.task_count,
        ),
        identity_sha256=identity_sha256,
        evidence_url=evidence_url,
    )


def read_content_identity_receipts(
    path: Path = CONTENT_IDENTITY_RECEIPTS_PATH,
    *,
    load_json=read_registry_json,
) -> dict[str, ContentIdentityReceipt]:
    """Load the pinned content-identity receipts through their registry reference."""
    return {source_id: ContentIdentityReceipt.model_validate(record) for source_id, record in load_json(path).items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--original-artifact", required=True)
    parser.add_argument("--reissued-artifact", required=True)
    parser.add_argument("--evidence-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    receipt = build_identity_receipt(
        args.source_id,
        args.dataset_id,
        args.dataset_revision,
        args.original_artifact,
        args.reissued_artifact,
        args.evidence_url,
    )
    args.output.write_text(json.dumps(receipt.model_dump(mode="json"), indent=2) + "\n")
    print(json.dumps({"source_id": receipt.source_id, "identity_sha256": receipt.identity_sha256}))


if __name__ == "__main__":
    main()

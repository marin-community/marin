# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate stored grading-boundary assessments for retained declarations."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq
from finestore.rl.curation import ActorCensusInput, GradingEnvironmentEvidence
from pydantic import BaseModel, model_validator
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grading_environment import count_grading_environment, grading_environment_census


class RetainedGradingReceipt(BaseModel):
    source_id: str
    input: ActorCensusInput
    normalized_path: str
    evidence: GradingEnvironmentEvidence

    @model_validator(mode="after")
    def _complete_population(self) -> "RetainedGradingReceipt":
        if self.evidence.assessed_tasks != self.input.measured_tasks:
            raise ValueError("Retained grading assessments do not cover the complete population")
        return self


def grading_receipt(census: dict, output_root: str) -> RetainedGradingReceipt:
    """Count the retained importer's persisted assessments without parsing task payloads."""
    counts = Counter()
    normalized = census["normalized_path"]
    for path in sorted(StoragePath(normalized + "/normalize/*.parquet").glob(), key=str):
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=1000, columns=["grading_environment"]):
                for assessment in batch.column(0).to_pylist():
                    if assessment is None:
                        raise ValueError(f"Missing retained grading assessment: {census['source_id']}")
                    count_grading_environment(counts, assessment)
    uri = StoragePath(output_root) / (hashlib.sha256(census["source_id"].encode()).hexdigest()[:20] + ".json")
    if uri.exists():
        raise FileExistsError(f"Retained grading evidence requires a new output version: {uri}")
    receipt = RetainedGradingReceipt(
        source_id=census["source_id"],
        input=ActorCensusInput.model_validate(census["actor"]["input"]),
        normalized_path=normalized,
        evidence=GradingEnvironmentEvidence(**grading_environment_census(counts), evidence_url=str(uri)),
    )
    uri.write_text(receipt.model_dump_json(indent=2) + "\n", auto_mkdir=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--censuses", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    censuses = [row for path in args.censuses for row in json.loads(StoragePath(path).read_text())]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure_coreweave_s3) as pool:
        futures = [pool.submit(grading_receipt, row, args.output_root) for row in censuses]
        receipts = [future.result().model_dump(mode="json") for future in futures]
    (StoragePath(args.output_root) / "grading.json").write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()

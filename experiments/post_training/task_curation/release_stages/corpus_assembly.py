# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Corpus assembly: project the frozen census populations into the curriculum corpus.

One maintained module replaces the per-release corpus-assembly workers (see
INVENTORY.md). It re-projects the exact frozen task bytes via the maintained
``prepare_sources`` over the admitted sources, then verifies the corpus inventory
totals against the sealed census receipt and the declared task counts — the declared
total is the expectation, never a literal in worker source.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.curriculum_backfill import FrozenCoverageSource, prepare_sources
from experiments.post_training.task_curation.release_stages.census import (
    CensusSourceRow,
    read_census_rows,
    resolve_source,
)
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.sources import all_sources

CENSUS_RECEIPT_SUFFIX = "admission-population-census"
CORPUS_KIND_SUFFIX = "curriculum-corpus"


@dataclass(frozen=True)
class CorpusAssemblyInputs:
    """Typed inputs for one corpus-assembly run; every value is a graph output."""

    release_id: str
    source_revision: str
    inputs: ArtifactRef
    census_receipt: ArtifactRef
    output_root: str
    expected_sources: int
    # The release's fresh populations: write mode assembles only these. The retained
    # curriculum already covers every other declared source (additions-mode compose).
    targets: tuple[str, ...]
    # Explicit per-source preparation workers, sized by the graph from StageResources.
    workers: int
    slice_output_root: str | None = None

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        if not self.release_id or not self.source_revision or not self.output_root:
            raise ValueError("release_id, source_revision and output_root are required")
        if self.expected_sources <= 0:
            raise ValueError(f"expected_sources must be positive, got {self.expected_sources}")
        if not self.targets:
            raise ValueError("targets must name the release's fresh populations")
        if self.workers <= 0:
            raise ValueError(f"workers must be positive, got {self.workers}")


def read_census_receipt(receipt: ArtifactRef, release_id: str, expected_sources: int) -> dict[str, dict]:
    """Digest-verify the sealed census receipt and index its per-source results."""
    payload = StoragePath(receipt.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != receipt.sha256:
        raise ValueError(f"census receipt digest mismatch: {receipt.uri}")
    census = json.loads(payload)
    expected_kind = f"atlas-{release_id}-{CENSUS_RECEIPT_SUFFIX}"
    if census.get("kind") != expected_kind:
        raise ValueError(f"census receipt kind is {census.get('kind')!r}, expected {expected_kind!r}")
    results = census.get("results") or []
    if len(results) != expected_sources:
        raise ValueError(f"census receipt holds {len(results)} results, expected {expected_sources}")
    return {row["source_id"]: row for row in results}


def frozen_sources_for(
    rows: list[CensusSourceRow],
    census_results: dict[str, dict],
    *,
    sources_fn=all_sources,
) -> list[FrozenCoverageSource]:
    """Build the frozen coverage sources from the census receipt's population pins.

    The 0-admission oracle-filtered sources ship excluded, zero-task declarations and
    stay out of the corpus (the prior releases carried the same shape).
    """
    configure_coreweave_s3()
    frozen = []
    for row in rows:
        census_result = census_results.get(row.source_id)
        if census_result is None:
            raise ValueError(f"sealed census receipt carries no result for {row.source_id}")
        if census_result.get("oracle_empty") is not None:
            # The 0-admission oracle filtration ships an excluded, zero-task
            # declaration; the corpus omits it exactly as the prior releases did.
            continue
        source = resolve_source(row.source_id, row.declaration, sources_fn)
        declaration = source.declaration
        binding = declaration.actor_census_input
        if binding is None:
            raise ValueError(f"declared source lacks its exact input census: {row.source_id}")
        census_result = census_results.get(row.source_id)
        if census_result is None:
            raise ValueError(f"sealed census receipt carries no result for {row.source_id}")
        frozen.append(
            FrozenCoverageSource(
                declaration=declaration,
                specs_dir=census_result["population"]["specs_output_dir"],
                task_column="task_spec",
                raw_count=declaration.task_count,
                task_specs_sha256=binding.task_specs_sha256,
                task_spec_scan_sha256=census_result["inventory"]["scanned_task_specs_sha256"],
                input_record_sha256=file_sha256(StoragePath(binding.evidence_url)),
                intended_use=declaration.intended_use,
                input_evidence=binding.evidence_url,
            )
        )
    return frozen


def verify_corpus(corpus: dict, rows: list[CensusSourceRow]) -> int:
    """The corpus inventory must cover exactly the declared task totals."""
    if len(corpus["sources"]) != len(rows):
        raise ValueError(f"corpus holds {len(corpus['sources'])} sources, expected {len(rows)}")
    declared_total = sum(row.declaration["task_count"] for row in rows)
    inventory_total = sum(entry["task_count"] for entry in corpus["inventory"])
    if inventory_total != declared_total:
        raise ValueError(f"corpus inventory holds {inventory_total} tasks, expected the declared {declared_total}")
    return declared_total


def run_corpus_assembly(
    inputs: CorpusAssemblyInputs,
    *,
    dry_run: bool,
    only: str | None = None,
    sources_fn=all_sources,
    feeds: tuple | None = None,
) -> dict:
    """Assemble the corpus; ``dry_run`` assembles one real source into the slice root."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "corpus_assembly", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        rows = read_census_rows(inputs.inputs, inputs.source_revision)
        census_results = read_census_receipt(inputs.census_receipt, inputs.release_id, inputs.expected_sources)
        if dry_run:
            if only is None:
                raise ValueError("dry-run corpus assembly requires --only: the preflight builds one real source")
            if inputs.slice_output_root is None:
                raise ValueError("dry-run corpus assembly requires slice_output_root in the typed inputs")
            rows = [row for row in rows if row.source_id == only]
            if not rows:
                raise ValueError(f"census inputs carry no row for {only!r}")
            output_root = inputs.slice_output_root
        else:
            if only is not None:
                raise ValueError("write-mode corpus assembly assembles every row; use dry-run for a slice")
            if len(rows) != inputs.expected_sources:
                raise ValueError(f"census inputs carry {len(rows)} rows, expected {inputs.expected_sources}")
            missing = sorted(set(inputs.targets) - {row.source_id for row in rows})
            if missing:
                raise ValueError(f"census inputs carry no row for targets: {missing[:5]}")
            rows = [row for row in rows if row.source_id in inputs.targets]
            output_root = inputs.output_root
        frozen = frozen_sources_for(rows, census_results, sources_fn=sources_fn)
        prepare_sources(frozen, output_root, workers=inputs.workers)
        # The reference binds the bytes prepare_sources wrote, which consumers digest-verify.
        corpus_raw = (StoragePath(output_root) / "corpus.json").read_bytes()
        corpus = json.loads(corpus_raw)
        shipped = [row for row in rows if (census_results.get(row.source_id) or {}).get("oracle_empty") is None]
        tasks = verify_corpus(corpus, shipped)
        corpus_ref = {"uri": f"{output_root}/corpus.json", "sha256": hashlib.sha256(corpus_raw).hexdigest()}
        proof = {
            "kind": f"atlas-{inputs.release_id}-{CORPUS_KIND_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "inputs": {"uri": inputs.inputs.uri, "sha256": inputs.inputs.sha256},
            "census_receipt": {"uri": inputs.census_receipt.uri, "sha256": inputs.census_receipt.sha256},
            "sources": len(shipped),
            "tasks": tasks,
            "corpus": corpus_ref,
            "dry_run": dry_run,
            "sql_writes": 0,
            "completed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        if not dry_run:
            proof_path = StoragePath(f"{output_root}/corpus-proof.json")
            raw_proof = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode()
            if proof_path.exists():
                raise ValueError(f"corpus proof already exists at {proof_path}; use a new output root")
            proof_path.write_bytes(raw_proof, auto_mkdir=True)
            if proof_path.read_bytes() != raw_proof:
                raise ValueError("corpus proof readback mismatch")
            proof["receipt"] = {"uri": str(proof_path), "sha256": hashlib.sha256(raw_proof).hexdigest()}
        alerts.stage_finished(
            inputs.release_id, "corpus_assembly", {"mode": mode, "sources": len(rows), "tasks": tasks}, alert_feeds
        )
        return proof
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "corpus_assembly", {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON CorpusAssemblyInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    parser.add_argument("--only", help="Assemble a single source (dry-run preflight slice)")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    document["inputs"] = ArtifactRef(**document["inputs"])
    document["census_receipt"] = ArtifactRef(**document["census_receipt"])
    inputs = CorpusAssemblyInputs(**document)
    proof = run_corpus_assembly(inputs, dry_run=args.mode == "dry-run", only=args.only)
    label = "CORPUS-PREFLIGHT-SLICE" if args.mode == "dry-run" else "CORPUS-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()

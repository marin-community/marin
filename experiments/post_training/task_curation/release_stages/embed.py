# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embedding: run the curriculum semantic pipeline over the assembled corpus.

One maintained module replaces the per-release curriculum-embed launchers (see
INVENTORY.md). The maintained pipeline (``curriculum_backfill.run_stage``) does the
work; this module adds the typed manifest boundary — digest-verified corpus, the
census population total as the expectation, GPU-vs-CPU decided by a valid backend
parity receipt for the serving runtime, resources from the measured sizing curve,
and direct stage alerts.

Device rule (BRIEF requirement 4): CUDA only with an accepted parity receipt AND a
typed accelerator runtime; otherwise CPU. The dry run builds the stage plan without
executing anything.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.curriculum_backfill import run_stage
from experiments.post_training.task_curation.embedding import COVERAGE_VERSION
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.release_stages.sizing import StageResources, embedding_sizing

EMBED_RECEIPT_SUFFIX = "curriculum-embed"
COVERAGE_STAGE = "coverage"


@dataclass(frozen=True)
class EmbedInputs:
    """Typed inputs for one embedding run; every value is a graph output."""

    release_id: str
    source_revision: str
    corpus: ArtifactRef
    artifact_root: str
    cache_root: str
    expected_source_count: int
    expected_task_count: int
    parity_receipt: ArtifactRef | None = None
    gpu_runtime: str | None = None
    only: str | None = None
    """The bounded slice source a preflight dry-run builds the plan for; None in write mode."""

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        if not self.release_id or not self.source_revision:
            raise ValueError("release_id and source_revision are required")
        if not self.artifact_root or not self.cache_root:
            raise ValueError("artifact_root and cache_root are required")
        if self.expected_source_count <= 0 or self.expected_task_count <= 0:
            raise ValueError("expected counts must be positive")
        if (self.parity_receipt is None) != (self.gpu_runtime is None):
            raise ValueError(
                "CUDA embedding requires both an accepted parity receipt and the typed accelerator"
                " runtime; CPU embedding requires neither"
            )

    @property
    def device(self) -> str:
        """The embedding device implied by the typed parity and runtime inputs."""
        return "cuda" if self.parity_receipt is not None else "cpu"


def read_corpus(corpus: ArtifactRef) -> dict:
    """Digest-verify the sealed corpus document and return it."""
    payload = StoragePath(corpus.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != corpus.sha256:
        raise ValueError(f"corpus digest mismatch: {corpus.uri}")
    return json.loads(payload)


def verify_corpus(corpus: dict, inputs: EmbedInputs) -> int:
    """The corpus inventory must match the declared source and population counts."""
    inventory = corpus.get("inventory") or []
    if len(inventory) != inputs.expected_source_count:
        raise ValueError(f"corpus carries {len(inventory)} sources, expected {inputs.expected_source_count}")
    total = sum(row["task_count"] for row in inventory)
    if total != inputs.expected_task_count:
        raise ValueError(f"corpus carries {total} tasks, expected the census total {inputs.expected_task_count}")
    if corpus.get("population") != "converted":
        raise ValueError(f"corpus population is {corpus.get('population')!r}, expected 'converted'")
    return total


def find_retained_coverage_receipt(artifact_root: str, corpus_uri: str) -> dict:
    """Locate this run's retained-coverage receipt under the artifact root."""
    pattern = StoragePath(artifact_root) / "**" / "retained-coverage-receipt.json"
    matches = pattern.glob()
    if not matches:
        raise ValueError(f"embedding completed without a retained-coverage receipt under {artifact_root}")
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} retained-coverage receipts under {artifact_root};"
            " use a fresh artifact root per release so the receipt is unambiguous"
        )
    receipt = json.loads(matches[0].read_text())
    if receipt.get("corpus") != corpus_uri:
        raise ValueError(f"retained-coverage receipt binds {receipt.get('corpus')!r}, expected {corpus_uri!r}")
    return receipt


def seal(path: str, value: dict) -> ArtifactRef:
    """Write-once seal with byte-identical readback; returns the artifact reference."""
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    target = StoragePath(path)
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError(f"sealed artifact {path} exists with different content; use a new output root")
    else:
        target.write_bytes(raw, auto_mkdir=True)
    if target.read_bytes() != raw:
        raise ValueError(f"seal readback mismatch at {path}")
    return ArtifactRef(uri=str(target), sha256=hashlib.sha256(raw).hexdigest())


def _resources(total_tasks: int) -> StageResources:
    return embedding_sizing(total_tasks)


def run_embed(
    inputs: EmbedInputs,
    *,
    dry_run: bool,
    feeds: tuple | None = None,
) -> dict:
    """Run one embedding pass; ``dry_run`` builds the stage plan without executing."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(
        inputs.release_id,
        "embed",
        {"mode": mode, "device": inputs.device, **({"only": inputs.only} if inputs.only else {})},
        alert_feeds,
    )
    try:
        inputs.validate()
        configure_coreweave_s3()
        corpus_document = read_corpus(inputs.corpus)
        total_tasks = verify_corpus(corpus_document, inputs)
        sizing = _resources(total_tasks)
        proof: dict = {
            "kind": f"atlas-{inputs.release_id}-{EMBED_RECEIPT_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "corpus": {"uri": inputs.corpus.uri, "sha256": inputs.corpus.sha256},
            "device": inputs.device,
            "gpu_runtime": inputs.gpu_runtime,
            "coverage_version": COVERAGE_VERSION,
            "tasks": total_tasks,
            "sources": len(corpus_document["inventory"]),
            "resources": sizing.as_resources(),
            "workers": sizing.workers,
            "worker_memory_gb": sizing.worker_memory_gb,
            "dry_run": dry_run,
            "sql_writes": 0,
        }
        if dry_run:
            alerts.stage_finished(
                inputs.release_id,
                "embed",
                {"mode": mode, "tasks": total_tasks, **({"only": inputs.only} if inputs.only else {})},
                alert_feeds,
            )
            return proof
        run_stage(
            inputs.corpus.uri,
            inputs.artifact_root,
            inputs.cache_root,
            COVERAGE_STAGE,
            inputs.device,
            sizing.workers,
            f"{sizing.worker_memory_gb}g",
            True,
            parity_receipt_uri=inputs.parity_receipt.uri if inputs.parity_receipt is not None else None,
        )
        coverage_receipt = find_retained_coverage_receipt(inputs.artifact_root, inputs.corpus.uri)
        proof["coverage_receipt"] = {
            "uri": coverage_receipt["coverage_uri"],
            "sha256": coverage_receipt["coverage_sha256"],
        }
        proof["release_receipt"] = {
            "uri": coverage_receipt["release_uri"],
            "sha256": coverage_receipt["release_sha256"],
        }
        proof["release_id"] = coverage_receipt["release_id"]
        proof["coverage_id"] = coverage_receipt["coverage_id"]
        proof["parity_receipt"] = coverage_receipt.get("parity_receipt")
        receipt = seal(f"{inputs.artifact_root}/embed-receipt.json", proof)
        proof["receipt"] = {"uri": receipt.uri, "sha256": receipt.sha256}
        alerts.stage_finished(
            inputs.release_id,
            "embed",
            {"mode": mode, "release_id": proof["release_id"], "coverage_id": proof["coverage_id"], **proof["receipt"]},
            alert_feeds,
        )
        return proof
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "embed", {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON EmbedInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    document["corpus"] = ArtifactRef(**document["corpus"])
    if document.get("parity_receipt") is not None:
        document["parity_receipt"] = ArtifactRef(**document["parity_receipt"])
    inputs = EmbedInputs(**document)
    proof = run_embed(inputs, dry_run=args.mode == "dry-run")
    label = "EMBED-DRY-RUN" if args.mode == "dry-run" else "EMBED-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()

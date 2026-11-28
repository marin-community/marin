# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify complete public-input equivalence before reusing curriculum alignment."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.curriculum_projection import (
    SourceReplacement,
    task_rebinding,
    write_task_identity_mappings,
)
from experiments.post_training.task_curation.release_stages.census import seal
from experiments.post_training.task_curation.release_stages.curriculum_compose import read_retained_release
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref


@dataclass(frozen=True)
class PublicInputReuseInputs:
    """Inputs for comparing complete retained and current task populations."""

    release_id: str
    source_revision: str
    archive_release: ArtifactRef
    census_receipt: ArtifactRef
    targets: tuple[str, ...]
    output_root: str


def run_public_input_reuse(inputs: PublicInputReuseInputs, *, dry_run: bool, feeds: tuple | None = None) -> dict:
    """Check every target task and seal provenance mappings only after complete verification."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "public_input_reuse", {"mode": mode}, alert_feeds)
    try:
        if not inputs.release_id or not inputs.source_revision or not inputs.output_root:
            raise ValueError("release_id, source_revision and output_root are required")
        if not inputs.targets or len(set(inputs.targets)) != len(inputs.targets):
            raise ValueError("targets must name distinct sources")
        configure_coreweave_s3()
        archive = read_retained_release(inputs.archive_release.model_dump(mode="json"))
        archived = {row.source_id: row for row in archive.sources}
        census = read_ref(inputs.census_receipt)
        if (
            census["kind"] != f"atlas-{inputs.release_id}-admission-population-census"
            or census["source_revision"] != inputs.source_revision
        ):
            raise ValueError("census receipt belongs to a different release or source revision")
        current = {row["source_id"]: row for row in census["results"]}
        if len(current) != len(census["results"]):
            raise ValueError("census receipt repeats a source")
        rebindings, results = [], []
        for source_id in sorted(inputs.targets):
            row = current[source_id]
            if row["old_task_specs_or_count_bindings_inherited"] or row["qa_bindings_inherited"]:
                raise ValueError(f"reuse verification requires a fresh census: {source_id}")
            prepared = row["public_preparation"]
            manifest = read_ref(ArtifactRef(**prepared["manifest"]))
            if (
                manifest["input"]["declaration"] != row["declaration"]
                or manifest["input"]["task_spec_scan_sha256"] != row["inventory"]["scanned_task_specs_sha256"]
                or manifest["converted_rows"] != row["inventory"]["retained_tasks"]
            ):
                raise ValueError(f"public preparation differs from its current census: {source_id}")
            replacement = SourceReplacement(
                source_id=source_id,
                source_manifest=prepared["manifest"],
                public_inputs=prepared["public_inputs"],
            )
            verified = task_rebinding(archived[source_id], replacement)
            rebindings.append(verified)
            results.append(
                {
                    "source_id": source_id,
                    "tasks": len(verified.task_ids),
                    "removed_tasks": len(verified.removed_task_ids),
                    "changed_task_ids": sum(
                        a != b for a, b in zip(verified.task_ids, verified.new_task_ids, strict=True)
                    ),
                    "all_public_inputs_unchanged": True,
                    "replacement": replacement.model_dump(mode="json"),
                }
            )
        proof = {
            "kind": f"atlas-{inputs.release_id}-public-input-reuse",
            "status": "PASS",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "archive_release": inputs.archive_release.model_dump(mode="json"),
            "census_receipt": inputs.census_receipt.model_dump(mode="json"),
            "results": results,
            "tasks": sum(row["tasks"] for row in results),
            "vectors_recomputed": 0,
            "scores_recomputed": 0,
            "dry_run": dry_run,
        }
        if not dry_run:
            mappings = write_task_identity_mappings(StoragePath(inputs.output_root), tuple(rebindings))
            proof["task_identity_mappings"] = {sid: ref.model_dump(mode="json") for sid, ref in mappings.items()}
            receipt = seal(f"{inputs.output_root}/reuse-receipt.json", proof)
            proof["receipt"] = receipt.model_dump(mode="json")
        alerts.stage_finished(
            inputs.release_id, "public_input_reuse", {"mode": mode, "tasks": proof["tasks"]}, alert_feeds
        )
        return proof
    except Exception as error:
        alerts.stage_failed(
            inputs.release_id, "public_input_reuse", {"mode": mode, "error": str(error)[:500]}, alert_feeds
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text())
    for field in ("archive_release", "census_receipt"):
        document[field] = ArtifactRef(**document[field])
    document["targets"] = tuple(document["targets"])
    print(json.dumps(run_public_input_reuse(PublicInputReuseInputs(**document), dry_run=args.mode == "dry-run")))


if __name__ == "__main__":
    main()

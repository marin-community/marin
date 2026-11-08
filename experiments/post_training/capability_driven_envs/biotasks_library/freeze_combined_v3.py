# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Version verified v2 reference/path fixes and quarantine unadjudicated consensus leads."""

import argparse
import base64
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from taskcompendium.models import JsonObjectInFinal, TaskSpec
from taskforge.content_hash import digest

from experiments.post_training.capability_driven_envs.biotasks_library.finalize_combined_v1 import (
    _read_bytes,
    _write_new,
)
from experiments.post_training.capability_driven_envs.biotasks_library.freeze_combined_v2 import V2_ROOT

V3_ROOT = V2_ROOT.replace("v2/snapshots/verified-corrections-1/", "v3/snapshots/reference-path-review-1/")
CORRECTED = {
    "biotasks-microarray_preprocessing-a8d21e94279b2b7d",
    "biotasks-alpha_diversity_metrics-eb3a136672356774",
    "biotasks-variance_stabilizing_transformation-33390457abb3578f",
    "biotasks-cigar_arithmetic-212da74cd61ea63e",
}
QUARANTINED = {
    "biotasks-benchmark_analysis-b3b76c13116a08f2",
    "biotasks-binding_data_curation-d897236d802e54d5",
    "biotasks-gene_set_enrichment-d9ecad5176f5e464",
    "biotasks-genomic_region_segmentation-4641f4a0992dec90",
    "biotasks-microarray_qc_reconciliation-1be32452fd9d10c7",
    "biotasks-spatial_proteomics_classification-33e40c00d4372ee7",
    "biotasks-temporal_covariate_smoothness-c11a03bd9b141b82",
    "biotasks-annotation_qc-062f6181cab5f2c1",
    "biotasks-genome_assembly_metadata_validation-24aa7ee24b013c7d",
    "biotasks-isobaric_reporter_quantitation-e132e3542a38ff69",
    "biotasks-motif_enrichment_auc_nes-3fd93adfbf7f9243",
    "biotasks-trajectory_inference_graph_algorithms-ed4857c08b42d708",
    "biotasks-isobaric_reporter_purity_correction-e069ecdab84c299b",
    "biotasks-single_cell_annotation-c7ef5e6abf38e9fc",
    "biotasks-flow_cytometry_gating_stats-4272ba2dfb150bfb",
    "biotasks-ontology_controlled_vocabulary_validation-578888025b21893d",
    "biotasks-variant_qc-95ee993fa4c958f4",
}


def _reference(task: TaskSpec) -> TaskSpec:
    if task.id not in CORRECTED:
        return task
    expected = json.loads(json.dumps(task.grader.parameters["expected"]))
    if "microarray_preprocessing" in task.id:
        if expected["max_abs_diff_between_samples"] != 0:
            raise ValueError("Unexpected microarray reference")
        expected["max_abs_diff_between_samples"] = 3.0
    elif "cigar_arithmetic" in task.id:
        if expected["r3"]["ref_span"] != 50:
            raise ValueError("Unexpected CIGAR reference")
        expected["r3"]["ref_span"] = 47
    elif "variance_stabilizing_transformation" in task.id:
        if (expected["u_1000"], expected["diff_1000"]) != (10.218, 0.252):
            raise ValueError("Unexpected VST reference")
        expected.update(u_1000=10.219, diff_1000=0.253)
    elif "alpha_diversity_metrics" in task.id:
        corrections = {
            "S1": (2.8284, 2.6667),
            "S2": (1.4835, 1.2270),
            "S3": (4.0, 4.0),
            "S4": (3.1842, 2.4096),
        }
        for sample in expected["samples"]:
            sid = sample["sample_id"]
            if sid not in corrections or abs(sample["inverse_simpson"] - corrections[sid][1]) > 0.0001:
                raise ValueError("Unexpected Hill sample reference")
            sample["hill_q1"], sample["hill_q2"] = corrections[sid]
        context = task.context.model_dump(mode="json")
        for event in context["events"]:
            if event["type"] == "message":
                event["content"] = event["content"].replace(
                    "Hill number of order 2 = 1/sqrt(lambda)", "Hill number of order 2 = 1/lambda"
                )
        if context == task.context.model_dump(mode="json"):
            raise ValueError("Hill instruction was not changed")
        task = task.model_copy(update={"context": task.context.model_validate(context)})
        groups = task.resources.model_dump(mode="json")
        found = 0
        for resource in groups["worker"]:
            if resource["path"] != "workspace/reference/metrics_notes.md":
                continue
            text = base64.b64decode(resource["source"]["content_base64"]).decode()
            if "q=2 is 1/sqrt(lambda)" not in text:
                raise ValueError("Unexpected Hill reference note")
            resource["source"]["content_base64"] = base64.b64encode(
                text.replace("q=2 is 1/sqrt(lambda)", "q=2 is 1/lambda").encode()
            ).decode()
            found += 1
        if found != 1:
            raise ValueError("Missing Hill reference note")
        task = task.model_copy(update={"resources": task.resources.model_validate(groups)})
    grader = task.grader.model_copy(update={"parameters": {**task.grader.parameters, "expected": expected}})
    return task.model_copy(update={"grader": grader})


def _fix(task: TaskSpec) -> tuple[TaskSpec, tuple[str, ...]]:
    reasons = []
    updated = _reference(task)
    if updated != task:
        reasons.append("independently_recomputed_reference")
    if any(event.type == "message" and "workspace/" in event.content for event in task.context.events):
        if task.environment_requirements.working_directory != "/workspace":
            raise ValueError("Unexpected task working directory")
        requirements = updated.environment_requirements.model_copy(update={"working_directory": "/"})
        updated = updated.model_copy(update={"environment_requirements": requirements})
        reasons.append("relative_workspace_path_resolution")
    if task.answer_format.kind != "json_value":
        raise ValueError("Unexpected v2 submission format")
    updated = updated.model_copy(update={"answer_format": JsonObjectInFinal()})
    reasons.append("reference_blind_single_json_object_submission")
    if reasons:
        updated = updated.model_copy(update={"id": task.id + "-v3"})
        updated = TaskSpec.model_validate(updated.model_dump(mode="json"))
    return updated, tuple(reasons)


def freeze(output: str, code_commit: str, source: str = V2_ROOT + "taskspecs.jsonl") -> dict[str, Any]:
    data = _read_bytes(source)
    records = [json.loads(line) for line in data.splitlines() if line]
    if hashlib.sha256(data).hexdigest() != "bafd5aa61ca5f3aa421cb9b865ba22057da53601a1268f35e637cff6a9a5790e":
        raise ValueError("Unexpected source v2 snapshot digest")
    if len(records) != 305 or {r["task_id"] for r in records} & (CORRECTED | QUARANTINED) != CORRECTED | QUARANTINED:
        raise ValueError("Unexpected frozen v2 source bank")
    tasks = []
    changes = []
    counts: Counter[str] = Counter()
    for row in records:
        old = TaskSpec.model_validate(row["task"])
        if digest(old) != row["task_digest"] or old.id != row["task_id"]:
            raise ValueError("Frozen source identity mismatch")
        original_disposition = row["audit_disposition"]
        if old.id in QUARANTINED:
            disposition = "consensus_disputed_unresolved"
        elif original_disposition == "audit_disputed_unresolved":
            disposition = original_disposition
        else:
            disposition = "provisionally_eligible_v3"
        # Never alter an excluded TaskSpec (including the 58 previously disputed items).
        revised, reasons = _fix(old) if disposition == "provisionally_eligible_v3" else (old, ())
        counts[disposition] += 1
        tasks.append(
            {
                **row,
                "v2_task_id": old.id,
                "v2_task_digest": digest(old),
                "task": revised.model_dump(mode="json"),
                "task_id": revised.id,
                "task_digest": digest(revised),
                "audit_disposition": disposition,
            }
        )
        if reasons:
            changes.append(
                {
                    "v2_task_id": old.id,
                    "v2_task_digest": digest(old),
                    "v3_task_id": revised.id,
                    "v3_task_digest": digest(revised),
                    "reasons": reasons,
                }
            )
    if (
        counts
        != {"provisionally_eligible_v3": 230, "consensus_disputed_unresolved": 17, "audit_disputed_unresolved": 58}
        or len({r["task_id"] for r in tasks}) != 305
    ):
        raise ValueError(f"Unexpected v3 bank: {counts}")
    tasks.sort(key=lambda r: r["task_id"])
    changes.sort(key=lambda r: r["v3_task_id"])
    artifacts = {
        "taskspecs.jsonl": b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in tasks),
        "changes.jsonl": b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in changes),
    }
    manifest = {
        "schema_version": "1",
        "code_commit": code_commit,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "internal_provisional_not_independently_verified_bank",
        "summary": {"dispositions": dict(counts), "changed_tasks": len(changes)},
        "source": {"uri": source, "sha256": hashlib.sha256(data).hexdigest()},
        "artifacts": {k: {"sha256": hashlib.sha256(v).hexdigest(), "bytes": len(v)} for k, v in artifacts.items()},
    }
    for name, content in artifacts.items():
        _write_new(output.rstrip("/") + "/" + name, content)
    _write_new(output.rstrip("/") + "/manifest.json", json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=V3_ROOT)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output, args.code_commit), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

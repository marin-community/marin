# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble a local replacement record from an existing evaluation archive."""

import argparse
import hashlib
import json
from pathlib import Path

from finestore.reader import ReadView
from marin.evaluation.evalchemy.result import FineStoreEvalchemyResult
from marin.evaluation.lm_eval_samples import summarize_native_eval_samples
from marin.evaluation.records import EvalRunRecord, RunStatus
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3


def audit_resume_rows(before: list[dict], after: list[dict]) -> dict:
    """Require a selective retry to preserve every successfully generated response."""
    old = {(row["namespace"], row["unit_key"]): row for row in before}
    new = {(row["namespace"], row["unit_key"]): row for row in after}
    if len(old) != len(before) or len(new) != len(after) or old.keys() != new.keys():
        raise ValueError("Retry changed the response identity set")
    preserved = 0
    replaced = 0
    for key, row in old.items():
        prior_output = json.loads(row["payload"])["output"]
        current_output = json.loads(new[key]["payload"])["output"]
        prior_failed = (
            isinstance(prior_output, dict) and prior_output.get("__evalchemy_resume_value__") == "failed_generation"
        )
        current_failed = (
            isinstance(current_output, dict) and current_output.get("__evalchemy_resume_value__") == "failed_generation"
        )
        if current_failed:
            raise ValueError("Retry still contains a failed generation")
        if prior_failed:
            if prior_output["failure_category"] != "model_transport":
                raise ValueError("Selective retry includes a non-transport failure")
            prior_sample = json.loads(row["payload"])["sample"]
            current_sample = json.loads(new[key]["payload"])["sample"]
            if prior_sample != current_sample:
                raise ValueError("Retry changed the failed request")
            replaced += 1
        else:
            if row != new[key]:
                raise ValueError("Retry changed a previously successful response")
            preserved += 1
    return {"preserved_responses": preserved, "retried_transport_failures": replaced}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True)
    parser.add_argument("--original-sha256", required=True)
    parser.add_argument("--assembly-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry-results")
    args = parser.parse_args()
    configure_coreweave_s3()
    filesystem, key = filesystem_for(args.record)
    original = filesystem.cat_file(key)
    if hashlib.sha256(original).hexdigest() != args.original_sha256:
        raise ValueError("Original record changed")
    record = EvalRunRecord.model_validate_json(original)
    if record.status != RunStatus.ARTIFACT_FAILED:
        raise ValueError("Recovery requires an artifact-failed record")
    results = args.retry_results or record.results_path
    audit = {}
    if args.retry_results:
        before = (
            ReadView(record.results_path)
            .scan("evalchemy_resume")
            .select(["namespace", "unit_key", "payload"])
            .to_pylist()
        )
        after = ReadView(results).scan("evalchemy_resume").select(["namespace", "unit_key", "payload"]).to_pylist()
        audit = audit_resume_rows(before, after)
    summary = summarize_native_eval_samples(results, tasks=record.evaluation.tasks)
    if not summary.coverage:
        raise ValueError("Archive has no scored coverage")
    for coverage in summary.coverage.values():
        if coverage.errors or not (coverage.n_scored == coverage.n_attempted == coverage.n_benchmark):
            raise ValueError("Refusing to recover incomplete or infra-failed coverage")
    updates = {
        "status": RunStatus.SUCCEEDED,
        "error": None,
        "results_path": results,
        "metrics": FineStoreEvalchemyResult(path=results).task_metrics(),
        "canonical_metrics": summary.canonical_metrics,
        "coverage": summary.coverage,
        "evaluation": record.evaluation.model_copy(update={"tasks": summary.tasks}),
        "description": (record.description or "") + f" Reassembled from retained artifacts at {args.assembly_commit}.",
    }
    if args.retry_results:
        updates.update(timing=None, inference_metrics=None)
    recovered = record.model_copy(update=updates)
    provenance = {
        "original_record": args.record,
        "original_sha256": args.original_sha256,
        "assembly_commit": args.assembly_commit,
        "results_path": results,
        "selective_retry": audit,
        "source_modified": False,
        "generation_performed_by_this_tool": False,
    }
    args.output.mkdir(exist_ok=False, parents=True)
    (args.output / "original-record.json").write_bytes(original)
    (args.output / "record.json").write_text(recovered.model_dump_json(indent=2, by_alias=True))
    (args.output / "recovery.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()

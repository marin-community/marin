# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify completed cluster selection ledgers in parallel using their task reports."""

import argparse
import logging

from iris.cluster.client.job_info import get_job_info
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.scripts.metadata_cluster_selection import (
    _digest,
    _read_json,
    _write_json,
    verify_selection_outputs,
)

logger = logging.getLogger(__name__)


def _ledger_paths(report: dict) -> list[str]:
    return [item["output"] for item in report["completed"] if item["status"] == "selected"]


def verify_selection_task(
    inventory: dict,
    report: dict,
    output: str,
    *,
    exact_catalog_root: str,
    fuzzy_catalog_root: str,
    target_cluster: int,
    task_index: int,
    task_count: int,
    batch_size: int = 8192,
) -> dict:
    """Verify one task's output bytes and persist a proof bound to its full input report."""
    config = report["config"]
    expected = {
        "inventory_sha256": _digest(inventory),
        "exact_catalog_root": exact_catalog_root,
        "fuzzy_catalog_root": fuzzy_catalog_root,
        "target_cluster": target_cluster,
        "task_index": task_index,
        "task_count": task_count,
        "max_shards": None,
    }
    if any(config[key] != value for key, value in expected.items()) or not report["complete"] or report["sampled"]:
        raise ValueError("Verification requires a complete full-selection task with matching frozen inputs")
    indexes = [item["index"] for item in report["completed"]]
    paths = _ledger_paths(report)
    if (
        len(indexes) != len(set(indexes))
        or len(indexes) != report["assigned_shards"]
        or any(index % task_count != task_index for index in indexes)
        or len(paths) != len(set(paths))
    ):
        raise ValueError("Selection report has repeated or mismatched task/shard output references")
    verified = verify_selection_outputs(
        inventory,
        [report],
        exact_catalog_root=exact_catalog_root,
        fuzzy_catalog_root=fuzzy_catalog_root,
        target_cluster=target_cluster,
        batch_size=batch_size,
    )
    if verified["verified_occurrences"] != report["selected_occurrences"] or verified["verified_shards"] != len(paths):
        raise ValueError("Verified ledger counts differ from the completed selection task")
    proof = {
        **expected,
        "selection_report_sha256": _digest(report),
        "ordered_outputs_sha256": _digest(paths),
        **verified,
        "validation": "completion markers, schema, input pins, occurrence hashes and ledger counts verified",
    }
    _write_json(output, proof)
    return proof


def aggregate_verification_reports(
    inventory: dict,
    selection_reports: list[dict],
    verification_reports: list[dict],
    selection_summary: dict,
) -> dict:
    """Accept complete disjoint verification proofs for the globally validated selection."""
    digest = _digest(inventory)
    if not selection_summary["complete_cluster"] or selection_summary["inventory_sha256"] != digest:
        raise ValueError("Verification aggregation requires the matching complete-cluster selection summary")
    task_count = selection_summary["task_count"]
    if sorted(report["config"]["task_index"] for report in selection_reports) != list(range(task_count)) or sorted(
        proof["task_index"] for proof in verification_reports
    ) != list(range(task_count)):
        raise ValueError("Verification does not cover every selection task exactly once")
    selections = {report["config"]["task_index"]: report for report in selection_reports}
    all_paths: set[str] = set()
    verified_rows = 0
    verified_shards = 0
    for proof in verification_reports:
        report = selections[proof["task_index"]]
        config = report["config"]
        paths = _ledger_paths(report)
        if (
            not report["complete"]
            or report["sampled"]
            or config["inventory_sha256"] != digest
            or config["target_cluster"] != selection_summary["target_cluster"]
            or config["task_count"] != task_count
            or config["max_shards"] is not None
            or proof["inventory_sha256"] != digest
            or proof["task_count"] != task_count
            or proof["target_cluster"] != config["target_cluster"]
            or proof["exact_catalog_root"] != config["exact_catalog_root"]
            or proof["fuzzy_catalog_root"] != config["fuzzy_catalog_root"]
            or proof["selection_report_sha256"] != _digest(report)
            or proof["ordered_outputs_sha256"] != _digest(paths)
            or proof["verified_shards"] != len(paths)
            or proof["verified_occurrences"] != report["selected_occurrences"]
            or len(paths) != len(set(paths))
            or all_paths.intersection(paths)
        ):
            raise ValueError("Verification proof does not match its unique completed selection outputs")
        all_paths.update(paths)
        verified_rows += proof["verified_occurrences"]
        verified_shards += proof["verified_shards"]
    summary_paths = [path for source in selection_summary["sources"].values() for path in source["ledgers"]]
    if (
        len(summary_paths) != len(all_paths)
        or set(summary_paths) != all_paths
        or verified_rows != selection_summary["selected_occurrences"]
        or verified_shards != selection_summary["selection_shards"]
    ):
        raise ValueError("Verified outputs do not reconcile with the full selection summary")
    return {
        "inventory_sha256": digest,
        "selection_summary_sha256": _digest(selection_summary),
        "target_cluster": selection_summary["target_cluster"],
        "task_count": task_count,
        "verified_shards": verified_shards,
        "verified_occurrences": verified_rows,
        "complete": True,
        "validation": "selection output bytes verified; original-provider reconstruction not established",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--selection-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--exact-catalog-root", required=True)
    parser.add_argument("--fuzzy-catalog-root", required=True)
    parser.add_argument("--cluster", type=int, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    info = get_job_info()
    task_index = args.task_index if args.task_index is not None else info.task_index if info is not None else None
    task_count = args.task_count if args.task_count is not None else info.num_tasks if info is not None else None
    if task_index is None or task_count is None:
        parser.error("Supply task index/count outside Iris")
    filename = f"task-{task_index:05d}-of-{task_count:05d}.json"
    proof = verify_selection_task(
        _read_json(args.inventory),
        _read_json(prefix_join(args.selection_prefix, filename)),
        prefix_join(args.output_prefix, filename),
        exact_catalog_root=args.exact_catalog_root,
        fuzzy_catalog_root=args.fuzzy_catalog_root,
        target_cluster=args.cluster,
        task_index=task_index,
        task_count=task_count,
        batch_size=args.batch_size,
    )
    logger.info("Verified %d occurrences across %d ledgers", proof["verified_occurrences"], proof["verified_shards"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

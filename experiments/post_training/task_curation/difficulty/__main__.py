# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import, inspect and reconcile Atlas difficulty attempts."""

import argparse
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from finestore.rl.difficulty import PROTOCOL_ID, Solver
from iris.client.client import get_iris_ctx
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.audit import audit_captured_runs, launch_import
from experiments.post_training.task_curation.difficulty.catalog import (
    DEFAULT_APPLET,
    available_artifact_uris,
    exclude_catalog_sources,
    served_catalog,
)
from experiments.post_training.task_curation.difficulty.controller import launch_controller, run_controller
from experiments.post_training.task_curation.difficulty.importer import catalog_sources, import_runs
from experiments.post_training.task_curation.difficulty.ledger import (
    DEFAULT_LEDGER,
    IMPORTS_DIRECTORY,
    LEDGER_VERSION,
    digest,
    read_rows,
)
from experiments.post_training.task_curation.difficulty.planner import action_id, plan
from experiments.post_training.task_curation.difficulty.report import cells_for_catalog, plan_report, status_report
from experiments.post_training.task_curation.difficulty.source_repairs import (
    record_completed_source_repairs,
    source_repair_handoffs,
)
from experiments.post_training.task_curation.difficulty.state import CoveragePolicy
from experiments.post_training.task_curation.difficulty_report import discover_run_roots


def since_time(value: str) -> datetime:
    if value.endswith("h"):
        return datetime.now(UTC) - timedelta(hours=float(value[:-1]))
    if value.endswith("d"):
        return datetime.now(UTC) - timedelta(days=float(value[:-1]))
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("--since requires a timezone or a duration such as 24h")
    return parsed.astimezone(UTC)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import", help="Backfill immutable run evidence, including ancestors")
    importer.add_argument("--run-root", action="append", default=[])
    importer.add_argument("--discover-root")
    importer.add_argument("--refresh-run-root", action="append", default=[])
    importer.add_argument("--ledger", default=DEFAULT_LEDGER)
    routes = commands.add_parser("record-source-repairs", help="Record typed handoffs from completed admission rechecks")
    routes.add_argument("--ledger", default=DEFAULT_LEDGER)
    for name in ("audit", "launch-import"):
        audit_command = commands.add_parser(name)
        audit_command.add_argument("--report", type=Path, required=True)
        audit_command.add_argument("--audit-log", type=Path, required=True)
        if name == "launch-import":
            audit_command.add_argument("--base", required=True)
            audit_command.add_argument("--ledger", default=DEFAULT_LEDGER)
            audit_command.add_argument("--job-name", required=True)
            audit_command.add_argument("--refresh-run-root", action="append", default=[])
    for name in ("status", "plan", "reconcile"):
        command = commands.add_parser(name)
        command.add_argument("--ledger", default=DEFAULT_LEDGER)
        command.add_argument("--applet-id", default=DEFAULT_APPLET)
        command.add_argument("--queue", help="Inspect a historical frozen catalog instead of the served catalog")
        command.add_argument("--queue-sha256")
        command.add_argument("--json", action="store_true")
        command.add_argument("--since", type=since_time)
        command.add_argument("--provider-available", action="append", type=Solver, default=[])
        command.add_argument("--large-protocol-id", default=PROTOCOL_ID)
        if name == "reconcile":
            command.add_argument("--once", action="store_true")
            command.add_argument("--secrets-env", type=Path, help="Load controller credentials from this dotenv file")
            budget = command.add_mutually_exclusive_group(required=True)
            budget.add_argument("--gpu-hour-budget", type=float)
            budget.add_argument("--unlimited-gpu-hours", action="store_true")
            command.add_argument("--serving-pool")
            command.add_argument("--exclude-source", action="append", default=[], help="Quarantine this source ID")
            command.add_argument("--exclude-cpu-tpu-name", action="append", default=[])
    args = parser.parse_args()
    configure_coreweave_s3()
    if args.command == "record-source-repairs":
        print(json.dumps({"handoffs": record_completed_source_repairs(args.ledger, read_rows(args.ledger))}, indent=2))
        return
    if args.command == "audit":
        proof = audit_captured_runs(args.report, args.audit_log)
        print(
            json.dumps(
                {
                    "run_roots": len(proof.run_roots),
                    "original_reasons": proof.original_reason_count,
                    "audit_log_sha256": proof.audit_log_sha256,
                    "unmapped": [],
                }
            )
        )
        return
    if args.command == "launch-import":
        launch_import(args.report, args.audit_log, args.base, args.ledger, args.job_name, tuple(args.refresh_run_root))
        return
    if args.command == "import":
        roots = args.run_root + (discover_run_roots(args.discover_root) if args.discover_root else [])
        if not roots:
            parser.error("import requires --run-root or --discover-root")
        print(json.dumps(import_runs(roots, args.ledger, frozenset([*args.refresh_run_root, *args.run_root])), indent=2))
        return
    if args.command == "reconcile" and args.queue:
        parser.error("reconcile uses the complete current served catalog")
    if args.command == "reconcile" and get_iris_ctx() is not None:
        run_controller(
            args.ledger,
            args.applet_id,
            args.once,
            frozenset(args.provider_available),
            args.gpu_hour_budget,
            args.serving_pool,
            excluded_sources=frozenset(args.exclude_source),
            excluded_cpu_tpu_names=tuple(args.exclude_cpu_tpu_name),
            large_protocol_id=args.large_protocol_id,
        )
        return
    if bool(args.queue) != bool(args.queue_sha256):
        parser.error("historical mode requires --queue and --queue-sha256")
    catalog = (
        catalog_sources(args.queue, args.queue_sha256)
        if args.queue
        else served_catalog(args.applet_id, ledger=args.ledger)
    )
    if args.command == "reconcile":
        catalog = exclude_catalog_sources(catalog, frozenset(args.exclude_source))
    rows = read_rows(args.ledger)
    policy = CoveragePolicy(large_protocol_id=args.large_protocol_id)
    cells = cells_for_catalog(rows, catalog, policy)
    # Planning snapshots current existence. The remediation functions remain pure.
    available = available_artifact_uris(cells)
    actions = plan(cells, rows, frozenset(args.provider_available), available)
    if args.command == "reconcile":
        if not (StoragePath(args.ledger) / LEDGER_VERSION / IMPORTS_DIRECTORY / "*.json").glob():
            raise ValueError("Initial backfill is unsealed; finish import before reconcile")
        if not actions:
            print(json.dumps({"submitted": [], "state": "no_new_evidence"}))
            return
        launch_controller(
            args.ledger,
            args.applet_id,
            args.once,
            frozenset(args.provider_available),
            digest([action_id(action) for action in actions]),
            args.gpu_hour_budget,
            args.secrets_env,
            args.serving_pool,
            excluded_sources=frozenset(args.exclude_source),
            excluded_cpu_tpu_names=tuple(args.exclude_cpu_tpu_name),
            large_protocol_id=args.large_protocol_id,
        )
        return
    if args.command == "plan":
        result = plan_report(actions)
    else:
        result = status_report(rows, catalog, policy, args.since, actions)
        result["source_repairs"] = [asdict(handoff) for handoff in source_repair_handoffs(args.ledger)]
    if args.json:
        print(json.dumps(result, indent=2))
    elif args.command == "plan":
        print(json.dumps(result["summary"], sort_keys=True))
        print(f"GPU-hour upper bound: {result['max_gpu_hours']:.1f}")
        for action in result["actions"]:
            print(json.dumps(action, sort_keys=True))
    else:
        print("solver     measured needs_attempts blocked terminal scored_since GPU-h_since")
        for solver, report in result["solvers"].items():
            counts = report["cells"]
            print(
                f"{solver:10} {counts.get('MEASURED', 0):8} {counts.get('NEEDS_ATTEMPTS', 0):14} "
                f"{counts.get('BLOCKED', 0):7} {counts.get('TERMINAL', 0):8} "
                f"{report['scored_attempts_since']:12} {report['gpu_hours_since']:11.3f}"
            )
        for cell in result["cells"]:
            if cell["state"] != "MEASURED":
                print(json.dumps(cell, sort_keys=True))


if __name__ == "__main__":
    main()

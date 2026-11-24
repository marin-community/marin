# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The lane's CLI: tasks, plan, run, publish, report.

``run`` is the live gate: it refuses to touch the relay without ``--allow-live`` plus the
relay credentials, because generation starts only after the coordinator releases the Ortet
relay (DESIGN.md §7; Phase 2b released it, BRIEF.md 2026-11-24). ``tasks``, ``plan``,
``report`` and ``publish`` never call a model.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Mapping

import fsspec
from iris.cluster.client import get_job_info
from rigging.filesystem.factory import open_url
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret
from shellbox.machine import MachineFactory
from taskcompendium.models import TaskSpec
from taskforge.effects.log import JsonlSink
from taskforge.policy import Policy

from experiments.post_training.task_curation.balanced_reasoning.bands import Band
from experiments.post_training.task_curation.balanced_reasoning.catalog import (
    CENSUS_ROOT,
    LEDGER_ROOT,
    build_tasks_manifest,
    read_census_entries,
)
from experiments.post_training.task_curation.balanced_reasoning.difficulty import (
    SOLVER_ALL,
    read_ledger_attempts,
    tally_attempts,
    url_protocol,
)
from experiments.post_training.task_curation.balanced_reasoning.lane import (
    KIND_GRACEFUL,
    RESOLUTION_UNSOLVED,
    run_selected_task,
    write_workdir_text,
)
from experiments.post_training.task_curation.balanced_reasoning.runtime import (
    LANE_GENERATOR_MODEL,
    lane_factories,
    lane_policy,
    lane_runtime,
    relay_or_fixed_source,
)
from experiments.post_training.task_curation.balanced_reasoning.selection import (
    DEFAULT_QUOTA,
    CandidateTask,
    Quota,
    read_selection,
    select_tasks,
    write_selection,
)
from experiments.post_training.task_curation.balanced_reasoning.store import (
    BlendFilter,
    build_store,
    publish_store,
)

logger = logging.getLogger(__name__)

GATE_MESSAGE = (
    "Generation is gated: pass --allow-live only after the coordinator has released the Ortet "
    "relay to this lane (Phase 2b, BRIEF.md 2026-11-24), with GLM_BASE_URL or GLM_RELAY_JOB set."
)


def read_candidate_tasks(path: str) -> list[CandidateTask]:
    """Read the verified-task manifest: one JSON object per line."""
    with fsspec.open(path, "rt") as source:
        candidates = []
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            candidates.append(
                CandidateTask(
                    task_id=row["task_id"],
                    source_id=row["source_id"],
                    task_sha256=row["task_sha256"],
                    domain=row.get("domain", "unknown"),
                    task_path=row["task_path"],
                )
            )
        return candidates


def read_task_spec(path: str) -> TaskSpec:
    """Load one task's ``TaskSpec`` from its JSON file."""
    with open_url(path, "rt") as source:
        return TaskSpec.model_validate_json(source.read())


def quota_from_arguments(arguments: argparse.Namespace) -> Quota:
    """The band quota, with CLI overrides over the v1 defaults."""
    overrides = {
        Band.EASY: arguments.easy_quota,
        Band.MEDIUM: arguments.medium_quota,
        Band.HARD: arguments.hard_quota,
        Band.UNSOLVED: arguments.unsolved_quota,
    }
    per_band = {
        band: overrides[band] if overrides[band] is not None else DEFAULT_QUOTA.per_band[band]
        for band in DEFAULT_QUOTA.per_band
    }
    return Quota(per_band)


def manifest_machines() -> tuple[Policy | None, Mapping[str, MachineFactory] | None]:
    """The lowering pre-check inputs when running inside an Iris job; ``None`` off-cluster."""
    try:
        if get_job_info() is None:
            return None, None
    except Exception:
        return None, None
    policy = lane_policy({})
    return policy, lane_factories(policy)


def command_plan(args: argparse.Namespace) -> int:
    candidates = read_candidate_tasks(args.tasks)
    difficulty = tally_attempts(read_ledger_attempts(args.ledger), solver=args.solver)
    selected, census = select_tasks(candidates, difficulty, quota_from_arguments(args))
    digest_value = write_selection(f"{args.output.rstrip('/')}/selection.json", selected, census)
    print(f"PLAN selected={len(selected)} digest={digest_value}")
    print(json.dumps(census, indent=2, sort_keys=True))
    return 0


def command_tasks(args: argparse.Namespace) -> int:
    """Build the verified-task manifest: ledger difficulty joined to census TaskSpecs."""
    difficulty = tally_attempts(read_ledger_attempts(args.ledger), solver=args.solver)
    entries = read_census_entries(args.census_root)
    policy, factories = manifest_machines()
    receipt = build_tasks_manifest(
        difficulty,
        entries,
        args.output,
        min_attempts=args.min_attempts,
        policy=policy,
        factories=factories,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def command_run(args: argparse.Namespace) -> int:
    if not args.allow_live:
        print(GATE_MESSAGE, file=sys.stderr)
        return 2
    # Fail fast on a missing or empty credential: the fleet's "Illegal header value
    # b'Bearer '" failure was an empty api-key file reaching the wire.
    token = require_secret("GLM_API_KEY")
    base_url = os.environ.get("GLM_BASE_URL")
    relay_job = os.environ.get("GLM_RELAY_JOB")
    digest_value, selected, _ = read_selection(args.selection)
    logger.info("RUN selection=%s tasks=%d", digest_value, len(selected))
    policy = lane_policy(
        {
            "provider.pool": args.pool,
            "sampling.temperature": args.temperature,
            "sampling.max_continuations": 0,
            "author.model": LANE_GENERATOR_MODEL,
        }
    )
    source = relay_or_fixed_source(base_url, relay_job, token, pool=args.pool)

    async def drive() -> list:
        semaphore = asyncio.Semaphore(args.concurrency)
        results = []

        async def one(entry) -> None:
            async with semaphore:
                task = read_task_spec(entry.task.task_path)
                workdir = f"{args.run_root.rstrip('/')}/tasks/{entry.task.task_id}"
                try:
                    async with lane_runtime(policy, source) as (resources, _factories):
                        outcome = await run_selected_task(entry, task, policy, resources, workdir)
                        results.append(outcome)
                except Exception as error:  # one task's failure must not stop the round
                    logger.exception("TASK %s failed", entry.task.task_id)
                    write_workdir_text(workdir, "status.json", json.dumps({"failed": str(error)}))
                    results.append(None)

        await asyncio.gather(*(one(entry) for entry in selected))
        return results

    outcomes = asyncio.run(drive())
    completed = sum(1 for outcome in outcomes if outcome is not None)
    print(f"RUN completed={completed}/{len(selected)}")
    return 0 if completed == len(selected) else 1


def command_publish(args: argparse.Namespace) -> int:
    blend = BlendFilter.from_jsonl(args.blend_prompts) if args.blend_prompts else None
    manifest = build_store(args.run_root, selection_path=args.selection, blend=blend)
    if args.destination:
        written = publish_store(f"{args.run_root.rstrip('/')}/store", args.destination)
        print(f"PUBLISH {written}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def command_report(args: argparse.Namespace) -> int:
    """The pilot's cost meter: tokens, latency, retries, resolutions and solve rates by band."""
    protocol = url_protocol(args.run_root)
    filesystem = fsspec.filesystem(protocol)
    root = f"{args.run_root.rstrip('/')}/tasks"

    def full(path: str) -> str:
        # Object-store globs return bucket-key paths without the scheme; restore it.
        return path if protocol == "file" or "://" in path else f"{protocol}://{path}"

    bands: dict[str, dict[str, float]] = {}
    resolutions: dict[str, int] = {}
    graceful_forms: dict[str, int] = {}
    solve_by_band: dict[str, dict[str, int]] = {}
    retry_markers: dict[str, int] = {}
    completions = 0
    completions_with_retries = 0

    def bucket(name: str) -> dict[str, float]:
        return bands.setdefault(
            name,
            {
                "tasks": 0,
                "attempts": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "wall_seconds": 0.0,
            },
        )

    for attempt_path in sorted(filesystem.glob(f"{root}/*/attempt.json")):
        attempt_url = full(attempt_path)
        with fsspec.open(attempt_url, "rt") as source:
            document = json.load(source)
        name = document["band"]
        stats = bucket(name)
        stats["tasks"] += 1
        stats["attempts"] += len(document["attempts"])
        for attempt in document["attempts"]:
            stats["prompt_tokens"] += attempt["prompt_tokens"]
            stats["completion_tokens"] += attempt["completion_tokens"]
            stats["reasoning_tokens"] += attempt.get("reasoning_tokens") or 0
        resolutions[document["resolution_kind"]] = resolutions.get(document["resolution_kind"], 0) + 1
        task_solve = solve_by_band.setdefault(name, {"solved": 0, "tasks": 0})
        task_solve["tasks"] += 1
        if document["selected_index"] is not None and document["resolution_kind"] != RESOLUTION_UNSOLVED:
            task_solve["solved"] += 1
        for attempt in document["attempts"]:
            if attempt.get("kind") == KIND_GRACEFUL and attempt.get("graceful_resolution"):
                graceful_forms[attempt["graceful_resolution"]] = (
                    graceful_forms.get(attempt["graceful_resolution"], 0) + 1
                )

        events_url = attempt_url.replace("attempt.json", "events.jsonl")
        try:
            sink = JsonlSink(events_url)
            content = sink.read_bytes()
        except FileNotFoundError:
            continue
        for line in content.splitlines():
            record = json.loads(line)
            stats["wall_seconds"] += float(record.get("wall_time") or 0.0)
            event = record.get("event") or {}
            if not str(event.get("type", "")).endswith("Resolved"):
                continue
            value = (event.get("payload") or {}).get("value")
            if not isinstance(value, dict) or "usage" not in value:
                continue
            completions += 1
            retries = value.get("retries") or []
            if retries:
                completions_with_retries += 1
            for marker in retries:
                outcome = str(marker).split(":", 1)[0]
                retry_markers[outcome] = retry_markers.get(outcome, 0) + 1

    for name, stats in bands.items():
        tokens = stats["completion_tokens"]
        stats["wall_tok_per_s"] = round(tokens / stats["wall_seconds"], 1) if stats["wall_seconds"] else 0.0
        del stats["decode_seconds"]
        stats["solve_rate"] = (
            round(solve_by_band[name]["solved"] / solve_by_band[name]["tasks"], 3)
            if solve_by_band[name]["tasks"]
            else 0.0
        )
    graceful_passes = graceful_forms.get("graceful_best_effort", 0) + graceful_forms.get("graceful_uncertain", 0)
    graceful_attempts = sum(graceful_forms.values())
    report = {
        "bands": bands,
        "resolutions": resolutions,
        "solve_by_band": solve_by_band,
        "graceful": {
            "forms": graceful_forms,
            "attempts": graceful_attempts,
            "form_pass_rate": round(graceful_passes / graceful_attempts, 3) if graceful_attempts else None,
        },
        "transport": {
            "completions": completions,
            "completions_with_retries": completions_with_retries,
            "retry_markers": retry_markers,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="balanced_reasoning", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="band and quota the verified tasks (no model calls)")
    plan.add_argument(
        "--tasks", required=True, help="JSONL of verified tasks: task_id, source_id, task_sha256, domain, task_path"
    )
    plan.add_argument("--ledger", default=LEDGER_ROOT, help="Atlas difficulty ledger root")
    plan.add_argument("--output", required=True, help="Run root for selection.json")
    plan.add_argument(
        "--solver",
        default=SOLVER_ALL,
        help="Solver family to band by; 'all' pools every pinned solver (the measured ledger "
        "carries ~1 hosted attempt per task, which cannot band)",
    )
    plan.add_argument("--easy-quota", type=int, default=None)
    plan.add_argument("--medium-quota", type=int, default=None)
    plan.add_argument("--hard-quota", type=int, default=None)
    plan.add_argument("--unsolved-quota", type=int, default=None)
    plan.set_defaults(handler=command_plan)

    tasks = commands.add_parser("tasks", help="build the verified-task manifest (no model calls)")
    tasks.add_argument("--ledger", default=LEDGER_ROOT, help="Atlas difficulty ledger root")
    tasks.add_argument("--census-root", default=CENSUS_ROOT, help="Retained-census release root")
    tasks.add_argument("--output", required=True, help="Output prefix for tasks.jsonl and task specs")
    tasks.add_argument("--solver", default=SOLVER_ALL, help="Solver family to tally difficulty under")
    tasks.add_argument(
        "--min-attempts", type=int, default=1, help="Drop ledger tasks scored on fewer attempts than this"
    )
    tasks.set_defaults(handler=command_tasks)

    run = commands.add_parser("run", help="generate and verify traces for a selection (gated)")
    run.add_argument("--selection", required=True)
    run.add_argument("--run-root", required=True)
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--pool", default="bulk")
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument("--allow-live", action="store_true", help="acknowledge the relay gate is released")
    run.set_defaults(handler=command_run)

    publish = commands.add_parser("publish", help="build (and optionally upload) the SFT store bundle")
    publish.add_argument("--run-root", required=True)
    publish.add_argument("--selection", required=True)
    publish.add_argument("--blend-prompts", default=None, help="JSONL of RL blend prompts to decontaminate against")
    publish.add_argument("--destination", default=None, help="artifact prefix under the canonical root")
    publish.set_defaults(handler=command_publish)

    report = commands.add_parser("report", help="token and resolution totals by band")
    report.add_argument("--run-root", required=True)
    report.set_defaults(handler=command_report)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    # A no-op inside CoreWeave pods (runtime config wins); configures fsspec on a dev box.
    configure_coreweave_s3()
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

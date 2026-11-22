# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The lane's CLI: plan, run, publish, report.

``run`` is the live gate: it refuses to touch the relay without ``--allow-live`` plus the
relay credentials, because generation starts only after the coordinator releases the Ortet
relay (DESIGN.md §7). ``plan``, ``report`` and ``publish`` never call a model.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import fsspec
from rigging.filesystem.factory import open_url
from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation.balanced_reasoning.bands import Band
from experiments.post_training.task_curation.balanced_reasoning.difficulty import (
    DEFAULT_LEDGER,
    read_ledger_attempts,
    tally_attempts,
)
from experiments.post_training.task_curation.balanced_reasoning.lane import run_selected_task
from experiments.post_training.task_curation.balanced_reasoning.runtime import (
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
    "Generation is gated: the Ortet relay stays with the tool-conversion lane until the "
    "coordinator releases it. Pass --allow-live with relay credentials only after that release."
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


def command_plan(args: argparse.Namespace) -> int:
    candidates = read_candidate_tasks(args.tasks)
    ledger_root = DEFAULT_LEDGER if args.difficulty_ledger is None else args.difficulty_ledger
    difficulty = tally_attempts(read_ledger_attempts(ledger_root), solver=args.solver)
    selected, census = select_tasks(candidates, difficulty, quota_from_arguments(args))
    digest_value = write_selection(f"{args.output.rstrip('/')}/selection.json", selected, census)
    print(f"PLAN selected={len(selected)} digest={digest_value}")
    print(json.dumps(census, indent=2, sort_keys=True))
    return 0


def command_run(args: argparse.Namespace) -> int:
    token = os.environ.get("GLM_API_KEY")
    base_url = os.environ.get("GLM_BASE_URL")
    relay_job = os.environ.get("GLM_RELAY_JOB")
    if not args.allow_live or not token or bool(base_url) == bool(relay_job):
        print(GATE_MESSAGE, file=sys.stderr)
        return 2
    digest_value, selected, _ = read_selection(args.selection)
    logger.info("RUN selection=%s tasks=%d", digest_value, len(selected))
    policy = lane_policy(
        {
            "provider.pool": args.pool,
            "sampling.temperature": args.temperature,
            "sampling.max_continuations": 0,
        }
    )
    source = relay_or_fixed_source(base_url, relay_job, token, pool=args.pool)

    async def drive() -> list:
        semaphore = asyncio.Semaphore(args.concurrency)
        results = []

        async def one(entry) -> None:
            async with semaphore:
                task = read_task_spec(entry.task.task_path)
                workdir = Path(args.run_root) / "tasks" / entry.task.task_id
                try:
                    async with lane_runtime(policy, source) as (resources, _factories):
                        outcome = await run_selected_task(entry, task, policy, resources, workdir)
                        results.append(outcome)
                except Exception as error:  # one task's failure must not stop the round
                    logger.exception("TASK %s failed", entry.task.task_id)
                    workdir.mkdir(parents=True, exist_ok=True)
                    (workdir / "status.json").write_text(json.dumps({"failed": str(error)}))
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
    """Token and resolution totals over a run root, by band -- the pilot's cost meter."""
    root = Path(args.run_root) / "tasks"
    totals: dict[str, dict[str, int]] = {}
    for attempt_path in sorted(root.glob("*/attempt.json")):
        document = json.loads(attempt_path.read_text())
        bucket = totals.setdefault(
            document["band"],
            {"tasks": 0, "prompt_tokens": 0, "completion_tokens": 0, "attempts": 0},
        )
        bucket["tasks"] += 1
        bucket["attempts"] += len(document["attempts"])
        for attempt in document["attempts"]:
            bucket["prompt_tokens"] += attempt["prompt_tokens"]
            bucket["completion_tokens"] += attempt["completion_tokens"]
    print(json.dumps(totals, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="balanced_reasoning", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="band and quota the verified tasks (no model calls)")
    plan.add_argument(
        "--tasks", required=True, help="JSONL of verified tasks: task_id, source_id, task_sha256, domain, task_path"
    )
    plan.add_argument(
        "--difficulty-ledger", default=None, help="Atlas difficulty ledger root; defaults to the production ledger"
    )
    plan.add_argument("--output", required=True, help="Run root for selection.json")
    plan.add_argument("--solver", default="hosted", help="Solver family to band by")
    plan.add_argument("--easy-quota", type=int, default=None)
    plan.add_argument("--medium-quota", type=int, default=None)
    plan.add_argument("--hard-quota", type=int, default=None)
    plan.add_argument("--unsolved-quota", type=int, default=None)
    plan.set_defaults(handler=command_plan)

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
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

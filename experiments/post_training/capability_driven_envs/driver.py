# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Taskforge's loop over the capability catalog with scripted models.

Every catalog capability (or each ``--capability`` named) becomes an idea that ``RecordSource``
proposes from; ``scaling_task.PROGRAM`` builds each proposal without a model and ``scripted_solver``
runs its validation trials. The loop keeps each item's rounds (its draft, ``task.json`` among it)
under ``ROOT/items/``; this script writes ``ROOT/summary.json``, every item's terminal and every
accepted task, and prints its path. A relaunch on the same root resumes every item from its event
log::

    uv run --package marin-taskforge --frozen python -m \\
        experiments.post_training.capability_driven_envs.driver --root /tmp/capability-run
"""

import argparse
import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter
from taskforge.atomic_file import write_atomic
from taskforge.builder.author import adopt
from taskforge.builder.run import DRAFT_DIR, item_id_for, run_build
from taskforge.ledger.jsonl import JsonlLedger, read_entries
from taskforge.llm.client import NoModel
from taskforge.llm.policy import LLMPolicy
from taskforge.loop.events import EventKind, ProposalOrigin, Terminal, events
from taskforge.loop.policy import LoopPolicy, install_policy
from taskforge.loop.program import ITEMS_DIR, LEDGER_DIR, ROUNDS_DIR, Loop, run_idea, run_item
from taskforge.loop.stages import Stages
from taskforge.review.decision import BandOutcome
from taskforge.review.rules import BandChoice, BandRules, decide
from taskforge.sandbox.factories import factory_capabilities, machine_factories
from taskforge.triage.verdict import accept
from taskforge.validate.calibration import CalibrationBand
from taskforge.validate.run import ValidationPolicy, run_validation
from taskforge.validate.solver import ModelFactory
from taskforge.validate.trials import Deadlines, EngineSettings, RetryBackoff

from experiments.post_training.capability_driven_envs.catalog import (
    CapabilityIdea,
    capability_idea_record,
    load_capability_ideas,
)
from experiments.post_training.capability_driven_envs.scaling_task import PROGRAM, scripted_solver
from experiments.post_training.capability_driven_envs.source import RecordSource

CATALOG = Path(__file__).resolve().parent / "data" / "catalog.json"
SUMMARY_FILE = "summary.json"
BACKOFF = RetryBackoff(initial=1.0, maximum=10.0, factor=2.0, jitter=0.1)


def capability_policy(k: int, proposals_per_idea: int) -> LoopPolicy:
    """The capability run's policy: a too-easy task is accepted and labelled with its band, a too-hard one
    rejected."""
    return LoopPolicy(
        proposals_per_idea=proposals_per_idea,
        max_idea_reproposals=1,
        max_build_revisions=1,
        max_validation_retries=2,
        retry_backoff=BACKOFF,
        band_rules=BandRules(too_easy=BandChoice.ACCEPT, too_hard=BandChoice.REJECT),
        validation=ValidationPolicy(
            k=k,
            band=CalibrationBand(min_solve_rate=0.125, max_solve_rate=0.875),
            sampling=LLMPolicy(max_continuations=0),
            deadlines=Deadlines(total_turn_timeout=60.0, attempt_timeout=120.0),
            max_retries=1,
            token_contract_retries=0,
            retry_backoff=BACKOFF,
        ),
    )


def engine_settings() -> EngineSettings:
    """The ShellSim factories, and the session limits of every validation trial."""
    return EngineSettings(
        factories=machine_factories(),
        capabilities=factory_capabilities(),
        max_turns=8,
        command_timeout=30.0,
        tool_turn_timeout=60.0,
        model_turn_timeout=60.0,
        cleanup_timeout=30.0,
    )


def capability_loop(root: Path, policy: LoopPolicy, rollout_models: ModelFactory) -> Loop[CapabilityIdea]:
    """The library's stage functions, with ``PROGRAM`` as every item's program and no proposing or building model."""
    stages = Stages(
        propose=RecordSource().propose,
        triage=accept,
        author=adopt(PROGRAM),
        build=run_build,
        validate=run_validation,
        review=decide,
    )
    return Loop(
        stages=stages,
        policy=install_policy(root, policy),
        root=root,
        ledger=JsonlLedger(root / LEDGER_DIR),
        engine=engine_settings(),
        rollout_models=rollout_models,
        model=NoModel(),
        sampling=LLMPolicy(),
        describe_idea=capability_idea_record,
    )


def selected_ideas(catalog: Path, capabilities: Sequence[str]) -> dict[str, CapabilityIdea]:
    """The catalog's ideas, or only ``capabilities`` when any are named; an unknown id raises."""
    ideas = load_capability_ideas(catalog)
    if not capabilities:
        return ideas
    unknown = sorted(set(capabilities) - set(ideas))
    if unknown:
        raise ValueError(f"{catalog} has no capabilities {unknown}")
    return {capability: ideas[capability] for capability in capabilities}


@dataclass(frozen=True)
class AcceptedTask:
    """An accepted item's task: its round, digest and draft directory (relative to the root), and its band."""

    round: int
    task_digest: str
    draft: str
    band: BandOutcome


@dataclass(frozen=True)
class RunSummary:
    """Every item's terminal, and the accepted ones' tasks."""

    items: dict[str, Terminal]
    accepted: dict[str, AcceptedTask]


SUMMARY: TypeAdapter[RunSummary] = TypeAdapter(RunSummary)


def accepted_task(root: Path, item_id: str) -> AcceptedTask:
    """The task of an accepted item, read from its last ``DECIDED`` event."""
    log = events(read_entries(JsonlLedger(root / LEDGER_DIR).path_for(item_id)))
    decided = [entry for entry in log if entry.step == EventKind.DECIDED][-1]
    assert decided.input_hash is not None, f"{item_id}: DECIDED points at its task"
    return AcceptedTask(
        round=decided.round,
        task_digest=decided.input_hash,
        draft=str(Path(ITEMS_DIR) / item_id / ROUNDS_DIR / str(decided.round) / DRAFT_DIR),
        band=BandOutcome(decided.attrs["band"]),
    )


async def run(
    root: Path, ideas: Mapping[str, CapabilityIdea], policy: LoopPolicy, rollout_models: ModelFactory
) -> RunSummary:
    """Carry every idea's proposals to a terminal, one item at a time, and write ``summary.json``."""
    loop = capability_loop(root, policy, rollout_models)
    items: dict[str, Terminal] = {}
    for idea_id, idea in ideas.items():
        for proposal in await run_idea(loop, idea_id, idea):
            items[item_id_for(proposal)] = await run_item(loop, proposal, ProposalOrigin.GENERATED)
    accepted = {item: accepted_task(root, item) for item, terminal in items.items() if terminal is Terminal.ACCEPTED}
    summary = RunSummary(items=items, accepted=accepted)
    write_atomic(root / SUMMARY_FILE, SUMMARY.dump_json(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="the run root directory")
    parser.add_argument("--catalog", type=Path, default=CATALOG, help="the capability catalog JSON file")
    parser.add_argument("--capability", action="append", default=[], help="run only this capability id; repeatable")
    parser.add_argument("--k", type=int, default=4, help="solver trials per task")
    parser.add_argument("--proposals", type=int, default=1, help="proposals per capability")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    root = args.root.resolve()
    ideas = selected_ideas(args.catalog, args.capability)
    asyncio.run(run(root, ideas, capability_policy(args.k, args.proposals), scripted_solver))
    print(root / SUMMARY_FILE)


if __name__ == "__main__":
    main()

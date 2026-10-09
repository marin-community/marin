# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Taskforge's queue over the capability catalog with scripted models.

Every catalog capability (or each ``--capability`` named) becomes an idea that ``RecordSource``
proposes from; ``ScriptedBuilder`` builds each proposal with the standard template and
``scripted_solver`` runs its validation trials. The queue writes each accepted task's draft
(``task.json`` among it) under ``ROOT/items/`` and ``ROOT/summary.json``, and this script prints the
summary's path. A relaunch on the same root resumes every item from its event log::

    uv run --project lib/taskforge --frozen python -m \\
        experiments.post_training.capability_driven_envs.driver --root /tmp/capability-run
"""

import argparse
import asyncio
import logging
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path

from taskforge.builder.sdk import ModelEndpoint
from taskforge.llm.policy import LLMPolicy
from taskforge.loop.policy import LoopPolicy
from taskforge.queue.config import EngineConfig, RunConfig
from taskforge.queue.job import SUMMARY_FILE, RunInputs, RunModel, run_job
from taskforge.queue.run import FailedItems, RunSummary
from taskforge.review.rules import BandChoice, BandRule, BandRules
from taskforge.sandbox.factories import MachineHost
from taskforge.validate.calibration import CalibrationBand
from taskforge.validate.run import ValidationPolicy
from taskforge.validate.trials import Deadlines, RetryBackoff

from experiments.post_training.capability_driven_envs.catalog import (
    CapabilityIdea,
    capability_idea_record,
    load_capability_ideas,
)
from experiments.post_training.capability_driven_envs.scaling_task import ScriptedBuilder, scripted_solver
from experiments.post_training.capability_driven_envs.source import RecordSource

CATALOG = Path(__file__).resolve().parent / "data" / "catalog.json"
IMAGE_CACHE_DIR = "image-cache"
BACKOFF = RetryBackoff(initial=1.0, maximum=10.0, factor=2.0, jitter=0.1)


def capability_policy(k: int, proposals_per_idea: int) -> LoopPolicy:
    """The capability run's policy: a task outside the band is not revised; a too-easy one is accepted and
    labelled with its band, a too-hard one rejected."""
    return LoopPolicy(
        proposals_per_idea=proposals_per_idea,
        max_idea_reproposals=1,
        max_build_revisions=1,
        max_repairs=1,
        max_validation_retries=2,
        retry_backoff=BACKOFF,
        band_rules=BandRules(
            too_easy=BandRule(repairs=0, then=BandChoice.ACCEPT), too_hard=BandRule(repairs=0, then=BandChoice.REJECT)
        ),
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


def run_config(root: Path, policy: LoopPolicy) -> RunConfig:
    """A laptop run rooted at ``root``; ShellSim machines need no image cache, but a laptop config names one."""
    engine = EngineConfig(
        max_turns=8, command_timeout=30.0, tool_turn_timeout=60.0, model_turn_timeout=60.0, cleanup_timeout=30.0
    )
    return RunConfig(
        run_id=root.name,
        root=root,
        host=MachineHost.LAPTOP,
        image_cache=root / IMAGE_CACHE_DIR,
        policy=policy,
        engine=engine,
        width=8,
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


def capability_inputs(
    ideas: Mapping[str, CapabilityIdea], client: ModelEndpoint, root: Path
) -> RunInputs[CapabilityIdea]:
    """A ``queue.job.InputsFactory`` once ``ideas`` is bound."""
    return RunInputs(ideas=ideas, source=RecordSource(), describe_idea=capability_idea_record)


def scripted_model() -> RunModel:
    return RunModel(client=ScriptedBuilder(), rollout_models=scripted_solver)


async def run(
    root: Path, ideas: Mapping[str, CapabilityIdea], policy: LoopPolicy, model: RunModel, failed: FailedItems
) -> RunSummary:
    return await run_job(run_config(root, policy), partial(capability_inputs, ideas), failed, model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="the run root, an absolute directory")
    parser.add_argument("--catalog", type=Path, default=CATALOG, help="the capability catalog JSON file")
    parser.add_argument("--capability", action="append", default=[], help="run only this capability id; repeatable")
    parser.add_argument("--k", type=int, default=4, help="solver trials per task")
    parser.add_argument("--proposals", type=int, default=1, help="proposals per capability")
    parser.add_argument("--retry-failed", action="store_true", help="re-enter items that ended FAILED")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    failed = FailedItems.RETRY if args.retry_failed else FailedItems.SKIP
    policy = capability_policy(args.k, args.proposals)
    ideas = selected_ideas(args.catalog, args.capability)
    asyncio.run(run(args.root.resolve(), ideas, policy, scripted_model(), failed))
    print(args.root.resolve() / SUMMARY_FILE)


if __name__ == "__main__":
    main()

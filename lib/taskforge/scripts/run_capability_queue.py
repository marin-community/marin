# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an unattended Taskforge queue over the capability catalog.

Every catalog capability (or each ``--capability`` named) becomes a ``CapabilityIdea``; the run
proposes from it with ``CapabilitySource``, triages with the structural checks and a ``GlmRubric``
of ``--rubric-samples`` samples that is shown each capability's catalog record, and carries every
proposal through ``queue.run.run_queue``. ``CONFIG`` is a ``queue.config.RunConfig`` file
(``docs/policy.example.json`` is the committed unattended configuration; set its ``relay_job``).
The run writes ``summary.json`` into the run root and exits non-zero when any item ended ``FAILED``.

Laptop, through the GLM port-forward on the interactive pool (a copy of the example with
``"host": "laptop"``, ``"image_cache"`` a local directory and
``"glm": {"kind": "laptop", ..., "pool": "high"}``)::

    cd lib/taskforge && uv run python scripts/run_capability_queue.py run.json \\
      --catalog <catalog file> --rubric-samples 3 --capability d01.algebra.linear-transformations

Iris takes the same arguments in place of ``scripts/run_queue.py``'s ``--inputs`` (see that
script's docstring for the job command); the catalog is not checked in, so ``--catalog`` names a
file the job can read. A relaunch on the same root resumes every item from its event log.
"""

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path

from taskforge.llm.client import GlmClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.store import CallStore
from taskforge.loop.events import Terminal
from taskforge.proposal.sources.capability import (
    CapabilityIdea,
    CapabilitySource,
    capability_prompt_record,
    load_capability_ideas,
    source_ref,
)
from taskforge.queue.config import load_run_config
from taskforge.queue.job import RunInputs, run_job
from taskforge.queue.run import FailedItems
from taskforge.triage.checks import ALL_COMBINATIONS, CHECKS, CheckContext
from taskforge.triage.program import GlmRubric

CALLS_DIR = "calls"
# Proposal and rubric calls sample at the model maximum.
PROPOSAL_POLICY = LLMPolicy()


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
    ideas: Mapping[str, CapabilityIdea], rubric_samples: int, client: GlmClient, root: Path
) -> RunInputs[CapabilityIdea]:
    """A ``queue.job.InputsFactory`` once ``ideas`` and ``rubric_samples`` are bound.

    The rubric's calls are kept under ``root / CALLS_DIR``, so a relaunch on the same root replays them.
    """
    records = {source_ref(idea): capability_prompt_record(idea) for idea in ideas.values()}
    rubric = GlmRubric(CallStore(root / CALLS_DIR, client), PROPOSAL_POLICY, rubric_samples, records)
    return RunInputs(
        ideas=ideas,
        source=CapabilitySource(client, PROPOSAL_POLICY),
        checks=CHECKS,
        rubric=rubric,
        check_context=CheckContext(allowed_combinations=ALL_COMBINATIONS),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--catalog", type=Path, required=True, help="the capability catalog JSON file")
    parser.add_argument("--rubric-samples", type=int, required=True, help="independent triage rubric samples")
    parser.add_argument("--capability", action="append", default=[], help="run only this capability id; repeatable")
    parser.add_argument("--retry-failed", action="store_true", help="re-enter items that ended FAILED")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_run_config(args.config)
    ideas = selected_ideas(args.catalog.expanduser(), args.capability)
    failed = FailedItems.RETRY if args.retry_failed else FailedItems.SKIP
    summary = asyncio.run(run_job(config, partial(capability_inputs, ideas, args.rubric_samples), failed))
    print("RUN_SUMMARY " + json.dumps(summary.summary_json()), flush=True)
    if Terminal.FAILED in summary.items.values() or summary.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

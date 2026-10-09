# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The boundary between a run's configuration and the host it runs on.

``run_job`` is the one entry point. It places the run root, copies the policy in, takes the machine
factories for the host, builds the run's services, runs ``queue.run.run_queue`` and writes
``summary.json``. The run's models come with its inputs (``RunInputs.model`` and
``RunInputs.rollout_models``). A run lives on a laptop; its ledger is the per-item JSONL files under
the run root.
"""

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from taskforge.atomic_file import write_atomic
from taskforge.builder.sdk import BuildServices, ModelEndpoint
from taskforge.builder.template import standard
from taskforge.content_hash import pretty_json
from taskforge.ledger.jsonl import JsonlLedger
from taskforge.ledger.records import Ledger
from taskforge.llm.policy import LLMPolicy
from taskforge.loop.policy import POLICY
from taskforge.loop.program import LEDGER_DIR, LoopServices
from taskforge.proposal.source import ProposalSource
from taskforge.queue.config import RunConfig
from taskforge.queue.run import FailedItems, RunSummary, run_queue
from taskforge.sandbox.factories import MachineHost, factory_capabilities, machine_factories
from taskforge.validate.solver import ModelFactory

POLICY_FILE = "policy.json"
SUMMARY_FILE = "summary.json"
# Builders sample at the model maximum and continue on length; only validation rollouts may not continue.
BUILD_POLICY = LLMPolicy()


def run_root(config: RunConfig) -> Path:
    """The run root on this host: ``config.root`` on a laptop.

    Raises:
        ValueError: the config names the Iris host.
    """
    if config.host is not MachineHost.LAPTOP:
        raise ValueError(f"a {config.host} run root lives under $IRIS_OUTPUT_DIR; run_job runs on a laptop")
    return config.root


@dataclass(frozen=True)
class RunInputs[IdeaT]:
    """What a run takes beyond its config: its ideas, the source that proposes from them, each idea's
    record (``LoopServices.describe_idea``), and its models.

    The capability layer supplies these for the capability catalog.

    Attributes:
        model: The model every build step calls. A transient failure must raise ``GlmUnavailable``: the
            item then ends ``FAILED`` and a launch that retries failed items re-enters it, where any
            other exception costs a build revision.
        rollout_models: Builds each solver trial's rollout model. A transient failure must raise
            ``GlmUnavailable`` (``MODEL_UNAVAILABLE``), which the trial and then a ``Retry`` run again. Any
            other exception settles the trial ``UNCLASSIFIED``: every ``Retry`` reads it back from disk,
            and the item ends ``ABANDONED`` on every launch.
    """

    ideas: Mapping[str, IdeaT]
    source: ProposalSource[IdeaT]
    describe_idea: Callable[[IdeaT], Mapping[str, object]]
    model: ModelEndpoint
    rollout_models: ModelFactory


type InputsFactory[IdeaT] = Callable[[Path], RunInputs[IdeaT]]
"""Builds a run's inputs from the run root."""


def prepare_root(config: RunConfig) -> Path:
    """Place the run root on this host and copy the policy in, or check it against the policy already there.

    Raises:
        ValueError: the root's ``policy.json`` has another digest (``LoopPolicy.digest``) than ``config.policy``.
    """
    root = run_root(config)
    root.mkdir(parents=True, exist_ok=True)
    policy_file = root / POLICY_FILE
    if not policy_file.exists():
        write_atomic(policy_file, POLICY.dump_json(config.policy, indent=2))
        return root
    stored = POLICY.validate_json(policy_file.read_bytes()).digest
    if stored != config.policy.digest:
        raise ValueError(
            f"{root} was started under policy {stored}, not the requested {config.policy.digest}; relaunch with "
            "the policy this root started with, or use a new run root"
        )
    return root


@asynccontextmanager
async def loop_services[IdeaT](
    config: RunConfig, root: Path, ledger: Ledger, inputs: InputsFactory[IdeaT]
) -> AsyncIterator[tuple[LoopServices[IdeaT], Mapping[str, IdeaT]]]:
    """The run's ``LoopServices`` and ideas: the inputs' models, the host's factories, ``width`` slots.

    Builders sample at ``BUILD_POLICY``.
    """
    factories = machine_factories(config.host, None, config.image_cache)
    run_inputs = inputs(root)
    services = LoopServices(
        source=run_inputs.source,
        describe_idea=run_inputs.describe_idea,
        template=standard,
        build=BuildServices(
            client=run_inputs.model, policy=BUILD_POLICY, host=config.host, factories=factories, ledger=ledger
        ),
        engine=config.engine.settings(factories, factory_capabilities(config.host)),
        rollout_models=run_inputs.rollout_models,
        ledger=ledger,
        root=root,
        slots=asyncio.Semaphore(config.width),
    )
    yield services, run_inputs.ideas


async def run_job[IdeaT](config: RunConfig, inputs: InputsFactory[IdeaT], failed: FailedItems) -> RunSummary:
    """Run ``config`` on this host to completion and write ``summary.json`` into the run root."""
    root = prepare_root(config)
    ledger = JsonlLedger(root / LEDGER_DIR)
    async with loop_services(config, root, ledger, inputs) as (services, ideas):
        summary = await run_queue(ideas, config.policy, services, failed)
    write_atomic(root / SUMMARY_FILE, pretty_json({"run_id": config.run_id, **summary.summary_json()}).encode())
    return summary

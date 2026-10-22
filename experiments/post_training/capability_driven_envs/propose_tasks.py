# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Propose tasks from the capability catalog through Taskforge's stages, and triage each proposal.

The driver follows the staged path from the catalog to proposals, then to ``stage20_triage``, then
to the build, and stops before the build. ``generate_tasks`` follows the unstaged path instead: GLM
writes a program per catalog entry, and the program writes a TaskSpec directly. For each selected
capability, the driver:

1. plans ``--slots`` slots and writes a proposal per slot with
   ``proposal_source.CapabilitySource``, in the environments ``--environment`` names;
2. triages every proposal with ``stage20_triage.preflight.preflight``, Taskforge's default triage:
   the deterministic checks, with no model call.

Building accepted proposals with ``stage40_build_with_program`` and validating the drafts belong to
Taskforge's stage coordinator (``taskforge.stage_coordinator``), so the driver stops after the
verdicts. When a capability's plan stays invalid, or its planning call runs out of retries or is
rejected, the driver records the error as that capability's failure. A slot whose own model call
fails is a slot failure, and the capability keeps its plan and its other slots. Either way the
driver still writes the run's outputs:

* ``OUTPUT_DIR/<capability id>/`` with ``plan.request.json`` and ``plan.completions.json`` (the
  invalid plan replies when the plan stays invalid, and none when the planning call fails), and per
  slot ``slot-NN.request.json``, ``slot-NN.completions.json``, then ``slot-NN.md`` and
  ``slot-NN.verdict.json`` for a proposal, or ``slot-NN.failure.txt`` for a slot failure.
* ``OUTPUT_DIR/summary.json``, mapping each capability id to its failure or to the decision or
  failure of each slot.

``--base-url``, ``--relay-job`` and ``--pool`` work as in ``generate_tasks``. On a laptop, with the
GLM router port-forwarded and ``GLM_BULK_TOKEN`` exported::

    uv run --package marin-taskforge --frozen python -m \\
        experiments.post_training.capability_driven_envs.propose_tasks \\
        --output-dir /tmp/capability-proposals --base-url http://127.0.0.1:18000/v1 --pool bulk \\
        --environment reasoning --environment shellsim
"""

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import click
from rigging.log_setup import configure_logging
from taskforge.content_hash import pretty_json
from taskforge.llm.client import ModelClient, ModelRequestRejected, ModelUnavailable
from taskforge.llm.glm_client import EndpointSource, GlmClient, Pool
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.structured import StructuredOutputError
from taskforge.stage10_proposal.model import Environment, render
from taskforge.stage10_proposal.source import ProposalBatch, SlotProposal, propose_all
from taskforge.stage20_triage.preflight import preflight
from taskforge.stage20_triage.verdict import Verdict

from experiments.post_training.capability_driven_envs.catalog import (
    CapabilityIdea,
    example_catalog_path,
)
from experiments.post_training.capability_driven_envs.generate_tasks import (
    FILE_SAFE_ID,
    GLM_HIGH_TOKEN_ENV,
    GLM_PROVIDER,
    SUMMARY_FILE,
    endpoint_source,
    selected_ideas,
)
from experiments.post_training.capability_driven_envs.proposal_source import CapabilitySource, plan_request
from experiments.post_training.glm import DEFAULT_GLM_RELAY_JOB, GLM_BULK_TOKEN_ENV

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProposedCapability:
    """A capability's proposal batch and the verdict of each proposal, by slot, or the error that stopped it.

    When planning failed, ``failure`` holds the error, ``verdicts`` is empty, and ``batch`` holds the
    planning request, the invalid plan replies (none when the planning call itself failed), and no slots.
    """

    capability_id: str
    batch: ProposalBatch
    verdicts: Mapping[int, Verdict]
    failure: str | None


async def propose_capability(source: CapabilitySource, idea: CapabilityIdea, slots: int) -> ProposedCapability:
    """Propose ``slots`` slots for ``idea`` and triage every proposal with ``preflight``.

    A planning call that fails, or a plan still invalid after its repair, becomes the capability's
    ``failure``, with the planning request and the plan replies received. The function returns it in
    place of raising it, so the other capabilities proceed. A slot's model errors are already slot
    failures in the batch.
    """
    try:
        batch = await propose_all(source, idea, slots)
    except StructuredOutputError as error:
        planning = ProposalBatch(plan_request(idea, slots, source.environments), error.completions, ())
        return ProposedCapability(idea.capability_id, planning, {}, str(error))
    except (ModelRequestRejected, ModelUnavailable) as error:
        planning = ProposalBatch(plan_request(idea, slots, source.environments), (), ())
        return ProposedCapability(idea.capability_id, planning, {}, str(error))
    verdicts = {s.slot: await preflight(s.proposal) for s in batch.slots if isinstance(s, SlotProposal)}
    return ProposedCapability(idea.capability_id, batch, verdicts, None)


async def propose_tasks(
    model: ModelClient,
    ideas: Mapping[str, CapabilityIdea],
    slots: int,
    environments: frozenset[Environment],
    policy: LLMPolicy,
) -> list[ProposedCapability]:
    """Propose and triage every capability of ``ideas`` concurrently."""
    source = CapabilitySource(model, policy, environments)

    async def logged(idea: CapabilityIdea) -> ProposedCapability:
        proposed = await propose_capability(source, idea, slots)
        logger.info("%s: %s", idea.capability_id, proposed.failure or f"{len(proposed.verdicts)} proposals triaged")
        return proposed

    return await asyncio.gather(*(logged(idea) for idea in ideas.values()))


def capability_summary(proposed: ProposedCapability) -> str | dict[str, str]:
    """Summarize ``proposed`` as its failure, or as each slot's decision or slot failure."""
    if proposed.failure is not None:
        return proposed.failure
    return {
        str(outcome.slot): (
            str(proposed.verdicts[outcome.slot].decision) if isinstance(outcome, SlotProposal) else outcome.error
        )
        for outcome in proposed.batch.slots
    }


def write_outputs(output_dir: Path, proposed: Sequence[ProposedCapability]) -> None:
    """Write every capability's files and ``summary.json`` into ``output_dir``."""
    for capability in proposed:
        directory = output_dir / capability.capability_id
        directory.mkdir()
        batch = capability.batch
        (directory / "plan.request.json").write_text(json.dumps(list(batch.planning_request), indent=2))
        (directory / "plan.completions.json").write_text(pretty_json(batch.planning))
        for outcome in batch.slots:
            name = f"slot-{outcome.slot:02d}"
            (directory / f"{name}.request.json").write_text(json.dumps(list(outcome.request), indent=2))
            (directory / f"{name}.completions.json").write_text(pretty_json(outcome.completions))
            if isinstance(outcome, SlotProposal):
                (directory / f"{name}.md").write_text(render(outcome.proposal))
                (directory / f"{name}.verdict.json").write_text(capability.verdicts[outcome.slot].to_json())
            else:
                (directory / f"{name}.failure.txt").write_text(outcome.error)
    summary = {capability.capability_id: capability_summary(capability) for capability in proposed}
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))


async def _propose(
    source: EndpointSource,
    ideas: Mapping[str, CapabilityIdea],
    slots: int,
    environments: frozenset[Environment],
) -> list[ProposedCapability]:
    async with GlmClient(source, GLM_PROVIDER) as client:
        return await propose_tasks(client, ideas, slots, environments, LLMPolicy())


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory for the run's outputs; it must not exist yet.",
)
@click.option(
    "--catalog",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The capability catalog JSON file. Default: the package's data/example_catalog.json.",
)
@click.option("--capability", "capabilities", multiple=True, help="Only this capability id; repeatable.")
@click.option("--slots", type=click.IntRange(min=1), default=4, show_default=True, help="Slots per capability.")
@click.option(
    "--environment",
    "environments",
    type=click.Choice([environment.value for environment in Environment]),
    multiple=True,
    required=True,
    help="A task environment the build host runs; repeatable. The plan proposes slots in these alone.",
)
@click.option("--base-url", help="The GLM router's OpenAI-compatible base URL, ending in /v1.")
@click.option(
    "--relay-job",
    help=f"The Iris GLM relay job to resolve the endpoint from, e.g. {DEFAULT_GLM_RELAY_JOB} (inside Iris only).",
)
@click.option(
    "--pool",
    type=click.Choice([pool.value for pool in Pool]),
    default=Pool.BULK.value,
    show_default=True,
    help=(
        f"The router pool. It selects the token variable ({GLM_BULK_TOKEN_ENV} for bulk, {GLM_HIGH_TOKEN_ENV} "
        "for high) and the pool for the /health check. The token selects the pool serving requests, so this "
        "must match the token's pool."
    ),
)
def main(
    output_dir: Path,
    catalog: Path | None,
    capabilities: tuple[str, ...],
    slots: int,
    environments: tuple[str, ...],
    base_url: str | None,
    relay_job: str | None,
    pool: str,
) -> None:
    """Propose tasks from the capability catalog and triage each proposal with the preflight.

    Give exactly one of --base-url (a port-forwarded router) and --relay-job (inside an Iris task).
    """
    configure_logging(logging.INFO)
    source = endpoint_source(base_url, relay_job, Pool(pool))
    if output_dir.exists():
        raise click.BadParameter(f"{output_dir} already exists", param_hint="'--output-dir'")
    ideas = selected_ideas(catalog or example_catalog_path(), capabilities)
    unsafe = [capability_id for capability_id in ideas if FILE_SAFE_ID.match(capability_id) is None]
    if unsafe:
        raise click.ClickException(f"capability ids unsafe as directory names: {unsafe}")
    output_dir.mkdir(parents=True)
    hosted = frozenset(Environment(environment) for environment in environments)
    proposed = asyncio.run(_propose(source, ideas, slots, hosted))
    write_outputs(output_dir, proposed)
    click.echo(output_dir / SUMMARY_FILE)


if __name__ == "__main__":
    main()

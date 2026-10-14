# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check native harness capture on one unchanged complement task without training."""

from dataclasses import replace

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import skyrl_step

from experiments.post_training.bfcl_rl.collect import COLLECTION_EXECUTION, SMOKE_TASKS, collection_spec


def harness_capture_step(task: str, agent: str, agent_version: str) -> ArtifactStep:
    """Build a generation-only native harness capture check."""
    spec = collection_spec("student", task)
    recipe = yaml.safe_load(spec.config_yaml)
    harbor = recipe["terminal_bench"]["harbor"]
    harbor["name"] = agent
    harbor["version"] = agent_version
    harbor.pop("thinking_format")
    # Probe unsupported native agents before enabling their training capability.
    recipe["trainer"]["algorithm"]["off_policy_correction"] = "none"
    recipe["trainer"]["algorithm"]["tito_full"] = agent in {"mini-swe-agent", "opencode"}
    name = user_owned_name(f"rollouts/bfcl-rl-{agent}-capture-{task}")
    spec = replace(
        spec, name=name, version=resolve_version(name, None), config_yaml=yaml.safe_dump(recipe, sort_keys=False)
    )
    return skyrl_step(spec, COLLECTION_EXECUTION)


@click.command(help=__doc__)
@click.option("--task", type=click.Choice(SMOKE_TASKS), required=True)
@click.option("--agent", type=click.Choice(("mini-swe-agent", "opencode", "claude-code", "codex")), required=True)
@click.option("--agent-version", required=True)
@rl_build_options
def main(task: str, agent: str, agent_version: str) -> ArtifactStep:
    return harness_capture_step(task, agent, agent_version)


if __name__ == "__main__":
    main()

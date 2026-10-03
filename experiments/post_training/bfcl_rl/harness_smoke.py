# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check native Mini-SWE capture on one unchanged complement task without training."""

from dataclasses import replace

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import skyrl_step

from experiments.post_training.bfcl_rl.collect import COLLECTION_EXECUTION, SMOKE_TASKS, collection_spec


def mini_swe_capture_step(task: str, images: tuple[str, str, str]) -> ArtifactStep:
    """Build a generation-only native Mini-SWE capture check."""
    spec = collection_spec("student", task, images)
    recipe = yaml.safe_load(spec.config_yaml)
    harbor = recipe["terminal_bench"]["harbor"]
    harbor["name"] = "mini-swe-agent"
    harbor["version"] = "2.1.0"
    harbor.pop("thinking_format")
    # This operation has no policy loss. Validate exact evidence from its retained
    # result before declaring Mini-SWE's training capability.
    recipe["trainer"]["algorithm"]["off_policy_correction"] = "none"
    recipe["trainer"]["algorithm"]["tito_full"] = False
    name = user_owned_name(f"rollouts/bfcl-rl-mini-swe-capture-{task}")
    spec = replace(
        spec, name=name, version=resolve_version(name, None), config_yaml=yaml.safe_dump(recipe, sort_keys=False)
    )
    return skyrl_step(spec, COLLECTION_EXECUTION)


@click.command(help=__doc__)
@click.option("--task", type=click.Choice(SMOKE_TASKS), required=True)
@click.option("--python-image", required=True)
@click.option("--java-image", required=True)
@click.option("--javascript-image", required=True)
@rl_build_options
def main(task: str, python_image: str, java_image: str, javascript_image: str) -> ArtifactStep:
    return mini_swe_capture_step(task, (python_image, java_image, javascript_image))


if __name__ == "__main__":
    main()

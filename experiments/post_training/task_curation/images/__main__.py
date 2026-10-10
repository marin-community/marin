# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the environments the RL data catalog declares, and record each as an artifact under MARIN_PREFIX.

Run with ``python -m experiments.post_training.task_curation.images``. The artifact types and build steps
live in ``images.build``: an artifact recorded from ``__main__`` would name ``__main__`` as its result
type, which a driver could not load.
"""

import logging

import click
from marin.execution.lazy import run
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.task_curation.datasets.environments import COMPILER_GRADER_PACKAGES, GRADER_PACKAGES
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.images.build import (
    DEFAULT_REPOSITORY,
    environment_artifact,
    identity_digest,
)


def declared_environments() -> dict[str, Environment]:
    """Every environment the catalog declares that the pipeline builds, by identity."""
    declared = (GRADER_PACKAGES, COMPILER_GRADER_PACKAGES)
    return {identity_digest(environment): environment for environment in declared}


@click.command(help=__doc__)
@click.option("--all", "build_all", is_flag=True, help="Build every environment the catalog declares.")
@click.option(
    "--identity",
    "identities",
    multiple=True,
    help="Identity prefix of a declared environment, as in images/env-<identity>; repeat to select several.",
)
@click.option("--repository", default=DEFAULT_REPOSITORY, show_default=True, help="Image repository a build pushes to.")
def main(build_all: bool, identities: tuple[str, ...], repository: str) -> None:
    if build_all == bool(identities):
        raise click.UsageError("Pass --all or at least one --identity")
    declared = declared_environments()
    selected = list(declared.values()) if build_all else []
    for prefix in identities:
        matches = [environment for identity, environment in declared.items() if identity.startswith(prefix)]
        if len(matches) != 1:
            raise click.UsageError(f"{len(matches)} declared environments have identity prefix {prefix}")
        selected.extend(matches)
    logging.basicConfig(level=logging.INFO)
    # A workstation reaches the CoreWeave artifact prefix through its ambient CW_KEY_* pair.
    configure_coreweave_s3()
    for built in run(*(environment_artifact(environment, repository) for environment in selected)):
        click.echo(f"{built.path}: {built.image or built.lock_url}")


if __name__ == "__main__":
    main()

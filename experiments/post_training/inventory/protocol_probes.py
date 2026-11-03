# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the agent protocol probes as a validation source for the quick RL loop.

The probes are the ``protocol-probes`` dataset of ``taskcompendium.examples.protocol_probes``. The
answer file is graded in the worker from the verifyit lock of an environment with no extra packages.
Evaluation reports each check's results as ``eval/protocol-probes/protocol/<check>/<result>`` with an
``_assessed`` denominator.
"""

from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.examples.protocol_probes import PROBE_DATASET, lock_grader_environment, probe_tasks
from taskcompendium.runtime.local import LOCK_FILE
from zephyr.writers import write_parquet_file

from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.images.build import environment_artifact
from experiments.post_training.task_curation.inventory import TASK_SCHEMA, task_records

PROBES_FILENAME = "probes.parquet"
PROBE_ARTIFACT_NAME = f"rl-data/{PROBE_DATASET}"
PROBE_ARTIFACT_VERSION = "1"
"""Bump when the probe tasks change; the grader environment and variant count enter the identity."""
PROBE_VARIANTS = 8
"""Tasks per check; one evaluation of all variants gives each check's rate."""
GRADER_ENVIRONMENT = Environment()
"""verifyit alone; the exact-match grader of the answer file needs nothing else."""


def write_probe_tasks(output_path: str, grader_environment: str, variants: int) -> None:
    """Write the probes as canonical TaskSpec rows, graded from the built ``grader_environment``'s lock."""
    grader = lock_grader_environment(str(StoragePath(grader_environment) / LOCK_FILE))
    tasks = probe_tasks(grader_environment=grader, variants=variants)
    destination = str(StoragePath(output_path) / PROBES_FILENAME)
    write_parquet_file(task_records(iter(tasks)), destination, schema=TASK_SCHEMA)


def probe_tasks_step() -> ArtifactStep[Artifact]:
    """The probe rows under ``PROBE_ARTIFACT_VERSION``, for the current grader environment and variant count."""
    return apply(
        PROBE_ARTIFACT_NAME,
        write_probe_tasks,
        version=PROBE_ARTIFACT_VERSION,
        output_path=OUT,
        grader_environment=environment_artifact(GRADER_ENVIRONMENT),
        variants=PROBE_VARIANTS,
    )

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validated Recursive Task Synthesis packages, preserved as native Harbor tasks."""

from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.native_harbor import native_task, unpack_native
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

DATASET = "laion/rts-1k-rootless-apptainer"
REVISION = "6dbb24eedc0d7ffe5431e1b052b0de8e19eaf6fe"
VERSION = "2026.11.15.1"
RUBRIC = """Compare the public terminal instruction with the original native verifier and solution.
Check whether required behavior is specified and whether the verifier measures it faithfully.
The original environment files and compatibility patches are part of the task contract.
Source validation on aarch64 Apptainer does not establish execution on the current Harbor backend.
"""


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            info=SourceInfo(
                origin=SourceOrigin.RST,
                upstream_dataset=DATASET,
                upstream_name="Recursive Task Synthesis validated RTS-1K",
                build_path="Pinned native Harbor archive intake",
                family="terminal",
                tags=("license:cc-by-4.0", "recursive-task-synthesis"),
                task_format=TaskFormat.HARBOR,
                notes=(
                    "984 source-validated native packages; original archive bytes, modes and mtimes preserved. "
                    "Mutable Dockerfile bases and build downloads remain subject to preparation. "
                    "Pending current-pin execution qualification and admission."
                ),
            ),
            config=CurationRecipe(
                name="rst-validated",
                version=VERSION,
                source=HfSource(DATASET, REVISION, ("tasks.parquet",), SourceFormat.PARQUET, decode=unpack_native),
                convert=native_task,
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                oracle_solutions=True,
            ),
            pipeline=process_rows,
        )
    ]

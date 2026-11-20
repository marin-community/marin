# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The repaired agentic SWE-rebench V2 source, built from the author's V2.1 release."""

from experiments.post_training.task_curation.datasets.swe_rebench_v2.convert import (
    IMAGE_BINDINGS_FILENAME,
    IMAGE_BINDINGS_INPUT,
    VERSION,
    convert_row,
)
from experiments.post_training.task_curation.datasets.swe_rebench_v2.groups import (
    UPSTREAM_DATASET,
    UPSTREAM_REVISION,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

REPAIR_RUBRIC = """
Compare the public problem statement with the hidden test patch, restored trusted test files, and the
FAIL_TO_PASS/PASS_TO_PASS ids, across the row's own test command and log parser. The condensed shared base
image plus the task-start setup must reproduce the row's digest-pinned upstream environment; a missing local
checkout is a runtime limitation, not an underspecified issue. Do not expose private tests or the golden patch.
"""

UPSTREAM_FILES = tuple(f"data/train-{i:05d}-of-00006.parquet" for i in range(6))


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            info=SourceInfo(
                origin=SourceOrigin.SWE_REBENCH,
                upstream_dataset=UPSTREAM_DATASET,
                upstream_name=UPSTREAM_DATASET,
                build_path="SWE-rebench V2.1 Marin repair",
                family="swe-repo",
                tags=(
                    "agentic",
                    "multi-turn",
                    "license:cc-by-4.0",
                    "language:python",
                    "language:go",
                    "language:javascript",
                    "language:typescript",
                    "language:rust",
                    "language:java",
                    "language:php",
                    "language:kotlin",
                    "language:julia",
                    "language:elixir",
                    "language:scala",
                    "language:swift",
                    "language:dart",
                    "language:c",
                    "language:cpp",
                    "language:csharp",
                    "language:r",
                    "language:clojure",
                    "language:ocaml",
                    "language:lua",
                ),
                task_format=TaskFormat.HARBOR,
                notes=(
                    f"Repaired from the original author's re-verified {UPSTREAM_DATASET}@{UPSTREAM_REVISION} "
                    "(license PASS cc-by-4.0); ten consolidated shared base images replace the per-task "
                    "digest-pinned images, the upstream combine.Dockerfile.j2 build step becomes the declared "
                    "task-start setup, and the source's own Harbor verifier grades behind the SWE_REBENCH "
                    "shared-source contract (upstream test.sh/grade.py bytes verbatim; reward channel "
                    "unchanged). Fidelity-gated per group against the upstream digest images before the full "
                    "population run; per-version toolchains and interpreter provisioning are declared base "
                    "properties. Pending qualification."
                ),
            ),
            config=CurationRecipe(
                name="swe-rebench-v2-1-repaired",
                version=VERSION,
                source=HfSource(
                    UPSTREAM_DATASET,
                    UPSTREAM_REVISION,
                    UPSTREAM_FILES,
                    SourceFormat.PARQUET,
                ),
                # The digest-pinned base-image bindings input is wired here once the Kaniko
                # build job stages them; the converter rejects rows with a typed defect until then.
                inputs={},
                convert=convert_row,
                intended_use=IntendedUse.TRAIN,
                rubric=REPAIR_RUBRIC,
                oracle_solutions=True,
            ),
            pipeline=process_rows,
        ),
    ]

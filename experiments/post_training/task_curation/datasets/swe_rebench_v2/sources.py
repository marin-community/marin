# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The repaired agentic SWE-rebench V2 source, built from the author's V2.1 release.

The Atlas repair lane declares this source (atlas declaration
``SWE-rebench:ibragim-bad/SWE-rebench-V2.1@repaired-20261119``, pipeline
``swe-rebench-v2-1-repaired`` at converter version 2026.11.20.2) while its admission
lane finishes; the user ruling of 2026-11-26 treats it as about to be admitted, so the
catalog binds the declaration itself. The Atlas recipe also sets ``oracle_solutions``;
this catalog's pipeline has no oracle-filter stage, so the flag has no consumer here
and is not declared.
"""

from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from finestore.rl.status_taxonomy import DataRestriction

from experiments.post_training.task_curation.datasets.swe_rebench_v2.convert import (
    IMAGE_BINDINGS_INPUT,
    VERSION,
    convert_row,
)
from experiments.post_training.task_curation.datasets.swe_rebench_v2.groups import (
    UPSTREAM_DATASET,
    UPSTREAM_REVISION,
)
from experiments.post_training.task_curation.datasets.swe_repair_license_receipts import (
    SWE_REBENCH_V2_1_RECEIPT,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

REPAIR_RUBRIC = """
Compare the public problem statement with the hidden test patch, restored trusted test files, and the
FAIL_TO_PASS/PASS_TO_PASS ids, across the row's own test command and log parser. The condensed shared base
image plus the task-start setup must reproduce the row's digest-pinned upstream environment; a missing local
checkout is a runtime limitation, not an underspecified issue. Do not expose private tests or the golden patch.
"""

UPSTREAM_FILES = tuple(f"data/train-{i:05d}-of-00006.parquet" for i in range(6))


def sources() -> list[RlDataSource[CurationRecipe]]:
    """The declared repaired source at its exact Atlas pin."""
    return [
        RlDataSource(
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=SourceOrigin.SWE_REBENCH,
                upstream_dataset=UPSTREAM_DATASET,
                upstream_name=UPSTREAM_DATASET,
                variant="repaired-20261119",
                build_path="SWE-rebench V2.1 Marin repair",
                family="swe-repo",
                tags=(
                    "agentic",
                    "single-turn",
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
                license_receipt=SWE_REBENCH_V2_1_RECEIPT,
                notes=(
                    f"Repaired from the original author's re-verified {UPSTREAM_DATASET}@{UPSTREAM_REVISION} "
                    "(license PASS cc-by-4.0); ten consolidated shared base images replace the per-task "
                    "digest-pinned images, the upstream combine.Dockerfile.j2 build step becomes the declared "
                    "task-start setup, and the source's own Harbor verifier grades behind the SWE_REBENCH "
                    "shared-source contract (upstream test.sh/grade.py bytes verbatim; reward channel "
                    "unchanged). Fidelity-gated per group against the upstream digest images before the full "
                    "population run; per-version toolchains and interpreter provisioning are declared base "
                    "properties. Declared limits deviate from the upstream source declaration "
                    "(6 CPUs / 24,576 MB / 32,768 MB storage, recorded verbatim in convert.py): "
                    "the emitted execution limits are Daytona's per-sandbox cap (4 CPUs / 8,192 MB "
                    "/ 10,240 MB, the setup deadline 3600 s), reduced by the approved user decision "
                    "2026-11-25 (reason: the upstream spec exceeds the Daytona 4/8/10 per-sandbox "
                    "cap; evidence: the fidelity gate ran the full population at 4 CPUs, where "
                    "every admitted row passed its NoP 0.0 + oracle 1.0 trials). "
                    "Pending qualification."
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
                # The staged conversion inputs: the digest-pinned base-image bindings and the
                # admission receipts (the upstream_image_defect labels), pinned at an exact
                # HF dataset revision (the pipeline stages HfSource files; the census tools
                # expect repo/revision sources).
                inputs={
                    IMAGE_BINDINGS_INPUT: HfSource(
                        repo="open-athena/marin-rl-data-swe-repair-inputs",
                        revision="07ab2022c3f8de5d8ca50b7c03119d48bc0b3a84",
                        files=("base-image-bindings.json", "admission-receipts.json"),
                        format=SourceFormat.JSON,
                    ),
                },
                convert=convert_row,
                intended_use=IntendedUse.TRAIN,
                rubric=REPAIR_RUBRIC,
            ),
            pipeline=process_rows,
        ),
    ]

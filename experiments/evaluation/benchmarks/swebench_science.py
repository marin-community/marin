# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-bench Science with native isolated grading.

Release tooling and metadata are MIT licensed. Task sources retain their upstream licenses;
the default selection follows the release's unrestricted manifest.
"""

from functools import partial

from taskcompendium.evaluation.spec import MiniSweAgent, PiAgent
from taskcompendium.importers.harbor.package import JobSettings
from taskcompendium.importers.harbor.science import lower_science_package
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, RawRow

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.harbor import (
    MAX_OUTPUT_TOKENS,
    PI_MAX_OUTPUT_TOKENS,
    HarborDataset,
    convert_package_row,
    harbor_evaluation_source,
    pi_install,
)
from experiments.evaluation.preparation import EvaluationCorpusRecipe
from experiments.post_training.task_curation.harbor_packages import package_rows
from experiments.post_training.task_curation.pipeline import HarborRepoSource
from experiments.post_training.task_curation.source import RlDataSource

DATASET = "OpenMOSS-Team/SWE-bench-Science"
REVISION = "997408298e8e4d09978b1120b95c4540d1a71bcb"
VERSION = "2026.11.24.2"
BENCHMARK = "swebench-science"
# Exact selections/default-96.json in the pinned release.
TASK_IDS = tuple(
    """
    001 002 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 022 024 025
    027 028 029 030 031 033 034 036 037 038 039 040 041 042 043 044 045 046 047 048
    049 050 051 052 053 054 055 056 058 059 060 061 062 063 064 065 067 068 069 070
    071 072 073 076 077 078 079 080 081 086 087 088 089 090 091 092 093 094 095 099
    103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 119
    """.split()
)

# The hosted selection adds these reviewed packages to the upstream default.
HOSTED_TASK_IDS = (*TASK_IDS, *"003 020 021 023 032 057 066 074 075 082 083 084 085 096 097 098 100 118".split())
PI = PiAgent(version="0.87.0", thinking="xhigh", setup_timeout=360)


def convert(row: RawRow, _context: ConversionContext, *, job: JobSettings) -> TaskSpec | ImportRejection:
    return convert_package_row(row, partial(lower_science_package, task_id=row.id, source=row.source, job=job))


def science_source(
    benchmark: str, task_ids: tuple[str, ...], harness: PiAgent | MiniSweAgent
) -> RlDataSource[EvaluationCorpusRecipe]:
    pi = isinstance(harness, PiAgent)
    protocol_name = "pi-xhigh" if pi else "mini-swe-agent"
    job = JobSettings(storage_mb=10240, agent_install=pi_install(harness) if isinstance(harness, PiAgent) else None)
    dataset = HarborDataset(
        recipe=f"{benchmark}-{protocol_name}",
        benchmark=benchmark,
        title=f"SWE-bench Science ({len(task_ids)} tasks)",
        source=HarborRepoSource(
            DATASET, REVISION, read=package_rows, root="tasks", packages=frozenset(f"task_{key}" for key in task_ids)
        ),
        version=VERSION,
        count=len(task_ids),
        attempt_timeout=10800,
    )
    return harbor_evaluation_source(
        dataset,
        harness,
        partial(convert, job=job),
        max_input_tokens=65536 if pi else None,
        max_output_tokens=PI_MAX_OUTPUT_TOKENS if pi else MAX_OUTPUT_TOKENS,
        trials=1,
        protocol=protocol_name,
    )


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        science_source(BENCHMARK, TASK_IDS, PI),
        science_source(BENCHMARK, TASK_IDS, MiniSweAgent()),
        science_source(f"{BENCHMARK}-hosted114", HOSTED_TASK_IDS, PI),
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources())

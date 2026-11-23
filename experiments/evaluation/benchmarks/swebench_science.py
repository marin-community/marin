# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-bench Science's default-96 release, with native isolated grading and mini-swe-agent.

Release tooling and metadata are MIT licensed. Task sources retain their upstream licenses;
the selection excludes every task gated by the release's restricted-license policy.
"""

from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from taskcompendium.evaluation.spec import AgentEpisode, EpisodeLimits, MiniSweAgent
from taskcompendium.importers.harbor.science import lower_science_package
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import IntendedUse, RawRow

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.harbor import EPISODE_LABEL, harbor_item, harbor_protocol
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.harbor_packages import extract_package, package_rows
from experiments.post_training.task_curation.pipeline import CurationRecipe, HarborRepoSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, TaskFormat

DATASET = "OpenMOSS-Team/SWE-bench-Science"
REVISION = "997408298e8e4d09978b1120b95c4540d1a71bcb"
VERSION = "2026.11.20.1"
BENCHMARK = "swebench-science"
# Exact selections/default-96.json in the pinned release. The other 23 tasks
# require upstream's explicit restricted-license opt-in and are not selected.
TASK_IDS = tuple(
    """
    001 002 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 022 024 025
    027 028 029 030 031 033 034 036 037 038 039 040 041 042 043 044 045 046 047 048
    049 050 051 052 053 054 055 056 058 059 060 061 062 063 064 065 067 068 069 070
    071 072 073 076 077 078 079 080 081 086 087 088 089 090 091 092 093 094 095 099
    103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 119
    """.split()
)


def convert(row: RawRow, _context: ConversionContext) -> TaskSpec:
    with TemporaryDirectory(prefix="swe-science-row-") as temporary:
        directory = Path(temporary) / row.data["name"]
        extract_package(row.data["archive"], directory)
        lowered = lower_science_package(directory, task_id=row.id, source=row.source)
    return lowered.task.model_copy(
        update={"labels": {**lowered.task.labels, EPISODE_LABEL: lowered.episode.model_dump_json()}, "tags": ("eval",)}
    )


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    agent = AgentEpisode(
        harness=MiniSweAgent(),
        limits=EpisodeLimits(
            max_turns=None,
            max_total_tokens=None,
            model_turn_timeout=600,
            total_turn_timeout=None,
            verifier_timeout=None,
            attempt_timeout=10800,
        ),
        max_input_tokens=None,
    )
    recipe = CurationRecipe(
        name=BENCHMARK,
        source=HarborRepoSource(
            DATASET, REVISION, read=package_rows, root="tasks", packages=frozenset(f"task_{key}" for key in TASK_IDS)
        ),
        convert=convert,
        version=VERSION,
        intended_use=IntendedUse.EVAL,
    )
    protocol = harbor_protocol(BENCHMARK, VERSION, len(TASK_IDS), max_output_tokens=8192, requires_judge=False, trials=1)
    return [
        RlDataSource(
            info=SourceInfo(
                id="Harbor:swebench-science",
                title="SWE-bench Science, unrestricted 96",
                origin="Harbor",
                family="agentic",
                tags=("eval", "science", "agentic", "multi-turn", "benchmark"),
                count=len(TASK_IDS),
                task_format=TaskFormat.HARBOR,
            ),
            config=EvaluationCorpusRecipe(
                recipe, protocol, partial(harbor_item, agent=agent), protocol_name="mini-swe-agent", agent=agent
            ),
            pipeline=prepare_evaluation,
        )
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources())

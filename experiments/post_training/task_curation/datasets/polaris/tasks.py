# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""POLARIS-53K training intake with its original deterministic DeepScaleR reward."""

from pathlib import Path

from finestore.rl.curation import TaskFormat
from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.script_grader import grade_script, script_package, shipped_files
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, RawRow
from verifyit.spec import DEFAULT_OUTPUT

from experiments.post_training.task_curation.datasets.environments import POLARIS_GRADER_PACKAGES
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin

HERE = Path(__file__).parent
REVISION = "296f8e34132e63f4a1d70e0dcc8bddebb43f03e4"
SCORER_REVISION = "eda12fe22dd83aca59ec46a67147240dc483f62b"
SCORER = grade_script(HERE / "grade.py", *shipped_files(HERE / "scorers", "polaris_math.py", "LICENSE"))
RUBRIC = """
Assess the complete original problem and hidden answer independently. Preserve proof requests,
domains, LaTeX and the released difficulty estimate. The estimate is model performance, not truth.
The native deterministic DeepScaleR reward requires </think> and extracts the boxed answer from
the segment immediately after its first occurrence. A bare correct box receives zero by design;
this format restriction is a property of the source. The LLM fallback is disabled as upstream.
Check missing premises, malformed references and contradictions without repairing the source.
The released answer is a hidden grading reference; no executable passing oracle is supplied.
"""


def convert(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    problem, answer = row.data.get("problem"), row.data.get("answer")
    if not isinstance(problem, str) or not problem.strip() or not isinstance(answer, str) or not answer.strip():
        return source_defect("missing_prompt_or_reference", "problem and answer must be nonempty strings")
    package = script_package(
        SCORER,
        {"problem": problem, "answer": answer},
        environment=required_grader_environment(context),
        timeout=60.0,
        answer_path=DEFAULT_OUTPUT,
    )
    return conversation_task(
        row,
        events=(TextMessage(role="user", content=problem),),
        package=package,
        evidence={"difficulty": row.data.get("difficulty"), "scorer_revision": SCORER_REVISION},
    )


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=SourceOrigin.POLARIS_PROJECT,
                upstream_dataset="Polaris-Dataset-53K",
                upstream_name="POLARIS-53K",
                build_path="POLARIS training intake",
                family="math-answer",
                tags=("rlvr", "single-turn", "license:apache-2.0"),
                task_format=TaskFormat.TASKSPEC,
                notes=f"Original deterministic DeepScaleR reward {SCORER_REVISION}; LLM fallback disabled.",
            ),
            config=CurationRecipe(
                name="polaris-53k",
                source=HfSource(
                    "POLARIS-Project/Polaris-Dataset-53K", REVISION, ("polaris-data-53K.jsonl",), SourceFormat.JSONL
                ),
                convert=convert,
                version="2026.11.16.1",
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                controls=Controls(),
                grader=POLARIS_GRADER_PACKAGES,
            ),
        )
    ]

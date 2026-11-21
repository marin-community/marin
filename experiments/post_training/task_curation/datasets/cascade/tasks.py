# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cascade-2 TRAIN subsets, preserving supported original NeMo grading contracts."""

from pathlib import Path

from finestore.rl.curation import TaskFormat
from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import unsupported
from taskcompendium.convert.nemotron_ultra import blend_task, text_request
from taskcompendium.convert.script_grader import grade_script, script_package, shipped_files
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, NormalizedTask, RawRow
from verifyit.spec import DEFAULT_OUTPUT

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin

HERE = Path(__file__).parent
REVISION = "05bbaf03bac608e804efc86c5f8bf99844f2d197"
SCORER_REVISION = "39d2bde61e426f93e4bc00c0255fc1d839f859e6"
SUPPORTED_AGENTS = ("mcqa_simple_agent", "structured_outputs_simple_agent", "math_with_judge_simple_agent")
SCORER = grade_script(
    HERE / "grade.py",
    *shipped_files(HERE / "scorers", "mcqa.py", "structured_outputs.py", "math_with_judge.py", "LICENSE"),
)
RUBRIC = """
Assess the full public conversation against the original hidden scoring contract. Check missing inputs,
conflicting choices, indefensible answer keys, and contradictory structural requirements. A source
reference is grading data, not proof of a passing oracle. MCQA retains the original boxed/regex parsing;
structured outputs retain the original strict schema transformation and format parsers. Math retains
the primary server configuration's boxed-answer requirement and disabled LLM fallback. Flag defects
in that contract separately from candidate quality. Live workplace environments and empty SWE requests
must not be converted into terminal-answer tasks.
"""


def convert(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    agent = row.data["agent_ref"]["name"]
    if agent == "instruction_following_simple_agent":
        return unsupported("grader_package_license_f2", "Pinned source verifier package has no package-level license")
    if agent == "swe_agents_train":
        return unsupported(
            "swe_workspace_reconstruction_required",
            "Released input is empty; metadata requires the original SWE workspace reconstruction runtime",
        )
    if agent == "workplace_assistant_simple_agent":
        return unsupported(
            "workplace_live_provider_required", "The original task requires its 27 live workplace tools and state"
        )
    request = text_request(row.data, SUPPORTED_AGENTS)
    if isinstance(request, ImportRejection):
        return request
    package = script_package(
        SCORER,
        request.contract,
        environment=required_grader_environment(context),
        timeout=60.0,
        answer_path=DEFAULT_OUTPUT,
    )
    return blend_task(row, request, package)


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=SourceOrigin.NEMOTRON,
                upstream_dataset="Nemotron-Cascade-2-RL-data",
                upstream_name="Nemotron Cascade-2 RL data",
                subset=subset,
                build_path="Cascade-2 training intake",
                family="mixed",
                tags=("license:odc-by",),
                task_format=TaskFormat.TASKSPEC,
                notes=(
                    f"Original NeMo terminal graders {SCORER_REVISION}; "
                    "unsupported components retain typed rejections."
                ),
            ),
            config=CurationRecipe(
                name="nemotron-cascade-2-" + subset.lower(),
                source=HfSource(
                    "nvidia/Nemotron-Cascade-2-RL-data", REVISION, (f"{subset}/train.jsonl",), SourceFormat.JSONL
                ),
                convert=convert,
                version="2026.11.15.1",
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                controls=Controls(),
                grader=GRADER_PACKAGES,
            ),
        )
        for subset in ("IF-RL", "multi-domain-RL", "MOPD", "SWE-RL")
    ]

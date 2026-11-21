# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The released IFBench training split with its original IFEvalG verifier."""

import ast
import json
from pathlib import Path

from finestore.rl.status_taxonomy import DataRestriction
from pydantic import ValidationError
from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.script_grader import grade_script, script_package, shipped_files
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, RawRow
from verifyit.spec import DEFAULT_OUTPUT

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

HERE = Path(__file__).parent
SCORER_REVISION = "28c1816664491493242d453c2c7280ee56e6c20a"
SCORER = grade_script(
    HERE / "grade.py",
    *shipped_files(
        HERE / "scorers",
        "open_instruct/IFEvalG/instructions.py",
        "open_instruct/IFEvalG/instructions_registry.py",
        "open_instruct/IFEvalG/instructions_util.py",
        "LICENSE",
    ),
)
RUBRIC = """
Compare every public requirement with the hidden original IFEvalG instruction IDs and arguments.
Flag contradictory constraints, impossible word/letter bounds, absent requested inputs, or a mismatch
between the public wording and scorer. Formal constraint scores alone do not certify useful content.
The original verifier averages constraint rewards. This is the explicit training split; IFBench test
questions must not be included. No passing response is released as an executable oracle.
"""


def constraint_bundle(value: str) -> dict:
    """Decode the release's Python-literal singleton wrapper without evaluating code."""
    bundles = ast.literal_eval(value)
    if not isinstance(bundles, list) or len(bundles) != 1:
        raise ValueError("Expected one IFBench constraint bundle")
    bundle = bundles[0]
    if isinstance(bundle, str):
        bundle = json.loads(bundle)
    if not isinstance(bundle, dict):
        raise ValueError("Expected a constraint dictionary")
    names, arguments = bundle.get("instruction_id"), bundle.get("kwargs")
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or not isinstance(arguments, list)
        or len(names) != len(arguments)
        or not all(argument is None or isinstance(argument, dict) for argument in arguments)
    ):
        raise ValueError("Constraint names and arguments must be nonempty aligned arrays")
    return bundle


def convert(row: RawRow, context: ConversionContext) -> TaskSpec | ImportRejection:
    try:
        bundle = constraint_bundle(row.data["ground_truth"])
        events = tuple(TextMessage.model_validate(message) for message in row.data["messages"])
        ConversationInput(events=events)
    except (ValueError, SyntaxError, KeyError, TypeError, ValidationError) as error:
        return source_defect("invalid_ifbench_contract", str(error))
    package = script_package(
        SCORER,
        bundle,
        environment=required_grader_environment(context),
        timeout=60.0,
        answer_path=DEFAULT_OUTPUT,
    )
    evidence = {key: value for key, value in row.data.items() if key not in {"messages", "ground_truth", "path"}}
    evidence["scorer_revision"] = SCORER_REVISION
    return conversation_task(row, events=events, package=package, evidence=evidence)


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            pipeline=process_rows,
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=SourceOrigin.ALLENAI,
                upstream_dataset="IF_multi_constraints_upto5",
                upstream_name="IFBench training split",
                build_path="IFBench training intake",
                family="instruction-following",
                tags=("license:odc-by",),
                task_format=TaskFormat.TASKSPEC,
                notes=f"Original IFEvalG scorer {SCORER_REVISION}; no released oracle; pending final-pin QA.",
            ),
            config=CurationRecipe(
                name="ifbench-train",
                source=HfSource(
                    "allenai/IF_multi_constraints_upto5",
                    "2e3a77407b7fce69f95b248d64a884e3ae1c2423",
                    ("data/train-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                convert=convert,
                version="2026.11.15.1",
                intended_use=IntendedUse.TRAIN,
                rubric=RUBRIC,
                controls=Controls(),
                grader=GRADER_PACKAGES,
            ),
        )
    ]

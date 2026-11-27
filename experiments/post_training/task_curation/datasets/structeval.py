# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert StructEval's non-renderable structured-output tasks with the official path validator.

The coordinator ruled this source eval-use with only the non-renderable subset converted: rows
whose native ``rendering`` flag is False (JSON/CSV/TOML/XML/YAML) carry path-based ``raw_output_metric``
validation, and the renderable rows (VQA/VLM-scored) are the typed grader-missing exclusion. The
grader is the official TIGER-AI-Lab/StructEval non-renderable evaluator, shipped as a script with
the official extraction, parse, path grammar and ``0.2 * parse_ok + 0.8 * path_fraction`` score.
"""

from pathlib import Path

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.convert.answers import source_defect
from taskcompendium.convert.script_grader import grade_script, script_package
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.models import TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, NormalizationChange, NormalizedTask, RawRow
from verifyit.spec import DEFAULT_OUTPUT

from experiments.post_training.task_curation.datasets.structeval_paths import tokenize_path
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

HERE = Path(__file__).parent
REPO = "TIGER-Lab/StructEval"
REVISION = "936be10548688c6d417cd435dd906158f19ba14b"
SOURCE_ID = "Tiger-lab:StructEval"
VERSION = "structeval-path-validation-v1"
COUNT = 2035
CONVERTIBLE = 950
GRADER_TIMEOUT = 330.0
COMPLETION_INSTRUCTION = (
    "\n\nIMPORTANT: Only output the required output format. You must start the format/code with "
    "<|BEGIN_CODE|> and end the format/code with <|END_CODE|>. No other text output (explanation, "
    "comments, etc.) are allowed.  Do not use markdown code fences."
)
SUPPORTED_FORMATS = frozenset({"json", "yaml", "csv", "toml", "xml"})
GRADER_REQUIREMENTS = ("pyyaml==6.0.2", "xmltodict==0.14.2", "toml==0.10.2")
STRUCTEVAL_GRADE = grade_script(HERE / "structeval_grade.py")
REQUIRED_FIELDS = frozenset(
    {
        "task_id",
        "query",
        "feature_requirements",
        "task_name",
        "input_type",
        "output_type",
        "query_example",
        "VQA",
        "raw_output_metric",
        "rendering",
    }
)


def convert_problem(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    native = dict(row.data)
    if not REQUIRED_FIELDS <= set(native):
        return source_defect("incomplete_structeval_contract", "The StructEval row fields are required")
    task_id = native["task_id"]
    query = native["query"]
    metrics = native["raw_output_metric"]
    if not isinstance(task_id, str) or not task_id.strip():
        return source_defect("invalid_structeval_metadata", "The task_id must be a nonempty string")
    if not isinstance(query, str) or not query.strip():
        return source_defect("incomplete_structeval_contract", "The query must be a nonempty string")
    if native["rendering"] is not False:
        # The coordinator's typed grader-missing exclusion: renderable rows are VQA/VLM-scored.
        return source_defect(
            "grader_missing_renderable_structeval",
            "Renderable StructEval rows need VLM-based VQA scoring; no programmatic grader exists",
        )
    if not isinstance(native["output_type"], str) or native["output_type"].lower() not in SUPPORTED_FORMATS:
        return source_defect("unsupported_structeval_output", "The output type has no shipped parser")
    if (
        not isinstance(metrics, list)
        or not metrics
        or not all(isinstance(metric, str) and metric.strip() for metric in metrics)
    ):
        return source_defect("invalid_metric_paths", "raw_output_metric must be a nonempty list of path strings")
    for metric in metrics:
        try:
            tokenize_path(metric)
        except ValueError as error:
            return source_defect("invalid_metric_path", f"The metric path fails the official tokenizer: {error}")
    config = {"task_id": task_id, "output_type": native["output_type"].lower(), "raw_output_metric": list(metrics)}
    package = script_package(
        STRUCTEVAL_GRADE,
        config,
        environment=required_grader_environment(context),
        timeout=GRADER_TIMEOUT,
        answer_path=DEFAULT_OUTPUT,
    )
    events = (TextMessage(role="user", content=query + COMPLETION_INSTRUCTION),)
    task = conversation_task(row, events=events, package=package).model_copy(
        update={
            "tags": ("eval", "structured-output", "path-validation", "structeval"),
            "labels": {
                "source_id": SOURCE_ID,
                "native_task_id": task_id,
                "task_name": native["task_name"],
                "input_type": native["input_type"],
                "output_type": native["output_type"],
            },
        }
    )
    change = NormalizationChange(
        field="context.events",
        reason="Append the official StructEval completion-format instruction to the verbatim query.",
        original="The native query fixes the target format but not the reply envelope.",
        replacement=(
            "The task appends the benchmark's own <|BEGIN_CODE|>/<|END_CODE|> envelope instruction; "
            "the grader's official extractor still accepts fences or a bare payload."
        ),
    )
    return NormalizedTask(task, (change,))


def source(files: tuple[str, ...]) -> RlDataSource:
    return RlDataSource(
        info=SourceInfo(
            origin=SourceOrigin.TIGER_LAB,
            upstream_dataset="StructEval",
            upstream_name="StructEval",
            build_path=VERSION,
            family="structured-output",
            data_restriction=DataRestriction.OPEN,
            task_format=TaskFormat.TASKSPEC,
            count=COUNT,
            notes=(
                "Eval-use per the coordinator ruling: only the non-renderable subset converts. The 950 "
                "rendering=False rows (JSON 250, CSV 200, TOML 50, XML 200, YAML 250) carry path-based "
                "raw_output_metric validation; the 1,085 renderable rows (VQA/VLM-scored across 14 types) "
                "are the typed grader-missing exclusion. The context is the verbatim native query plus the "
                "benchmark's own completion-envelope instruction, recorded as a normalization. The grader "
                "is the official TIGER-AI-Lab/StructEval non-renderable evaluator shipped as a script: the "
                "official extraction priority, per-format parse, path grammar (csv:: header tokens, "
                "backtick segments, [N] indices, * wildcards on lists only, @attr fallback) and the "
                "official 0.2 parse + 0.8 path-fraction score. Contract defects fail closed."
            ),
        ),
        config=CurationRecipe(
            name="structeval-path-validation",
            version=VERSION,
            source=HfSource(REPO, REVISION, files, SourceFormat.PARQUET),
            grader=Environment(requirements=GRADER_REQUIREMENTS),
            convert=convert_problem,
            intended_use=IntendedUse.EVAL,
        ),
        pipeline=process_rows,
    )

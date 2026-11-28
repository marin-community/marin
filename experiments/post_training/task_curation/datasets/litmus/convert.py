# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the source conversation and its private NeMo verification contract."""

import json
from pathlib import Path

from finestore.rl.curation import TaskFormat
from taskcompendium.convert.nemotron_ultra import blend_task, text_request
from taskcompendium.convert.script_grader import grade_script, script_package, shipped_files
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import Controls, ImportRejection, NormalizedTask, RawRow, Reply
from taskcompendium.runtime.resources import resource_bytes
from verifyit.spec import DEFAULT_OUTPUT

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.ingest_stages.converter_binding import RowConverterBinding

HERE = Path(__file__).parent
SCORER = grade_script(HERE / "grade.py", *shipped_files(HERE / "scorers", "litmus_score.py", "LICENSE"))


def convert(row: RawRow, context: ConversionContext) -> NormalizedTask | ImportRejection:
    request = text_request(row.data, ("litmus_agent_agent",))
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


def golden(task: TaskSpec) -> Reply:
    """Render the private source answer in each of its published output wrappers."""
    resource = next(item for item in task.resources.verifier if item.path == "config.json")
    value = str(json.loads(resource_bytes(resource))["expected_answer"])
    wrappers = (
        "(({value}))",
        "(Answer: {value})",
        "Final answer: ({value})",
        "Answer is [{value}]",
        "[Answer: {value}]",
        "[[{value}]]",
        "Correct Answer: [{value}]",
        "\\boxed{{{value}}}",
        "{{{{{value}}}}}",
        "Answer Value: {{{value}}}",
        "<<{value}>>",
        "<{value}>",
        "<Answer: {value}>",
        "<final_answer>{value}</final_answer>",
        "Final Answer: ||{value}||",
        "The answer is: |{value}|",
        "**Answer: {value}**",
        "**Final answer is: {value}**",
        "Answer: *{value}*",
        "## Answer: {value} ##",
        "ANSWER IS {value}",
        "Response: {value}",
        "Final Answer -> {value}",
        "Final value is: {value}",
        "Correct Answer >> {value}",
        "Answer Value: {value}",
        "Final Answer = {value}",
        "Correct answer is {value}",
        "Final Answer: {value}",
    )
    return answer_reply(task, "\n".join(template.format(value=value) for template in wrappers))


INGEST_CONVERTER = RowConverterBinding(
    convert=convert,
    source_format=SourceFormat.JSONL,
    task_format=TaskFormat.MARINSKYRL,
    grader=GRADER_PACKAGES,
    controls=Controls(golden=golden),
)

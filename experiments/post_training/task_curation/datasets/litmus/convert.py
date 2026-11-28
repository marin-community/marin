# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the source conversation and its private NeMo verification contract."""

from pathlib import Path

from finestore.rl.curation import TaskFormat
from taskcompendium.convert.nemotron_ultra import blend_task, text_request
from taskcompendium.convert.script_grader import grade_script, script_package, shipped_files
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask, RawRow
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


INGEST_CONVERTER = RowConverterBinding(
    convert=convert, source_format=SourceFormat.JSONL, task_format=TaskFormat.MARINSKYRL, grader=GRADER_PACKAGES
)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FinanceBench source prompt and answer conversion."""

from taskcompendium.convert.answers import answer_task
from taskcompendium.importers.evalchemy.judge_prompts import financebench_spec
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow

FINANCE_PROMPT = (
    "Based on the following financial document excerpt, answer the question.\n\nDocument:\n{evidence_text}"
    "\n\nQuestion: {question}\n\nProvide a direct, concise answer."
)


def convert_finance(row: RawRow, _context: ConversionContext) -> TaskSpec:
    task = answer_task(
        row,
        prompt=FINANCE_PROMPT.format_map(row.data),
        spec=financebench_spec(row.data["question"], str(row.data["answer"])),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", "benchmark:FinanceBench")})

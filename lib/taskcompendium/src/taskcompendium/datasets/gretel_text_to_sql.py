# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Text-to-SQL normalization and its private evaluator contract."""

import re
import sqlite3

from taskcompendium.datasets.direct_contracts import contract_task
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.models import (
    ImportFailureKind,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPolicy,
)

RUBRIC = ReviewRubric(
    id="gretel_text_to_sql-quality",
    version="1",
    criteria=(
        "Evaluate the requested ordering, join keys and grouping against the public database fixtures. "
        "Reference SQL can be defective; do not infer a dialect or demand textual equality.",
        "Missing runtime binding is a readiness limitation, not a task quality defect. Identify missing "
        "public context separately from implementation difficulty.",
    ),
)


def validate_seeded_reference(context: str, reference: str) -> None:
    """Reject definite source defects before binding the original SQL scorer."""
    if re.match(r"\s*(?:select|with)\b", reference, re.IGNORECASE) is None or ";" in reference.strip().rstrip(";"):
        raise ValueError("Gretel reference must be one SELECT")
    if re.search(r"\b(?:random|randomblob|current_date|current_time|current_timestamp)\b", reference, re.IGNORECASE):
        raise ValueError("Gretel reference is nondeterministic")
    if not re.search(r"\bcreate\s+(?:temp(?:orary)?\s+)?table\b", context, re.IGNORECASE):
        raise ValueError("Gretel context has no CREATE TABLE")
    if not re.search(r"\binsert\s+(?:into|or)\b", context, re.IGNORECASE):
        raise ValueError("Gretel context has no INSERT")
    with sqlite3.connect(":memory:") as database:
        try:
            database.executescript(context)
            database.execute(reference).fetchone()
        except sqlite3.Error as error:
            raise ValueError("Gretel seeded reference cannot execute") from error


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    question, context, reference = (row.data.get(key) for key in ("sql_prompt", "sql_context", "sql"))
    if not all(isinstance(value, str) and value.strip() for value in (question, context, reference)):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_prompt_or_reference",
            detail="SQL prompt, context and reference are required",
        )
    contract = {
        key: row.data[key] for key in ("sql", "sql_context", "sql_explanation", "sql_complexity", "sql_task_type", "id")
    }
    prompt = f"{question}\n\nDatabase context:\n{context}"
    return contract_task(
        row,
        (TextMessage(role="user", content=prompt),),
        "gretel_text_to_sql",
        contract,
        ("source SQL dialect and execution or semantic comparison policy",),
    )


def policy() -> TaskPolicy:
    return TaskPolicy(normalize=normalize, rubric=RUBRIC)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned gretel_text_to_sql source and its private evaluator contract."""

from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.datasets.direct_contracts import contract_task
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
)

DATASET = "gretelai/synthetic_text_to_sql"
REVISION = "740ab236e64503fba51be1101df7a1be83bf455d"
CONFIG = "default"
SPLIT = "train"
SOURCE_FILE = "synthetic_text_to_sql_train.snappy.parquet"
SOURCE_FORMAT = "parquet"

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


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    question, context, reference = (row.data.get(key) for key in ("sql_prompt", "sql_context", "sql"))
    if not all(isinstance(value, str) and value.strip() for value in (question, context, reference)):
        return ImportRejection(
            reason="missing_prompt_or_reference", detail="SQL prompt, context and reference are required"
        )
    contract = {key: row.data[key] for key in ("sql", "sql_explanation", "sql_complexity", "sql_task_type", "id")}
    prompt = f"{question}\n\nDatabase context:\n{context}"
    return contract_task(
        row,
        (TextMessage(role="user", content=prompt),),
        "gretel_text_to_sql",
        contract,
        ("source SQL dialect and execution or semantic comparison policy",),
    )


def recipe() -> DatasetRecipe:
    return DatasetRecipe(
        name="gretel_text_to_sql",
        version="gretel_text_to_sql-v1",
        source=HFSource(DATASET, REVISION, CONFIG, SPLIT),
        normalize=normalize,
        intended_use=IntendedUse.TRAIN,
        rubric=RUBRIC,
    )

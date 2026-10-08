# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a SQL reply with the image-installed SkyRL text-to-SQL comparator on the seeded database.

Usage: sql_grade.py SKYRL_GYM_ROOT CONFIG ANSWER SCORE. The config holds ``context_sql`` (CREATE TABLE
and INSERT statements) and ``reference_sql`` (one deterministic SELECT). A reference that does not
run or does not match itself fails the grader rather than scoring the reply.
"""

import importlib
import json
import sys
from pathlib import Path

SQL_MODULE = "skyrl_gym.envs.text_to_sql.scoring"


# Each grade script ships alone to /tests in the grader image, so lcb_grade.py keeps its own copy.
def source_module(root: Path, name: str):
    """Import ``name`` from the checkout at ``root``, refusing a copy installed elsewhere."""
    sys.path.insert(0, str(root))
    module = importlib.import_module(name)
    expected = (root / (name.replace(".", "/") + ".py")).resolve()
    if module.__file__ is None or Path(module.__file__).resolve() != expected:
        raise RuntimeError(f"Unexpected installed source module: {name}")
    return module


def sql_score(root: Path, config: dict, answer: str) -> dict:
    sql = source_module(root, SQL_MODULE)
    statements = [item for item in sql.split_statements(config["reference_sql"]) if item.strip()]
    if len(statements) != 1 or sql.classify_statement(statements[0]) != "select":
        raise ValueError("Reference is not one SELECT")
    reference = statements[0]
    if sql.is_nondeterministic(reference):
        raise ValueError("Reference is nondeterministic")
    creates, inserts, names = [], [], []
    for statement in sql.split_statements(config["context_sql"]):
        kind = sql.classify_statement(statement)
        if kind == "create_table":
            if sql.create_table_is_schema_qualified(statement):
                raise ValueError("Context has a schema-qualified CREATE TABLE")
            creates.append(statement)
            name = sql.create_table_name(statement)
            if name:
                names.append(name)
        elif kind == "insert":
            inserts.append(statement)
        else:
            raise ValueError(f"Unsupported context statement: {kind}")
    if not creates or not inserts:
        raise ValueError("Context requires CREATE TABLE and INSERT")
    spec = {
        "schema_sql": "\n".join(item.strip().rstrip(";") + ";" for item in creates),
        "insert_sql": "\n".join(item.strip().rstrip(";") + ";" for item in inserts),
        "reference_sql": reference,
        "order_significant": sql.has_top_level_order_by(reference),
        "table_names": sorted(names),
    }
    outcome, detail = sql.grade(spec, reference)
    if outcome == sql.GradeOutcome.INFRA:
        raise ValueError(f"Invalid reference: {detail}")
    outcome, detail = sql.grade(spec, sql.extract_sql(answer))
    if outcome == sql.GradeOutcome.INFRA:
        raise ValueError(f"Invalid grading: {detail}")
    return {"reward": 1.0 if outcome == sql.GradeOutcome.MATCH else 0.0, "detail": {"comparison": detail}}


def main() -> None:
    root, config_path, answer_path, score_path = map(Path, sys.argv[1:5])
    config = json.loads(config_path.read_text())
    answer = answer_path.read_text() if answer_path.exists() else ""
    score_path.write_text(json.dumps(sql_score(root, config, answer), allow_nan=False))


if __name__ == "__main__":
    main()

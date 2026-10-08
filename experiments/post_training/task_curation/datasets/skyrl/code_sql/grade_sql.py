# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade SQL queries through the original image-installed SQL comparator."""

import importlib
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path("/opt/skyrl_gym")
SQL_PATH = SOURCE_ROOT / "skyrl_gym/envs/text_to_sql/scoring.py"


def source_module(name: str, expected: Path):
    sys.path.insert(0, str(SOURCE_ROOT))
    module = importlib.import_module(name)
    if module.__file__ is None or Path(module.__file__).resolve() != expected:
        raise RuntimeError(f"Unexpected installed source module: {name}")
    return module


def sql_score(config: dict, answer: str) -> dict:
    sql = source_module("skyrl_gym.envs.text_to_sql.scoring", SQL_PATH)
    statements = [item for item in sql.split_statements(config["reference_sql"]) if item.strip()]
    if len(statements) != 1 or sql.classify_statement(statements[0]) != "select":
        raise ValueError("Gretel reference is not one SELECT")
    reference = statements[0]
    if sql.is_nondeterministic(reference):
        raise ValueError("Gretel reference is nondeterministic")
    creates, inserts, names = [], [], []
    for statement in sql.split_statements(config["context_sql"]):
        kind = sql.classify_statement(statement)
        if kind == "create_table":
            if sql.create_table_is_schema_qualified(statement):
                raise ValueError("Gretel context has a schema-qualified CREATE TABLE")
            creates.append(statement)
            name = sql.create_table_name(statement)
            if name:
                names.append(name)
        elif kind == "insert":
            inserts.append(statement)
        else:
            raise ValueError(f"Unsupported Gretel context statement: {kind}")
    if not creates or not inserts:
        raise ValueError("Gretel context requires CREATE TABLE and INSERT")
    spec = {
        "schema_sql": "\n".join(item.strip().rstrip(";") + ";" for item in creates),
        "insert_sql": "\n".join(item.strip().rstrip(";") + ";" for item in inserts),
        "reference_sql": reference,
        "order_significant": sql.has_top_level_order_by(reference),
        "table_names": sorted(names),
    }
    outcome, detail = sql.grade(spec, reference)
    if outcome == sql.GradeOutcome.INFRA:
        raise ValueError(f"Invalid Gretel source reference: {detail}")
    outcome, detail = sql.grade(spec, sql.extract_sql(answer))
    if outcome == sql.GradeOutcome.INFRA:
        raise ValueError(f"Invalid Gretel source grading: {detail}")
    return {"reward": 1.0 if outcome == sql.GradeOutcome.MATCH else 0.0, "detail": {"comparison": detail}}


def main() -> None:
    config_path, answer_path, score_path = map(Path, sys.argv[1:4])
    config = json.loads(config_path.read_text())
    answer = answer_path.read_text() if answer_path.exists() else ""
    score_path.write_text(json.dumps(sql_score(config, answer), allow_nan=False))


if __name__ == "__main__":
    main()

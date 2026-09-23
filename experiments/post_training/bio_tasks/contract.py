# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trusted, record-level verification; copied into Harbor's private tests directory."""

import argparse
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from tasktrove_verify.grade import Reward, infra_error, invalid_task, scored, write_reward

MAX_ANSWER_BYTES = 2 * 1024 * 1024
Scalar = int | float | str | None


class Column(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["integer", "number", "text"]
    description: str
    unit: str
    nullable: bool = False
    atol: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    rtol: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    def accepts(self, value: object) -> bool:
        if value is None:
            return self.nullable
        if self.kind == "text":
            return isinstance(value, str)
        if self.kind == "integer":
            return type(value) is int
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal["1"] = "1"
    columns: dict[str, Column]
    expected: dict[str, dict[str, Scalar]]

    @model_validator(mode="after")
    def validate_reference(self) -> "Contract":
        if not self.columns or not self.expected or "id" in self.columns:
            raise ValueError("a contract needs columns, expected records, and a separate id field")
        for record_id, row in self.expected.items():
            if not record_id or set(row) != set(self.columns):
                raise ValueError("reference IDs must be nonempty and every record must have every column")
            for name, column in self.columns.items():
                value = row[name]
                if not column.accepts(value):
                    raise ValueError(f"invalid reference type or nonfinite value: {record_id}.{name}")
                if column.kind == "number" and value is not None:
                    if not math.isfinite(column.atol + column.rtol * abs(float(value))):
                        raise ValueError(f"nonfinite tolerance: {record_id}.{name}")
        return self

    def answer(self) -> list[dict]:
        return [{"id": key, **value} for key, value in self.expected.items()]

    def instructions(self) -> str:
        lines = [
            "Write /app/answer.json as a JSON array of records, with one record per requested id.",
            "Each record must contain a string id and exactly the fields below. Record order is irrelevant.",
            "Do not duplicate or omit IDs. Extra IDs/fields, booleans as numbers, and nonfinite numbers fail.",
        ]
        for name, column in self.columns.items():
            tolerance = ""
            if column.kind == "number":
                tolerance = f"; absolute tolerance {column.atol}, relative tolerance {column.rtol}"
            nullable = "; use null when specified" if column.nullable else ""
            lines.append(f"- {name}: {column.kind} ({column.unit}); {column.description}{nullable}{tolerance}.")
        lines.append("Numeric tolerance is abs(actual - reference) <= atol + rtol * abs(reference).")
        return "\n".join(lines)


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def grade_answer(contract: Contract, text: str) -> Reward:
    """Check identities, schema, types, missingness, and every biological quantity."""
    try:
        records = json.loads(text, object_pairs_hook=unique_object)
    except (ValueError, RecursionError) as error:
        return scored(0, reason="malformed_json", error=str(error))
    if not isinstance(records, list):
        return scored(0, reason="expected_record_array")
    by_id = {}
    fields = {"id", *contract.columns}
    for row in records:
        if not isinstance(row, dict) or set(row) != fields or not isinstance(row["id"], str):
            return scored(0, reason="record_schema")
        if row["id"] in by_id:
            return scored(0, reason="duplicate_id", id=row["id"])
        by_id[row["id"]] = row
    if set(by_id) != set(contract.expected):
        return scored(
            0,
            reason="record_identity",
            missing=sorted(set(contract.expected) - set(by_id)),
            unexpected=sorted(set(by_id) - set(contract.expected)),
        )
    checks = []
    for record_id, reference in contract.expected.items():
        for name, column in contract.columns.items():
            actual, expected = by_id[record_id][name], reference[name]
            passed = column.accepts(actual)
            if actual is None or expected is None or column.kind != "number":
                passed = passed and actual == expected
            elif passed:
                passed = abs(actual - expected) <= column.atol + column.rtol * abs(expected)
            reported = str(actual) if type(actual) is float and not math.isfinite(actual) else actual
            checks.append({"id": record_id, "field": name, "passed": passed, "expected": expected, "actual": reported})
    return scored(float(all(check["passed"] for check in checks)), checks=checks)


def grade_files(reference: Path, answer: Path) -> Reward:
    try:
        contract = Contract.model_validate_json(reference.read_text())
    except (FileNotFoundError, ValidationError, ValueError) as error:
        return invalid_task(str(error))
    except OSError as error:
        return infra_error(str(error))
    try:
        if answer.is_symlink() or not answer.is_file():
            return scored(0, reason="missing_or_nonregular_answer")
        with answer.open("rb") as handle:
            raw = handle.read(MAX_ANSWER_BYTES + 1)
        if len(raw) > MAX_ANSWER_BYTES:
            return scored(0, reason="answer_too_large")
        return grade_answer(contract, raw.decode("utf-8"))
    except UnicodeDecodeError:
        return scored(0, reason="malformed_utf8")
    except OSError as error:
        return infra_error(str(error))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("/tests/reference.json"))
    parser.add_argument("--answer", type=Path, default=Path("/app/answer.json"))
    parser.add_argument("--logs", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    write_reward(args.logs, grade_files(args.reference, args.answer))


if __name__ == "__main__":
    main()

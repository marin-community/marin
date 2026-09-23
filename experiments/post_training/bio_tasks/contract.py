# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trusted, record-level verification; copied into Harbor's private tests directory."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from tasktrove_verify.grade import Reward, infra_error, invalid_task, scored, write_reward

MAX_ANSWER_BYTES = 2 * 1024 * 1024
Scalar = int | float | str | None


class FastqContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    records: int = Field(ge=0)
    max_bytes: int = Field(gt=0, le=128 * 1024 * 1024)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def fastq_digest(path: Path, max_bytes: int) -> tuple[int, str]:
    """Stream an ordered FASTQ profile, hashing IDs, bases and qualities exactly."""
    digest = hashlib.sha256()
    identifiers = set()
    consumed = 0
    with path.open("rb") as handle:
        while header := handle.readline(max_bytes + 1):
            lines = [header, *(handle.readline(max_bytes + 1) for _ in range(3))]
            consumed += sum(map(len, lines))
            if consumed > max_bytes:
                raise ValueError("fastq_too_large")
            header, sequence, separator, qualities = (line.rstrip(b"\r\n") for line in lines)
            if not header.startswith(b"@") or not separator.startswith(b"+"):
                raise ValueError("malformed_fastq")
            identifier = header[1:].split()[0] if header[1:].split() else b""
            if not identifier or identifier in identifiers or not sequence or len(sequence) != len(qualities):
                raise ValueError("fastq_identity_or_length")
            if any(base not in b"ACGTRYMKSWBDHVN" for base in sequence) or any(q < 33 or q > 126 for q in qualities):
                raise ValueError("fastq_sequence_or_quality")
            identifiers.add(identifier)
            digest.update(identifier + b"\n" + sequence + b"\n" + qualities + b"\n")
    return len(identifiers), digest.hexdigest()


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
    fastq: dict[str, FastqContract] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reference(self) -> "Contract":
        if not self.columns or not self.expected or "id" in self.columns:
            raise ValueError("a contract needs columns, expected records, and a separate id field")
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in self.fastq):
            raise ValueError("FASTQ artifacts must use simple filenames")
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
        for name in self.fastq:
            lines.append(
                f"Also write /app/{name} as four-line FASTQ. Preserve retained input-record order, first-token "
                "read IDs, bases and qualities exactly as specified. Header comments and '+' comments are ignored. "
                "Complete records and identities are verified; a correct JSON summary alone does not pass."
            )
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
        verdict = grade_answer(contract, raw.decode("utf-8"))
        if verdict.reward != 1 or not contract.fastq:
            return verdict
        checks = {}
        for name, target in contract.fastq.items():
            path = answer.parent / name
            if path.is_symlink() or not path.is_file():
                return scored(0, reason="missing_or_nonregular_fastq", artifact=name)
            try:
                records, digest = fastq_digest(path, target.max_bytes)
            except ValueError as error:
                return scored(0, reason=str(error), artifact=name)
            checks[name] = {
                "records": records,
                "sha256": digest,
                "passed": records == target.records and digest == target.sha256,
            }
        return scored(float(all(check["passed"] for check in checks.values())), **verdict.detail, artifact_checks=checks)
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

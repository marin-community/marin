# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trusted, record-level verification; copied into Harbor's private tests directory."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import zlib
from itertools import combinations
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from tasktrove_verify.grade import Reward, infra_error, invalid_task, scored, write_reward

from experiments.post_training.bio_tasks.solvers.newick import newick, weighted_splits

MAX_ANSWER_BYTES = 2 * 1024 * 1024
MAX_MATRIX_LINE_BYTES = 64 * 1024
MATRIX_SORT_TIMEOUT = 120
Scalar = int | float | str | None


class FastaContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sequences: dict[str, str] = Field(min_length=1)
    max_bytes: int = Field(gt=0, le=128 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_sequences(self) -> "FastaContract":
        for identifier, sequence in self.sequences.items():
            if not re.fullmatch(r"[!-~]+", identifier) or identifier.startswith(">"):
                raise ValueError("FASTA identities must be nonempty ASCII header tokens")
            if not re.fullmatch(r"[A-Z*]+", sequence):
                raise ValueError("FASTA references must be nonempty uppercase unaligned sequences")
        return self


def check_fasta(path: Path, target: FastaContract) -> dict:
    """Verify every identity and residue without retaining submitted sequences."""
    identifiers = set()
    current = None
    offset = 0
    consumed = 0
    residues = 0
    with path.open("rb") as handle:
        while line := handle.readline(target.max_bytes - consumed + 1):
            consumed += len(line)
            if consumed > target.max_bytes:
                raise ValueError("fasta_too_large")
            text = line.decode("ascii").strip()
            if not text:
                continue
            if text.startswith(">"):
                if current is not None and offset != len(target.sequences[current]):
                    raise ValueError("fasta_missing_residues")
                fields = text[1:].split()
                if not fields or fields[0] in identifiers or fields[0] not in target.sequences:
                    raise ValueError("fasta_identity")
                current = fields[0]
                identifiers.add(current)
                offset = 0
                continue
            if current is None:
                raise ValueError("fasta_missing_header")
            if text.upper() != target.sequences[current][offset : offset + len(text)]:
                raise ValueError("fasta_changed_residues")
            offset += len(text)
            residues += len(text)
    if identifiers != set(target.sequences):
        raise ValueError("fasta_missing_records")
    if current is not None and offset != len(target.sequences[current]):
        raise ValueError("fasta_missing_residues")
    return {"passed": True, "records": len(identifiers), "residues": residues}


class AlignmentContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sequences: dict[str, str]
    scoring: dict[str, int]
    gap_open: int = Field(gt=0)
    gap_extend: int = Field(gt=0)
    minimum_score: int
    max_columns: int = Field(gt=0, le=4096)
    max_bytes: int = Field(gt=0, le=2 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_sequences(self) -> "AlignmentContract":
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        if not 2 <= len(self.sequences) <= 64:
            raise ValueError("Alignment needs 2 to 64 protein sequences")
        if set(self.scoring) != {a + b for a in amino_acids for b in amino_acids}:
            raise ValueError("Alignment scoring must cover all amino-acid pairs")
        for key, sequence in self.sequences.items():
            if not key or any(c.isspace() for c in key) or not sequence or len(sequence) > self.max_columns:
                raise ValueError("Invalid alignment input identity or length")
            if not set(sequence) <= set(amino_acids):
                raise ValueError("Alignment inputs must use canonical amino acids")
        return self


def alignment_score(sequences: list[str], scoring: dict[str, int], opening: int, extension: int) -> int:
    """Sum affine-gap pair scores after removing each pair's double-gap columns."""
    total = 0
    for left, right in combinations(sequences, 2):
        gap = 0
        for a, b in zip(left, right, strict=True):
            if a == b == "-":
                continue
            current = 1 if a == "-" else 2 if b == "-" else 0
            if current:
                total -= extension if current == gap else opening
            else:
                total += scoring[a + b]
            gap = current
    return total


def check_alignment(path: Path, target: AlignmentContract) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(target.max_bytes + 1)
    if len(raw) > target.max_bytes:
        raise ValueError("alignment_too_large")
    sequences = {}
    current = None
    for line in raw.decode("ascii").splitlines():
        line = line.strip()
        if line.startswith(">"):
            fields = line[1:].split()
            if not fields or fields[0] in sequences:
                raise ValueError("alignment_duplicate_or_empty_id")
            current = fields[0]
            sequences[current] = ""
        elif line:
            if current is None or not set(line.upper()) <= set("ACDEFGHIKLMNPQRSTVWY-"):
                raise ValueError("malformed_alignment")
            sequences[current] += line.upper()
    if set(sequences) != set(target.sequences):
        raise ValueError("alignment_identity")
    widths = {len(sequence) for sequence in sequences.values()}
    if len(widths) != 1 or not 0 < next(iter(widths)) <= target.max_columns:
        raise ValueError("alignment_shape")
    if any(sequence.replace("-", "") != target.sequences[key] for key, sequence in sequences.items()):
        raise ValueError("alignment_changed_residues")
    if any(set(column) == {"-"} for column in zip(*sequences.values(), strict=True)):
        raise ValueError("alignment_all_gap_column")
    score = alignment_score(list(sequences.values()), target.scoring, target.gap_open, target.gap_extend)
    return {"score": score, "minimum_score": target.minimum_score, "passed": score >= target.minimum_score}


class FastqContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    records: int = Field(ge=0)
    max_bytes: int = Field(gt=0, le=128 * 1024 * 1024)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TreeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    reference: str = Field(max_length=128 * 1024)
    atol: float = Field(ge=0, allow_inf_nan=False)
    rtol: float = Field(ge=0, allow_inf_nan=False)
    max_bytes: int = Field(gt=0, le=128 * 1024)

    @model_validator(mode="after")
    def validate_reference(self) -> "TreeContract":
        _, edges = weighted_splits(newick(self.reference))
        if any(not math.isfinite(self.atol + self.rtol * length) for length in edges.values()):
            raise ValueError("nonfinite tree tolerance")
        return self


class MatrixMarketContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    rows: int = Field(gt=0)
    columns: int = Field(gt=0)
    nonzeros: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_bytes: int = Field(gt=0, le=1024 * 1024 * 1024)
    max_decoded_bytes: int = Field(gt=0, le=2 * 1024 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_shape(self) -> "MatrixMarketContract":
        if self.nonzeros > self.rows * self.columns:
            raise ValueError("Matrix nonzero count exceeds its dimensions")
        return self


def check_matrix(path: Path, target: MatrixMarketContract) -> dict:
    """Check every positive integer entry, allowing arbitrary coordinate order."""
    if path.stat().st_size > target.max_bytes:
        raise ValueError("matrix_too_large")
    with tempfile.TemporaryDirectory(prefix="bio-matrix-") as directory:
        root = Path(directory)
        coordinates = root / "coordinates.txt"
        opener = gzip.open if path.suffix == ".gz" else open
        try:
            with opener(path, "rb") as source, coordinates.open("wb") as destination:
                header = source.readline(MAX_MATRIX_LINE_BYTES + 1)
                consumed = len(header)
                if len(header) > MAX_MATRIX_LINE_BYTES or consumed > target.max_decoded_bytes:
                    raise ValueError("matrix_decoded_size")
                if header.lower().split() != [b"%%matrixmarket", b"matrix", b"coordinate", b"integer", b"general"]:
                    raise ValueError("matrix_header")
                shape = None
                count = 0
                while line := source.readline(MAX_MATRIX_LINE_BYTES + 1):
                    consumed += len(line)
                    if len(line) > MAX_MATRIX_LINE_BYTES or consumed > target.max_decoded_bytes:
                        raise ValueError("matrix_decoded_size")
                    stripped = line.strip()
                    if not stripped or stripped.startswith(b"%"):
                        continue
                    fields = stripped.split()
                    if len(fields) != 3 or any(not re.fullmatch(rb"\+?[0-9]+", field) for field in fields):
                        raise ValueError("matrix_integer_coordinates")
                    values = tuple(int(field) for field in fields)
                    if shape is None:
                        shape = values
                        if shape != (target.rows, target.columns, target.nonzeros):
                            raise ValueError("matrix_shape")
                        continue
                    row, column, value = values
                    if not (1 <= row <= target.rows and 1 <= column <= target.columns and 0 < value < 2**63):
                        raise ValueError("matrix_coordinate_or_count")
                    count += 1
                    if count > target.nonzeros:
                        raise ValueError("matrix_extra_entries")
                    destination.write(f"{row} {column} {value}\n".encode())
                if shape is None or count != target.nonzeros:
                    raise ValueError("matrix_missing_entries")
        except (gzip.BadGzipFile, EOFError, zlib.error) as error:
            raise ValueError("malformed_gzip_matrix") from error
        ordered = root / "ordered.txt"
        with ordered.open("wb") as destination:
            subprocess.run(
                [
                    "sort",
                    "--buffer-size=32M",
                    "--parallel=1",
                    "--temporary-directory=" + str(root),
                    "-k1,1n",
                    "-k2,2n",
                    str(coordinates),
                ],
                stdout=destination,
                stderr=subprocess.PIPE,
                check=True,
                timeout=MATRIX_SORT_TIMEOUT,
                env={**os.environ, "LC_ALL": "C"},
            )
        digest = hashlib.sha256()
        previous = None
        with ordered.open("rb") as source:
            for line in source:
                coordinate = line.split()[:2]
                if coordinate == previous:
                    raise ValueError("matrix_duplicate_coordinate")
                previous = coordinate
                digest.update(line)
        actual = digest.hexdigest()
        return {"passed": actual == target.sha256, "nonzeros": count, "sha256": actual}


def check_tree(path: Path, target: TreeContract) -> dict:
    """Check all unrooted splits and edge lengths, independent of Newick layout."""
    with path.open("rb") as handle:
        raw = handle.read(target.max_bytes + 1)
    if len(raw) > target.max_bytes:
        raise ValueError("tree_too_large")
    tips, edges = weighted_splits(newick(raw.decode("ascii")))
    reference_tips, reference_edges = weighted_splits(newick(target.reference))
    if tips != reference_tips:
        raise ValueError("tree_identity")
    if edges.keys() != reference_edges.keys():
        return {"passed": False, "reason": "tree_topology"}
    failures = [
        key
        for key, length in reference_edges.items()
        if abs(edges[key] - length) > target.atol + target.rtol * abs(length)
    ]
    return {"passed": not failures, "tips": len(tips), "edges": len(edges), "changed_edges": failures[:20]}


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


class TableContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    columns: dict[str, Column]
    expected: dict[str, dict[str, Scalar]]
    max_bytes: int = Field(gt=0, le=128 * 1024 * 1024)

    @model_validator(mode="after")
    def validate_reference(self) -> "TableContract":
        validate_records(self.columns, self.expected)
        return self


def validate_records(columns: dict[str, Column], expected: dict[str, dict[str, Scalar]]) -> None:
    if not columns or not expected or "id" in columns:
        raise ValueError("a contract needs columns, expected records, and a separate id field")
    for record_id, row in expected.items():
        if not record_id or set(row) != set(columns):
            raise ValueError("reference IDs must be nonempty and every record must have every column")
        for name, column in columns.items():
            value = row[name]
            if not column.accepts(value):
                raise ValueError(f"invalid reference type or nonfinite value: {record_id}.{name}")
            if column.kind == "number" and value is not None:
                if not math.isfinite(column.atol + column.rtol * abs(float(value))):
                    raise ValueError(f"nonfinite tolerance: {record_id}.{name}")


def matches(column: Column, actual: object, expected: Scalar) -> bool:
    if not column.accepts(actual):
        return False
    if actual is None or expected is None or column.kind != "number":
        return actual == expected
    return abs(actual - expected) <= column.atol + column.rtol * abs(expected)


def check_table(path: Path, target: TableContract) -> dict:
    """Check every TSV quantity while bounding bytes, rows and diagnostic output."""
    identifiers = set()
    consumed = 0
    failures = []
    with path.open("rb") as handle:
        header = handle.readline(target.max_bytes + 1)
        consumed += len(header)
        if consumed > target.max_bytes:
            raise ValueError("table_too_large")
        fields = next(csv.reader([header.decode("utf-8")], delimiter="\t"))
        if len(fields) != len(set(fields)) or set(fields) != {"id", *target.columns}:
            raise ValueError("table_schema")
        while line := handle.readline(target.max_bytes - consumed + 1):
            consumed += len(line)
            if consumed > target.max_bytes:
                raise ValueError("table_too_large")
            values = next(csv.reader([line.decode("utf-8")], delimiter="\t"))
            if len(values) != len(fields):
                raise ValueError("table_schema")
            row = dict(zip(fields, values, strict=True))
            key = row.pop("id")
            if key not in target.expected or key in identifiers:
                raise ValueError("table_identity")
            identifiers.add(key)
            for name, column in target.columns.items():
                value = row[name]
                if value == "NA" and column.nullable:
                    actual = None
                elif column.kind == "integer":
                    actual = int(value)
                elif column.kind == "number":
                    actual = float(value)
                else:
                    actual = value
                if not matches(column, actual, target.expected[key][name]) and len(failures) < 20:
                    failures.append({"id": key, "field": name})
    missing = sorted(set(target.expected) - identifiers)
    return {
        "passed": not failures and not missing,
        "rows": len(identifiers),
        "missing": missing[:20],
        "failures": failures,
    }


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal["1"] = "1"
    columns: dict[str, Column]
    expected: dict[str, dict[str, Scalar]]
    fasta: dict[str, FastaContract] = Field(default_factory=dict)
    fastq: dict[str, FastqContract] = Field(default_factory=dict)
    alignments: dict[str, AlignmentContract] = Field(default_factory=dict)
    tables: dict[str, TableContract] = Field(default_factory=dict)
    trees: dict[str, TreeContract] = Field(default_factory=dict)
    matrices: dict[str, MatrixMarketContract] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reference(self) -> "Contract":
        validate_records(self.columns, self.expected)
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in self.artifacts()):
            raise ValueError("Artifacts must use simple filenames")
        if len(set(self.artifacts())) != len(self.artifacts()) or "answer.json" in self.artifacts():
            raise ValueError("Artifact filenames must be distinct")
        return self

    def answer(self) -> list[dict]:
        return [{"id": key, **value} for key, value in self.expected.items()]

    def artifacts(self) -> tuple[str, ...]:
        return (*self.fasta, *self.fastq, *self.alignments, *self.tables, *self.trees, *self.matrices)

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
        for name, target in self.fasta.items():
            lines.append(
                f"Also write /app/{name} as unaligned FASTA, at most {target.max_bytes} bytes. "
                "Every requested first-token ID and its complete sequence must occur exactly once. "
                "Record order, letter case, line wrapping, blank lines and header comments are ignored. "
                "Do not add gaps or change residues. Every sequence is verified; a correct JSON summary "
                "alone does not pass."
            )
        for name in self.fastq:
            lines.append(
                f"Also write /app/{name} as four-line FASTQ. Preserve retained input-record order, first-token "
                "read IDs, bases and qualities exactly as specified. Header comments and '+' comments are ignored. "
                "Complete records and identities are verified; a correct JSON summary alone does not pass."
            )
        for name, target in self.alignments.items():
            lines.append(
                f"Also write /app/{name} as aligned protein FASTA. Preserve each input ID and all residues in order; "
                f"insert only '-' gaps. All rows must have equal width, at most {target.max_columns} columns, "
                "with no all-gap columns. Letter case, line wrapping and record order are ignored. "
                f"The sum-of-pairs score must be at least {target.minimum_score}, using query.json's integer "
                "substitution matrix and gap penalties. For each unordered pair remove double-gap columns, "
                "charge gap_open for the first residue of each gap run and gap_extend for later residues, "
                "including terminal gaps. Switching the gapped sequence starts a new run. "
                "Different qualifying alignments are accepted; this objective does not establish "
                "true biological homology."
            )
        for name, target in self.tables.items():
            lines.append(
                f"Also write /app/{name} as UTF-8 TSV, with an id column and exactly these columns: "
                + ", ".join(target.columns)
                + ". Row and column order are irrelevant; duplicate, missing and extra IDs fail. "
                "Use NA for specified missing values. Fields must occupy one line. Every row is verified; "
                "a correct JSON summary alone does not pass."
            )
            for column_name, column in target.columns.items():
                lines.append(
                    f"- {name}.{column_name}: {column.kind} ({column.unit}); {column.description}; "
                    f"absolute tolerance {column.atol}, relative tolerance {column.rtol}."
                )
        for name, target in self.trees.items():
            lines.append(
                f"Also write /app/{name} as one Newick tree with a terminal semicolon. Use each requested "
                "accession exactly once as a leaf, and finite nonnegative lengths on every edge. "
                "The complete unrooted topology and every branch length are checked against the specified "
                f"method (absolute tolerance {target.atol}, relative tolerance {target.rtol}). "
                "Child order, internal labels, comments and root placement are ignored; a degree-two root's "
                "two edges are added. Do not include unary nodes or a nonzero root stem. "
                "This verifies reproduction of the specified analysis, not a true species phylogeny."
            )
        for name, target in self.matrices.items():
            lines.append(
                f"Also write /app/{name} as MatrixMarket coordinate integer general, "
                f"with {target.rows} rows, {target.columns} columns and {target.nonzeros} positive entries. "
                "Use one 1-based coordinate per nonzero count, with no duplicate coordinates or explicit zeros. "
                "Every count and coordinate is checked; entry order, whitespace and comments are ignored. "
                "Use gzip compression for a .gz filename. Integer counts must be below 2^63. "
                f"The file must be at most {target.max_bytes} bytes, at most {target.max_decoded_bytes} "
                f"decoded bytes, and at most {MAX_MATRIX_LINE_BYTES} bytes per line. "
                "A correct JSON summary alone does not pass."
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
            passed = matches(column, actual, expected)
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
        if verdict.reward != 1 or not contract.artifacts():
            return verdict
        checks = {}
        for name, target in contract.fasta.items():
            path = answer.parent / name
            if path.is_symlink() or not path.is_file():
                return scored(0, reason="missing_or_nonregular_fasta", artifact=name)
            try:
                checks[name] = check_fasta(path, target)
            except ValueError as error:
                return scored(0, reason=str(error), artifact=name)
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
        for name, target in contract.alignments.items():
            path = answer.parent / name
            if path.is_symlink() or not path.is_file():
                return scored(0, reason="missing_or_nonregular_alignment", artifact=name)
            try:
                checks[name] = check_alignment(path, target)
            except ValueError as error:
                return scored(0, reason=str(error), artifact=name)
        for name, target in contract.tables.items():
            path = answer.parent / name
            if path.is_symlink() or not path.is_file():
                return scored(0, reason="missing_or_nonregular_table", artifact=name)
            try:
                checks[name] = check_table(path, target)
            except (ValueError, csv.Error) as error:
                return scored(0, reason=str(error), artifact=name)
        for name, target in contract.trees.items():
            path = answer.parent / name
            if path.is_symlink() or not path.is_file():
                return scored(0, reason="missing_or_nonregular_tree", artifact=name)
            try:
                checks[name] = check_tree(path, target)
            except (ValueError, RecursionError) as error:
                return scored(0, reason=str(error), artifact=name)
        if any(not check["passed"] for check in checks.values()):
            return scored(0, **verdict.detail, artifact_checks=checks)
        for name, target in contract.matrices.items():
            path = answer.parent / name
            if path.is_symlink() or not path.is_file():
                return scored(0, reason="missing_or_nonregular_matrix", artifact=name)
            try:
                checks[name] = check_matrix(path, target)
            except ValueError as error:
                return scored(0, reason=str(error), artifact=name)
        return scored(float(all(check["passed"] for check in checks.values())), **verdict.detail, artifact_checks=checks)
    except UnicodeDecodeError:
        return scored(0, reason="malformed_utf8")
    except OSError as error:
        return infra_error(str(error))
    except subprocess.SubprocessError as error:
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

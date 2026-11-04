# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate deterministic, objectively graded repertoire-analysis TaskSpecs.

The input is a compact donor-by-cell-type count table extracted from a pinned
immune-repertoire object. The generator deliberately knows nothing about the
source notebook's conclusions: every hidden answer is recomputed from the same
solver-visible table under an explicit analysis contract.
"""

import argparse
import base64
import csv
import hashlib
import itertools
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    InlineFile,
    JsonValueAnswer,
    ResourceGroups,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import StructuredExactSpec

GENERATOR_VERSION = "covid-repertoire-objective-v1"
DATA_PATH = "workspace/data/repertoire_counts.csv"
DATA_DICTIONARY_PATH = "workspace/reference/data_dictionary.md"
BIOLOGY_CARD_PATH = "workspace/reference/mait_background.md"
DATA_DICTIONARY = b"""# Data dictionary

Each row is one pseudonymous donor and coarse T-cell state. `n_cells` is the number of post-QC cells,
`n_primary_vj_called` is the number with a callable primary VJ chain, and `n_trav1_2` is the called subset whose
primary V-alpha gene is TRAV1-2. Donors, rather than individual cells, are the biological replicate.
"""
BIOLOGY_CARD = b"""# MAIT receptor-usage background

Mucosal-associated invariant T (MAIT) cells commonly use a semi-invariant T-cell receptor containing TRAV1-2.
Strong TRAV1-2 enrichment that is consistent across donors therefore provides evidence supporting a proposed MAIT
annotation. TRAV1-2 is not exclusive to MAIT cells, so receptor usage alone does not prove the annotation or antigen
specificity. Orthogonal transcriptomic markers, MR1-tetramer binding, or functional evidence can strengthen it.

For the annotation-assessment tasks in this pilot, classify the evidence as `supported_not_proven` exactly when the
candidate median is at least 0.5, the comparator median is at most 0.1, at least five candidate donors pass the
support filter, and candidate enrichment is at least five-fold. Otherwise use `inconclusive`. The required limitation
code is `receptor_usage_not_annotation_proof`.

Source context: Stephenson et al., Nature Medicine (2021), DOI 10.1038/s41591-021-01329-2.
"""
REQUIRED_COLUMNS = (
    "donor_id",
    "cell_type_coarse",
    "n_cells",
    "n_primary_vj_called",
    "n_trav1_2",
)


@dataclass(frozen=True)
class CountRow:
    """One donor-by-cell-type count row."""

    donor_id: str
    cell_type: str
    n_cells: int
    n_called: int
    n_trav1_2: int

    @property
    def called_fraction(self) -> float:
        return self.n_trav1_2 / self.n_called


@dataclass(frozen=True)
class Candidate:
    """A deterministic task candidate before TaskSpec packaging."""

    family: str
    parameters: dict[str, Any]
    instruction: str
    expected: dict[str, Any]

    @property
    def identity(self) -> str:
        payload = json.dumps(
            {"family": self.family, "parameters": self.parameters}, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def read_counts(data: bytes) -> list[CountRow]:
    """Parse and fail loudly on malformed or internally inconsistent counts."""
    text = data.decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
        raise ValueError(f"expected columns {REQUIRED_COLUMNS}, got {tuple(reader.fieldnames or ())}")
    rows: list[CountRow] = []
    seen: set[tuple[str, str]] = set()
    for line_number, record in enumerate(reader, start=2):
        donor = record["donor_id"].strip()
        cell_type = record["cell_type_coarse"].strip()
        if not donor or not cell_type:
            raise ValueError(f"line {line_number} has an empty donor or cell type")
        try:
            n_cells = int(record["n_cells"])
            n_called = int(record["n_primary_vj_called"])
            n_trav1_2 = int(record["n_trav1_2"])
        except ValueError as error:
            raise ValueError(f"line {line_number} has a non-integer count") from error
        if not 0 <= n_trav1_2 <= n_called <= n_cells:
            raise ValueError(f"line {line_number} violates 0 <= TRAV1-2 <= called <= cells")
        key = (donor, cell_type)
        if key in seen:
            raise ValueError(f"duplicate donor/cell-type row {key}")
        seen.add(key)
        rows.append(CountRow(donor, cell_type, n_cells, n_called, n_trav1_2))
    if not rows:
        raise ValueError("count table is empty")
    return rows


def rounded(value: float) -> float:
    """Return the six-decimal representation required by every task."""
    return round(value, 6)


def eligible(rows: list[CountRow], cell_type: str, minimum_called: int) -> list[CountRow]:
    return [row for row in rows if row.cell_type == cell_type and row.n_called >= minimum_called]


def aggregate_candidates(rows: list[CountRow]) -> list[Candidate]:
    candidates = []
    for cell_type in sorted({row.cell_type for row in rows}):
        selected = [row for row in rows if row.cell_type == cell_type]
        called = sum(row.n_called for row in selected)
        trav = sum(row.n_trav1_2 for row in selected)
        if called == 0:
            continue
        expected = {"cell_type": cell_type, "n_called": called, "n_trav1_2": trav, "fraction": rounded(trav / called)}
        instruction = f"""Read /{DATA_PATH}. For cell_type_coarse = {cell_type!r}, sum n_primary_vj_called and
n_trav1_2 across donors. Define fraction = summed n_trav1_2 / summed n_primary_vj_called. Return exactly one JSON
object with keys cell_type, n_called, n_trav1_2, and fraction; round fraction to six decimal places."""
        candidates.append(Candidate("aggregate_usage", {"cell_type": cell_type}, instruction, expected))
    return candidates


def median_candidates(rows: list[CountRow]) -> list[Candidate]:
    candidates = []
    for cell_type, minimum_called in itertools.product(sorted({row.cell_type for row in rows}), (5, 10, 25, 50)):
        selected = eligible(rows, cell_type, minimum_called)
        if len(selected) < 5:
            continue
        value = statistics.median(row.called_fraction for row in selected)
        expected = {
            "cell_type": cell_type,
            "minimum_called": minimum_called,
            "n_donors": len(selected),
            "median_fraction": rounded(value),
        }
        instruction = f"""Read /{DATA_PATH}. Select rows with cell_type_coarse = {cell_type!r} and
n_primary_vj_called >= {minimum_called}. For each selected donor compute n_trav1_2 / n_primary_vj_called, then take
the ordinary median across donors (average the two middle values when needed). Return exactly one JSON object with
keys cell_type, minimum_called, n_donors, and median_fraction; round median_fraction to six decimal places."""
        candidates.append(
            Candidate(
                "donor_median_usage",
                {"cell_type": cell_type, "minimum_called": minimum_called},
                instruction,
                expected,
            )
        )
    return candidates


def prevalence_candidates(rows: list[CountRow]) -> list[Candidate]:
    candidates = []
    for cell_type, minimum_called, cutoff in itertools.product(
        sorted({row.cell_type for row in rows}), (10, 25), (0.01, 0.05, 0.10, 0.25)
    ):
        selected = eligible(rows, cell_type, minimum_called)
        if len(selected) < 5:
            continue
        passing = sum(row.called_fraction >= cutoff for row in selected)
        expected = {
            "cell_type": cell_type,
            "minimum_called": minimum_called,
            "cutoff": cutoff,
            "n_eligible": len(selected),
            "n_passing": passing,
            "prevalence": rounded(passing / len(selected)),
        }
        instruction = f"""Read /{DATA_PATH}. Select rows with cell_type_coarse = {cell_type!r} and
n_primary_vj_called >= {minimum_called}. A donor passes when n_trav1_2 / n_primary_vj_called >= {cutoff}. Return
exactly one JSON object with keys cell_type, minimum_called, cutoff, n_eligible, n_passing, and prevalence, where
prevalence = n_passing / n_eligible rounded to six decimal places."""
        candidates.append(
            Candidate(
                "donor_prevalence",
                {"cell_type": cell_type, "minimum_called": minimum_called, "cutoff": cutoff},
                instruction,
                expected,
            )
        )
    return candidates


def comparison_candidates(rows: list[CountRow]) -> list[Candidate]:
    candidates = []
    cell_types = sorted({row.cell_type for row in rows})
    for left, right in itertools.combinations(cell_types, 2):
        minimum_called = 10
        left_rows = eligible(rows, left, minimum_called)
        right_rows = eligible(rows, right, minimum_called)
        if min(len(left_rows), len(right_rows)) < 5:
            continue
        medians = {
            left: statistics.median(row.called_fraction for row in left_rows),
            right: statistics.median(row.called_fraction for row in right_rows),
        }
        if medians[left] == medians[right]:
            continue
        winner = max(medians, key=medians.__getitem__)
        expected = {
            "left_cell_type": left,
            "right_cell_type": right,
            "minimum_called": minimum_called,
            "left_median": rounded(medians[left]),
            "right_median": rounded(medians[right]),
            "higher_cell_type": winner,
            "absolute_difference": rounded(abs(medians[left] - medians[right])),
        }
        instruction = f"""Read /{DATA_PATH}. Compare {left!r} with {right!r}. Within each cell type retain donors
with n_primary_vj_called >= {minimum_called}, compute each donor's n_trav1_2 / n_primary_vj_called, and take the
ordinary median. Return exactly one JSON object with keys left_cell_type, right_cell_type, minimum_called,
left_median, right_median, higher_cell_type, and absolute_difference. Round all derived numbers to six decimal
places; absolute_difference is the absolute difference between the unrounded medians."""
        candidates.append(Candidate("median_comparison", {"left": left, "right": right}, instruction, expected))
    return candidates


def triangulation_candidates(rows: list[CountRow]) -> list[Candidate]:
    candidates = []
    minimum_called = 10
    mait_rows = eligible(rows, "MAIT", minimum_called)
    if len(mait_rows) < 5:
        return candidates
    mait_median = statistics.median(row.called_fraction for row in mait_rows)
    for comparator in sorted({row.cell_type for row in rows} - {"MAIT"}):
        comparator_rows = eligible(rows, comparator, minimum_called)
        if len(comparator_rows) < 5:
            continue
        comparator_median = statistics.median(row.called_fraction for row in comparator_rows)
        if comparator_median == 0:
            continue
        fold_enrichment = mait_median / comparator_median
        assessment = (
            "supported_not_proven"
            if mait_median >= 0.5 and comparator_median <= 0.1 and len(mait_rows) >= 5 and fold_enrichment >= 5
            else "inconclusive"
        )
        expected = {
            "candidate_cell_type": "MAIT",
            "comparator_cell_type": comparator,
            "minimum_called": minimum_called,
            "candidate_n_donors": len(mait_rows),
            "comparator_n_donors": len(comparator_rows),
            "candidate_median": rounded(mait_median),
            "comparator_median": rounded(comparator_median),
            "fold_enrichment": rounded(fold_enrichment),
            "annotation_assessment": assessment,
            "limitation_code": "receptor_usage_not_annotation_proof",
        }
        instruction = f"""Read /{DATA_PATH}, /{DATA_DICTIONARY_PATH}, and /{BIOLOGY_CARD_PATH}. Evaluate whether
TRAV1-2 usage supports the proposed MAIT annotation relative to {comparator!r}. Retain rows with
n_primary_vj_called >= {minimum_called}; compute ordinary donor-level medians and their candidate/comparator fold
enrichment. Apply the background card's decision policy. Return exactly one JSON object with keys
candidate_cell_type, comparator_cell_type, minimum_called, candidate_n_donors, comparator_n_donors,
candidate_median, comparator_median, fold_enrichment, annotation_assessment, and limitation_code. Round numeric
results to six decimal places."""
        candidates.append(Candidate("annotation_triangulation", {"comparator": comparator}, instruction, expected))
    return candidates


def analysis_unit_candidates(rows: list[CountRow]) -> list[Candidate]:
    candidates = []
    minimum_called = 10
    for cell_type in sorted({row.cell_type for row in rows}):
        selected = eligible(rows, cell_type, minimum_called)
        if len(selected) < 5:
            continue
        pooled = sum(row.n_trav1_2 for row in selected) / sum(row.n_called for row in selected)
        donor_median = statistics.median(row.called_fraction for row in selected)
        expected = {
            "cell_type": cell_type,
            "minimum_called": minimum_called,
            "n_donors": len(selected),
            "pooled_fraction": rounded(pooled),
            "donor_median_fraction": rounded(donor_median),
            "preferred_biological_unit": "donor",
            "reason_code": "avoid_cell_level_pseudoreplication",
        }
        instruction = f"""Read /{DATA_PATH} and /{DATA_DICTIONARY_PATH}. For {cell_type!r}, retain rows with
n_primary_vj_called >= {minimum_called}. Compute (1) the pooled fraction obtained by summing counts before division
and (2) the ordinary median of donor-level fractions. Then identify the preferred biological replicate for a cohort
claim. Return exactly one JSON object with keys cell_type, minimum_called, n_donors, pooled_fraction,
donor_median_fraction, preferred_biological_unit, and reason_code. Round fractions to six decimal places; use the
reason code stated by the data dictionary's replicate warning."""
        candidates.append(Candidate("analysis_unit", {"cell_type": cell_type}, instruction, expected))
    return candidates


def ranking_candidates(rows: list[CountRow]) -> list[Candidate]:
    candidates = []
    for minimum_called in (5, 10, 25, 50):
        medians = {}
        support = {}
        for cell_type in sorted({row.cell_type for row in rows}):
            selected = eligible(rows, cell_type, minimum_called)
            if len(selected) >= 5:
                medians[cell_type] = statistics.median(row.called_fraction for row in selected)
                support[cell_type] = len(selected)
        if len(medians) < 2:
            continue
        winner = max(medians, key=medians.__getitem__)
        expected = {
            "minimum_called": minimum_called,
            "eligible_cell_types": len(medians),
            "highest_cell_type": winner,
            "n_donors": support[winner],
            "median_fraction": rounded(medians[winner]),
        }
        instruction = f"""Read /{DATA_PATH}. For every cell type, retain donors with n_primary_vj_called >=
{minimum_called}. Discard cell types with fewer than five retained donors. Compute donor-level
n_trav1_2 / n_primary_vj_called and rank cell types by the ordinary median. Return exactly one JSON object with keys
minimum_called, eligible_cell_types, highest_cell_type, n_donors, and median_fraction for the highest-ranked cell
type. Round median_fraction to six decimal places. If tied, choose the alphabetically first cell type."""
        candidates.append(Candidate("rank_cell_types", {"minimum_called": minimum_called}, instruction, expected))
    return candidates


def candidates(rows: list[CountRow]) -> list[Candidate]:
    """Return candidates in balanced family order, not family-sized blocks."""
    families = [
        aggregate_candidates(rows),
        median_candidates(rows),
        prevalence_candidates(rows),
        comparison_candidates(rows),
        ranking_candidates(rows),
        triangulation_candidates(rows),
        analysis_unit_candidates(rows),
    ]
    ordered: list[Candidate] = []
    for position in range(max(map(len, families))):
        ordered.extend(family[position] for family in families if position < len(family))
    return ordered


def task_spec(candidate: Candidate, data: bytes, source_revision: str) -> TaskSpec:
    """Package one candidate as an ordinary in-process VerifyIt TaskSpec."""
    task_id = f"covid-repertoire-{candidate.family}-{candidate.identity}"
    package = verifyit_package(StructuredExactSpec(expected=candidate.expected))
    resources = tuple(
        TaskResource(path=path, source=InlineFile(content_base64=base64.b64encode(content).decode()))
        for path, content in (
            (DATA_PATH, data),
            (DATA_DICTIONARY_PATH, DATA_DICTIONARY),
            (BIOLOGY_CARD_PATH, BIOLOGY_CARD),
        )
    )
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=candidate.instruction),)),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics=CommandSemantics.LINUX_PROCESS,
            working_directory="/workspace",
        ),
        answer_type=AnswerType.JSON,
        answer_format=JsonValueAnswer(),
        grader=package.grader,
        source=Source(
            dataset="E-MTAB-10026-derived-tcr-counts",
            revision=source_revision,
            row=task_id,
            importer_revision=GENERATOR_VERSION,
        ),
        resources=ResourceGroups(worker=resources, verifier=package.resources),
        tags=("biology", "immune-repertoire", "objective-verifier", candidate.family),
    )


def generate(data: bytes, *, source_revision: str, count: int) -> list[TaskSpec]:
    """Generate the first ``count`` valid tasks under balanced family ordering."""
    rows = read_counts(data)
    available = candidates(rows)
    if count > len(available):
        raise ValueError(f"requested {count} tasks but this table supports only {len(available)} candidates")
    return [task_spec(candidate, data, source_revision) for candidate in available[:count]]


def write_release(output: Path, tasks: list[TaskSpec], data: bytes, source_revision: str) -> None:
    """Write TaskSpecs and a compact reproducibility manifest."""
    output.mkdir(parents=True, exist_ok=False)
    task_bytes = ("\n".join(task.model_dump_json() for task in tasks) + "\n").encode()
    (output / "tasks.jsonl").write_bytes(task_bytes)
    manifest = {
        "format": "covid-repertoire-objective-release-v1",
        "generator": GENERATOR_VERSION,
        "source_revision": source_revision,
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "tasks_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "count": len(tasks),
        "task_ids": [task.id for task in tasks],
        "release_status": "unreviewed_do_not_publish",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--source-revision", required=True, help="Immutable source/object digest or accession revision.")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    data = args.counts.read_bytes()
    tasks = generate(data, source_revision=args.source_revision, count=args.count)
    write_release(args.output, tasks, data, args.source_revision)
    print(f"wrote {args.count} tasks to {args.output}")


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate objective donor-paired IFN-beta response tasks from compact pseudobulk counts."""

import argparse
import csv
import hashlib
import io
import json
import math
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
    JsonValueAnswer,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource
from taskforge.sandbox.images import GRADER_BASE_IMAGE
from verifyit.spec import StructuredExactSpec

DATA_PATH = "workspace/data/pseudobulk_counts.csv"
DATA_DICTIONARY_PATH = "workspace/reference/data_dictionary.md"
BIOLOGY_CARD_PATH = "workspace/reference/ifnb_background.md"
GENERATOR_VERSION = "kang-ifnb-objective-v1"
REQUIRED_COLUMNS = (
    "donor_id",
    "condition",
    "cell_type",
    "n_cells",
    "total_counts",
    "gene",
    "gene_counts",
)
IFN_GENES = ("IFI6", "IFIT1", "IFIT2", "IFIT3", "IRF7", "ISG15", "MX1", "OAS1", "OAS2", "STAT1")
HOUSEKEEPING_GENES = ("ACTB", "GAPDH")
FOCAL_CELL_TYPES = ("CD14+ Monocytes", "FCGR3A+ Monocytes", "CD4 T cells", "CD8 T cells", "B cells", "NK cells")
MIN_PAIRED_DONORS = 6
CPM_PSEUDOCOUNT = 1.0

DATA_DICTIONARY = b"""# Data dictionary

Each row is one pseudonymous donor, condition, coarse cell type, and gene. `n_cells`
is the number of cells in that donor-condition-cell-type pseudobulk, `total_counts`
is the sum of counts over all measured genes, and `gene_counts` is the count for the
named gene. `ctrl` and `stim` are paired observations from the same donor before and
after six hours of IFN-beta stimulation. Donors, not cells, are biological
replicates. A missing donor-condition-cell-type row is missing data, not zero
expression.
"""

BIOLOGY_CARD = b"""# IFN-beta response background

Kang et al. profiled peripheral-blood mononuclear cells from eight donors before
and after six hours of IFN-beta stimulation. IFN-beta activates an antiviral
transcriptional response through interferon signaling. IFI6, IFIT1, IFIT2, IFIT3,
IRF7, ISG15, MX1, OAS1, OAS2, and STAT1 are used here as a predefined
interferon-response panel; ACTB and GAPDH are housekeeping comparators, not members
of that panel.

Transcript counts provide evidence for a transcriptional response, not direct
protein activity, antiviral protection, or clinical benefit. Cell types can differ
in response magnitude. Paired donor summaries preserve the biological replicate
and should be preferred over treating cells as independent observations. Pooled
counts can be reported descriptively but may be dominated by donors with more
recovered cells or reads.

Source context: Kang et al., Nature Biotechnology 36, 89-94 (2018), DOI
10.1038/nbt.4042, GEO GSE96583.
"""


@dataclass(frozen=True)
class CountRow:
    donor_id: str
    condition: str
    cell_type: str
    n_cells: int
    total_counts: int
    gene: str
    gene_counts: int


@dataclass(frozen=True)
class Candidate:
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


@dataclass(frozen=True)
class Pair:
    donor_id: str
    ctrl_counts: int
    ctrl_total: int
    stim_counts: int
    stim_total: int

    @property
    def ctrl_cpm(self) -> float:
        return self.ctrl_counts / self.ctrl_total * 1_000_000

    @property
    def stim_cpm(self) -> float:
        return self.stim_counts / self.stim_total * 1_000_000

    @property
    def log2fc(self) -> float:
        return math.log2((self.stim_cpm + CPM_PSEUDOCOUNT) / (self.ctrl_cpm + CPM_PSEUDOCOUNT))


def read_counts(data: bytes) -> list[CountRow]:
    """Parse and validate the sufficient-statistics table."""
    reader = csv.DictReader(io.StringIO(data.decode()))
    if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
        raise ValueError(f"expected columns {REQUIRED_COLUMNS}, got {reader.fieldnames}")
    rows = []
    keys = set()
    group_totals: dict[tuple[str, str, str], tuple[int, int]] = {}
    for raw in reader:
        row = CountRow(
            donor_id=raw["donor_id"],
            condition=raw["condition"],
            cell_type=raw["cell_type"],
            n_cells=int(raw["n_cells"]),
            total_counts=int(raw["total_counts"]),
            gene=raw["gene"],
            gene_counts=int(raw["gene_counts"]),
        )
        if row.condition not in {"ctrl", "stim"}:
            raise ValueError(f"unknown condition: {row.condition}")
        if not (0 <= row.gene_counts <= row.total_counts) or row.n_cells <= 0 or row.total_counts <= 0:
            raise ValueError(f"invalid counts for {row}")
        key = (row.donor_id, row.condition, row.cell_type, row.gene)
        if key in keys:
            raise ValueError(f"duplicate row: {key}")
        keys.add(key)
        group = key[:3]
        values = (row.n_cells, row.total_counts)
        if group in group_totals and group_totals[group] != values:
            raise ValueError(f"inconsistent group totals for {group}")
        group_totals[group] = values
        rows.append(row)
    if not rows:
        raise ValueError("count table is empty")
    return rows


def pairs(rows: list[CountRow], gene: str, cell_type: str) -> list[Pair]:
    selected = {(row.donor_id, row.condition): row for row in rows if row.gene == gene and row.cell_type == cell_type}
    donors = sorted({donor for donor, _ in selected})
    result = []
    for donor in donors:
        ctrl = selected.get((donor, "ctrl"))
        stim = selected.get((donor, "stim"))
        if ctrl is None or stim is None:
            continue
        result.append(Pair(donor, ctrl.gene_counts, ctrl.total_counts, stim.gene_counts, stim.total_counts))
    return result


def median_log2fc(rows: list[CountRow], gene: str, cell_type: str) -> tuple[int, float, int]:
    matched = pairs(rows, gene, cell_type)
    if len(matched) < MIN_PAIRED_DONORS:
        raise ValueError(f"{gene}/{cell_type} has only {len(matched)} paired donors")
    values = [pair.log2fc for pair in matched]
    return len(values), statistics.median(values), sum(value > 0 for value in values)


def paired_gene_candidates(rows: list[CountRow]) -> list[Candidate]:
    result = []
    for cell_type in FOCAL_CELL_TYPES:
        for gene in IFN_GENES:
            n_donors, median, positive = median_log2fc(rows, gene, cell_type)
            expected = {
                "gene": gene,
                "cell_type": cell_type,
                "n_paired_donors": n_donors,
                "median_log2fc": round(median, 4),
                "positive_donors": positive,
                "response_code": (
                    "consistent_upregulation"
                    if positive >= math.ceil(0.75 * n_donors) and median > 0
                    else "mixed_or_no_upregulation"
                ),
                "limitation_code": "transcript_response_not_protein_or_clinical_effect",
            }
            instruction = (
                f"Read {DATA_PATH} and {BIOLOGY_CARD_PATH}. For gene `{gene}` in `{cell_type}`, retain donors with "
                "both ctrl and stim rows. For each donor compute CPM = gene_counts / total_counts * 1,000,000 and "
                "log2FC = log2((stim_CPM + 1)/(ctrl_CPM + 1)). Report the ordinary median donor log2FC and number "
                "with log2FC > 0. Use `consistent_upregulation` exactly when at least 75% of paired donors are "
                "positive and the median is positive; otherwise use `mixed_or_no_upregulation`. Return exactly one "
                "JSON object with keys gene, cell_type, n_paired_donors, median_log2fc, positive_donors, "
                "response_code, and limitation_code. Round median_log2fc to four decimals and use limitation_code "
                "`transcript_response_not_protein_or_clinical_effect`."
            )
            result.append(
                Candidate("paired_gene_response", {"gene": gene, "cell_type": cell_type}, instruction, expected)
            )
    return result


def cell_type_comparison_candidates(rows: list[CountRow]) -> list[Candidate]:
    comparisons = (
        ("CD14+ Monocytes", "CD4 T cells"),
        ("CD14+ Monocytes", "B cells"),
        ("FCGR3A+ Monocytes", "NK cells"),
    )
    result = []
    for gene in ("ISG15", "IFIT1", "MX1", "OAS1"):
        for left, right in comparisons:
            left_n, left_median, _ = median_log2fc(rows, gene, left)
            right_n, right_median, _ = median_log2fc(rows, gene, right)
            delta = left_median - right_median
            expected = {
                "gene": gene,
                "cell_type_a": left,
                "cell_type_b": right,
                "n_paired_a": left_n,
                "n_paired_b": right_n,
                "median_log2fc_a": round(left_median, 4),
                "median_log2fc_b": round(right_median, 4),
                "delta_a_minus_b": round(delta, 4),
                "stronger_response": left if delta > 0 else right if delta < 0 else "tie",
                "limitation_code": "descriptive_cell_type_difference_not_mechanism",
            }
            instruction = (
                f"Read {DATA_PATH} and {BIOLOGY_CARD_PATH}. Compare the paired IFN-beta response for `{gene}` "
                f"between `{left}` (A) and `{right}` (B). Within each cell type retain donors with both conditions, "
                "calculate donor log2FC from CPM using a 1 CPM pseudocount, and take the ordinary median. Compute "
                "delta A minus B and identify the cell type with the larger median, or `tie` for exact equality. "
                "Return exactly one JSON object with keys gene, cell_type_a, cell_type_b, n_paired_a, n_paired_b, "
                "median_log2fc_a, median_log2fc_b, delta_a_minus_b, stronger_response, and limitation_code. Round "
                "effect values to four decimals and use limitation_code "
                "`descriptive_cell_type_difference_not_mechanism`."
            )
            result.append(
                Candidate(
                    "cell_type_response_comparison",
                    {"gene": gene, "a": left, "b": right},
                    instruction,
                    expected,
                )
            )
    return result


def gene_rank_candidates(rows: list[CountRow]) -> list[Candidate]:
    result = []
    for cell_type in FOCAL_CELL_TYPES:
        medians = {gene: median_log2fc(rows, gene, cell_type)[1] for gene in IFN_GENES}
        ranked = sorted(medians, key=lambda gene: (-medians[gene], gene))
        expected = {
            "cell_type": cell_type,
            "top_gene": ranked[0],
            "top_median_log2fc": round(medians[ranked[0]], 4),
            "bottom_gene": ranked[-1],
            "bottom_median_log2fc": round(medians[ranked[-1]], 4),
            "ranking_rule": "descending_median_then_gene_name",
        }
        instruction = (
            f"Read {DATA_PATH}. In `{cell_type}`, calculate paired donor CPM log2FC with a 1 CPM pseudocount for "
            f"every gene in the predefined IFN panel listed in {BIOLOGY_CARD_PATH}. Take each gene's ordinary median "
            "across paired donors, rank descending, and break exact ties by ascending gene name. Return exactly one "
            "JSON object with keys cell_type, top_gene, top_median_log2fc, bottom_gene, bottom_median_log2fc, and "
            "ranking_rule. Round medians to four decimals and set ranking_rule to "
            "`descending_median_then_gene_name`."
        )
        result.append(Candidate("ifn_panel_gene_rank", {"cell_type": cell_type}, instruction, expected))
    return result


def pooled_vs_paired_candidates(rows: list[CountRow]) -> list[Candidate]:
    result = []
    for cell_type in ("CD14+ Monocytes", "CD4 T cells", "B cells", "NK cells"):
        for gene in ("ISG15", "MX1"):
            matched = pairs(rows, gene, cell_type)
            n_donors, paired_median, _ = median_log2fc(rows, gene, cell_type)
            pooled_ctrl_counts = sum(pair.ctrl_counts for pair in matched)
            pooled_ctrl_total = sum(pair.ctrl_total for pair in matched)
            pooled_stim_counts = sum(pair.stim_counts for pair in matched)
            pooled_stim_total = sum(pair.stim_total for pair in matched)
            pooled_ctrl_cpm = pooled_ctrl_counts / pooled_ctrl_total * 1_000_000
            pooled_stim_cpm = pooled_stim_counts / pooled_stim_total * 1_000_000
            pooled = math.log2((pooled_stim_cpm + 1) / (pooled_ctrl_cpm + 1))
            expected = {
                "gene": gene,
                "cell_type": cell_type,
                "n_paired_donors": n_donors,
                "paired_median_log2fc": round(paired_median, 4),
                "pooled_log2fc": round(pooled, 4),
                "absolute_difference": round(abs(paired_median - pooled), 4),
                "preferred_biological_unit": "donor",
                "reason_code": "avoid_cell_and_read_depth_pseudoreplication",
            }
            instruction = (
                f"Read {DATA_PATH} and {DATA_DICTIONARY_PATH}. For `{gene}` in `{cell_type}`, use only donors "
                "represented in both conditions. Compute (1) each donor's CPM log2FC with a 1 CPM pseudocount and "
                "its ordinary median, and (2) a pooled log2FC after separately summing gene_counts and total_counts "
                "across those donors within ctrl and stim. Return exactly one JSON object with keys gene, cell_type, "
                "n_paired_donors, paired_median_log2fc, pooled_log2fc, absolute_difference, "
                "preferred_biological_unit, and reason_code. Round effects to four decimals, prefer `donor`, and use "
                "reason_code `avoid_cell_and_read_depth_pseudoreplication`."
            )
            result.append(Candidate("analysis_unit", {"gene": gene, "cell_type": cell_type}, instruction, expected))
    return result


def signature_candidates(rows: list[CountRow]) -> list[Candidate]:
    medians: dict[str, float] = {}
    donors: dict[str, int] = {}
    for cell_type in FOCAL_CELL_TYPES:
        by_donor: dict[str, list[float]] = {}
        for gene in IFN_GENES:
            for pair in pairs(rows, gene, cell_type):
                by_donor.setdefault(pair.donor_id, []).append(pair.log2fc)
        complete = [statistics.median(values) for values in by_donor.values() if len(values) == len(IFN_GENES)]
        if len(complete) < MIN_PAIRED_DONORS:
            continue
        medians[cell_type] = statistics.median(complete)
        donors[cell_type] = len(complete)
    ranked = sorted(medians, key=lambda cell_type: (-medians[cell_type], cell_type))
    expected = {
        "top_cell_type": ranked[0],
        "top_signature_median": round(medians[ranked[0]], 4),
        "bottom_cell_type": ranked[-1],
        "bottom_signature_median": round(medians[ranked[-1]], 4),
        "eligible_cell_types": len(ranked),
        "minimum_complete_donors": MIN_PAIRED_DONORS,
        "interpretation_code": "cell_type_specific_transcriptional_response",
        "limitation_code": "panel_score_not_direct_pathway_activity",
    }
    focal = ", ".join(FOCAL_CELL_TYPES)
    instruction = (
        f"Read {DATA_PATH} and {BIOLOGY_CARD_PATH}. For each focal cell type ({focal}), calculate paired donor log2FC "
        "for all ten IFN-panel genes using CPM and a 1 CPM pseudocount. For each donor take the median across all ten "
        f"genes; retain cell types with at least {MIN_PAIRED_DONORS} complete donors, then take the median donor "
        "signature. Rank descending with cell-type name as the tie-break. Return exactly one JSON object with keys "
        "top_cell_type, top_signature_median, bottom_cell_type, bottom_signature_median, eligible_cell_types, "
        "minimum_complete_donors, interpretation_code, and limitation_code. Round signature medians to four "
        "decimals. Use interpretation_code `cell_type_specific_transcriptional_response` and limitation_code "
        "`panel_score_not_direct_pathway_activity`."
    )
    return [
        Candidate(
            "ifn_signature_cell_type_rank",
            {"genes": IFN_GENES, "cell_types": FOCAL_CELL_TYPES},
            instruction,
            expected,
        )
    ]


def candidates(rows: list[CountRow]) -> list[Candidate]:
    """Enumerate the stable candidate portfolio before balanced selection."""
    return [
        *paired_gene_candidates(rows),
        *cell_type_comparison_candidates(rows),
        *gene_rank_candidates(rows),
        *pooled_vs_paired_candidates(rows),
        *signature_candidates(rows),
    ]


def select_candidates(portfolio: list[Candidate], count: int) -> list[Candidate]:
    """Interleave families so small releases exercise distinct reasoning contracts."""
    by_family: dict[str, list[Candidate]] = {}
    for candidate in portfolio:
        by_family.setdefault(candidate.family, []).append(candidate)
    for family_candidates in by_family.values():
        family_candidates.sort(key=lambda candidate: candidate.identity)
    selected = []
    while len(selected) < count and any(by_family.values()):
        for family in sorted(by_family):
            if by_family[family] and len(selected) < count:
                selected.append(by_family[family].pop(0))
    if len(selected) < count:
        raise ValueError(f"requested {count} tasks but this table supports only {len(selected)} candidates")
    return selected


def task_spec(candidate: Candidate, data: bytes, source_revision: str) -> TaskSpec:
    task_id = f"kang-ifnb-{candidate.family}-{candidate.identity}"
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=candidate.instruction),)),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image=GRADER_BASE_IMAGE,
            working_directory="/workspace",
        ),
        answer_type=AnswerType.JSON,
        answer_format=JsonValueAnswer(),
        grader=verifyit_package(StructuredExactSpec(expected=candidate.expected)).grader,
        source=Source(
            dataset="GSE96583-derived-pseudobulk-panel",
            revision=source_revision,
            row=task_id,
            importer_revision=GENERATOR_VERSION,
        ),
        resources=ResourceGroups(
            worker=(
                inline_resource(DATA_PATH, data),
                inline_resource(DATA_DICTIONARY_PATH, DATA_DICTIONARY),
                inline_resource(BIOLOGY_CARD_PATH, BIOLOGY_CARD),
            )
        ),
        tags=("biology", "single-cell", "ifn-beta", "objective-verifier", candidate.family),
    )


def generate(data: bytes, *, source_revision: str, count: int) -> list[TaskSpec]:
    selected = select_candidates(candidates(read_counts(data)), count)
    return [task_spec(candidate, data, source_revision) for candidate in selected]


def write_release(output: Path, tasks: list[TaskSpec], data: bytes, source_revision: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    task_bytes = ("\n".join(task.model_dump_json() for task in tasks) + "\n").encode()
    (output / "tasks.jsonl").write_bytes(task_bytes)
    manifest = {
        "format": "kang-ifnb-objective-tasks-v1",
        "generator_version": GENERATOR_VERSION,
        "status": "unreviewed_do_not_publish",
        "source_revision": source_revision,
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "tasks_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "count": len(tasks),
        "task_ids": [task.id for task in tasks],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    data = args.counts.read_bytes()
    tasks = generate(data, source_revision=args.source_revision, count=args.count)
    write_release(args.output, tasks, data, args.source_revision)
    print(f"wrote {len(tasks)} tasks to {args.output}")


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-check WP9's weight metrics against #9920's published results.

WP9 re-implements the #9920/#9984 weight-delta method from the issue text; the
original scripts are lost, but the per-tensor results survive on S3
(``s3://marin-us-east-02a/marin/scratch/karan/mopd2/review/wdelta-<model>.json``,
plus ``wdelta-all-summaries.json``). This script recomputes every published
quantity from the checkpoints — per-tensor ``b2``/``d2``/``changed``/``dmax``,
per-component ``rel_frobenius`` and ``frac_changed``, the cosine and sign
agreement against the expert delta E, and the subspace overlap with both nulls —
and writes one CSV row per compared number.

Control C8 is NOT applied here: the #9920 checkpoints are bf16 exports, so
:func:`paired_blocks` (the production, C8-checked entry) would refuse them by
design. This script reads both sides with :class:`SafetensorsCheckpoint`, widens
to fp32 itself, builds blocks the way :func:`checkpoint_blocks` does, and calls
the pure metric functions directly. Production weight metrics still go through
``paired_blocks``.

Two grouping differences from WP9's :class:`Component` split, per the #9920
comment: ``self_attn.attn_gate.weight`` and ``attn_gated_norm`` count under
attention, and ``mlp.router.bias`` counts under router rather than as a separate
selection bias. Every number is reported under both groupings — WP9's own and
#9920's (:func:`group_9920`); comparisons against the published results use the
#9920 grouping.

The #9920 comment does not say how per-matrix overlaps pool into one component
number, nor whether a fused expert tensor counted as one matrix or one per
expert, so the overlap is aggregated under four conventions:

- ``weighted``: energy-weighted, Σ projected / Σ total over matrices (WP9's
  current aggregate);
- ``mean``: an unweighted mean of per-matrix shares;
- ``stacked``: the fused ``[E, out, in]`` expert tensor counts as one
  ``[E·out, in]`` matrix instead of one per expert, under either pooling.

The four overlap conventions are compared against the published multiples in the
run report; the surviving convention is recorded in this docstring once known.

Run it where the checkpoints live (they are read one tensor at a time through
byte ranges, so a 134 GB checkpoint is never staged):

    python -m experiments.post_training.weight_analysis.reproduce_9920 \\
        --base s3://marin-us-east-02a/.../models/b-step92 \\
        --model E=s3://marin-us-east-02a/.../cruxeval-step92-lr1e-05/2026.10.07.5/hf/step-272 \\
        --model opd75=s3://marin-us-east-02a/.../exports/global_step_75/policy \\
        --csv /tmp/opd9920/comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum

import fsspec
import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.weight_analysis.checkpoint_io import SafetensorsCheckpoint
from experiments.post_training.weight_analysis.components import Block, Component, component_of
from experiments.post_training.weight_analysis.delta_stats import (
    DeltaComparison,
    DeltaSummary,
    MatrixOverlap,
    PairGeometry,
    ReferenceBases,
    block_comparison,
    block_summary,
    matrix_generator,
    overlap_against,
    reference_bases,
)

logger = logging.getLogger(__name__)

RESULTS_PREFIX = "s3://marin-us-east-02a/marin/scratch/karan/mopd2/review"

REFERENCE_LABEL = "E"

# Relative tolerance on b2/d2 and the derived per-component shares and cosines
# (plan steps 2 and 3). #9920 accumulated its sums in fp32, ours in float64: against
# E, ``changed`` and ``dmax`` match exactly on all 502 tensors (the weights are
# identical) while the sums differ by a median 1e-5, growing with tensor size to
# 7e-4 on lm_head. A GPU fp32 reduction order cannot be replayed, so the sums are
# held to fp32 accumulation accuracy.
SUM_TOLERANCE = 1e-3
# The published overlap, null and sign-agreement rows are read off a comment
# table at 2-3 significant figures; that is the agreement they can support.
PUBLISHED_TOLERANCE = 0.05

# #9920's per-tensor groups. Checked against the ``group`` field of every tensor
# in the result JSON at load time, so a misnamed pattern cannot pass silently.
GROUP_9920_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (group, re.compile(pattern))
    for group, pattern in (
        ("lm_head", r"lm_head\.weight"),
        ("embeddings", r"model\.(embed_tokens\.weight|embed_norm\.weight|embed_gated_norm\.(up|down)_proj\.weight)"),
        (
            "norms",
            r"(model\.norm\.weight|model\.final_gated_norm\.(up|down)_proj\.weight"
            r"|model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight"
            r"|model\.layers\.\d+\.mlp_gated_norm\.(up|down)_proj\.weight)",
        ),
        (
            "attention",
            r"model\.layers\.\d+\.(self_attn\.[qkvo]_proj\.weight|self_attn\.attn_gate\.weight"
            r"|attn_gated_norm\.(up|down)_proj\.weight)",
        ),
        ("expert_mlp", r"model\.layers\.\d+\.mlp\.experts\.(gate|up|down)_proj\.weight"),
        ("router", r"model\.layers\.\d+\.mlp\.router\.(weight|bias)"),
        ("shared_expert", r"model\.layers\.\d+\.shared_expert\.(gate|up|down)_proj\.weight"),
    )
)

# The published rows this script reproduces (#9920 comment 6300394038): the
# all-weights overlap with its random null, the all-weights sign agreement with
# E, and the per-component overlap multiples over the random null.
PUBLISHED_OVERLAP: dict[str, tuple[float, float]] = {
    "opd75": (0.203, 0.101),
    "rs282": (0.196, 0.108),
    "plain399": (0.184, 0.108),
}
PUBLISHED_SIGN_AGREEMENT: dict[str, float] = {"opd75": 0.598, "rs282": 0.581, "plain399": 0.562}
PUBLISHED_OVERLAP_MULTIPLE: dict[str, dict[str, float]] = {
    "attention": {"opd75": 3.9, "rs282": 3.0, "plain399": 2.7},
    "expert_mlp": {"opd75": 1.9, "rs282": 1.7, "plain399": 1.6},
    "shared_expert": {"opd75": 4.0, "rs282": 2.6, "plain399": 2.2},
    "router": {"opd75": 3.7, "rs282": 2.2, "plain399": 2.1},
    "lm_head": {"opd75": 391.0, "rs282": 710.0, "plain399": 686.0},
}

# The four overlap conventions, as (matrix kind, pooling) pairs.
OVERLAP_CONVENTIONS: tuple[tuple[str, str], ...] = (
    ("matrix", "weighted"),
    ("matrix", "mean"),
    ("stacked", "weighted"),
    ("stacked", "mean"),
)


def group_9920(name: str) -> str:
    """The #9920 group of a Grug tensor name; raises on a name no group claims."""
    for group, pattern in GROUP_9920_PATTERNS:
        if pattern.fullmatch(name):
            return group
    raise ValueError(f"{name!r} matches no #9920 group; extend GROUP_9920_PATTERNS")


def tolerance_of(metric: str) -> float:
    """The comparison tolerance a metric's row is judged under.

    ``changed`` and ``dmax`` must match exactly (both sides are exact functions
    of the bf16 values); sums and shares carry ``SUM_TOLERANCE``; rows against
    the published comment table carry ``PUBLISHED_TOLERANCE``.
    """
    if metric in ("changed", "dmax"):
        return 0.0
    if metric.startswith("published"):
        return PUBLISHED_TOLERANCE
    return SUM_TOLERANCE


@dataclass(frozen=True)
class ComparisonRow:
    """One compared number: a CSV row."""

    metric: str
    component: str
    model: str
    ours: float
    theirs: float

    @property
    def relative_error(self) -> float:
        """``|ours - theirs| / |theirs|``; 0 when both sides are 0."""
        if self.theirs == 0.0:
            return 0.0 if self.ours == 0.0 else math.inf
        return abs(self.ours - self.theirs) / abs(self.theirs)

    @property
    def within(self) -> bool:
        return self.relative_error <= tolerance_of(self.metric)


@dataclass
class TheirResults:
    """The #9920 result JSON for one model."""

    label: str
    model_path: str
    per_tensor: dict[str, dict]
    summary: dict[str, dict]
    cos_vs_ref0: dict[str, float]

    @classmethod
    def load(cls, label: str, results_prefix: str, model_path: str) -> TheirResults:
        filesystem, path = fsspec.core.url_to_fs(results_prefix)
        with filesystem.open(f"{path.rstrip('/')}/wdelta-{label}.json") as handle:
            document = json.load(handle)
        if document["model"] != model_path:
            # The recorded prefixes have been deleted; the runs read staged copies of
            # the same checkpoints. The per-tensor comparison below is the identity
            # check — wrong weights fail it loudly — so a path difference only logs.
            logger.warning(
                "wdelta-%s.json recorded %s; scoring the staged copy %s",
                label,
                document["model"],
                model_path,
            )
        mismatched = [
            (name, group_9920(name), entry["group"])
            for name, entry in document["per_tensor"].items()
            if group_9920(name) != entry["group"]
        ]
        if mismatched:
            raise ValueError(f"the #9920 grouping disagrees with the result JSON, e.g. {mismatched[:5]}")
        cos_vs_ref0: dict[str, float] = {}
        with filesystem.open(f"{path.rstrip('/')}/wdelta-all-summaries.json") as handle:
            all_summaries = json.load(handle)
        for component, entry in all_summaries.get(label, {}).get("summary", {}).items():
            if "cos_vs_ref0" in entry:
                cos_vs_ref0[component] = float(entry["cos_vs_ref0"])
        return cls(
            label=label,
            model_path=model_path,
            per_tensor=document["per_tensor"],
            summary=document["summary"],
            cos_vs_ref0=cos_vs_ref0,
        )


@dataclass
class OverlapShares:
    """Per-matrix overlap shares of one key, kept raw for both pooling conventions."""

    projected: float = 0.0
    total: float = 0.0
    random: float = 0.0
    permuted: float = 0.0
    matrices: int = 0
    shares: list[float] = field(default_factory=list)
    random_shares: list[float] = field(default_factory=list)

    def add(self, overlap: MatrixOverlap) -> None:
        if overlap.zero_reference:
            # The reference did not move, so it has no subspace; its delta energy
            # stays out of every denominator, as in WP9's aggregate.
            return
        self.projected += overlap.projected
        self.total += overlap.total
        self.random += overlap.random
        self.permuted += overlap.permuted
        self.matrices += 1
        if overlap.total > 0.0:
            self.shares.append(overlap.projected / overlap.total)
            self.random_shares.append(overlap.random / overlap.total)

    def pooled(self, pooling: str) -> tuple[float, float]:
        """(overlap, random null) under ``pooling``, as shares of delta energy."""
        if pooling == "weighted":
            if self.total == 0.0:
                raise ValueError("no delta energy in any matrix for this key; the weighted overlap is undefined")
            return self.projected / self.total, self.random / self.total
        if pooling == "mean":
            if not self.shares:
                raise ValueError("no matrix with delta energy for this key; the mean share is undefined")
            return math.fsum(self.shares) / len(self.shares), math.fsum(self.random_shares) / len(self.random_shares)
        raise ValueError(f"unknown pooling {pooling!r}")


def _new_comparison() -> DeltaComparison:
    return DeltaComparison(PairGeometry())


@dataclass
class ModelReport:
    """Everything accumulated for one model against B."""

    summary: DeltaSummary = field(default_factory=DeltaSummary)
    groups: dict[str, DeltaSummary] = field(default_factory=dict)
    per_tensor: dict[str, DeltaSummary] = field(default_factory=dict)
    max_abs_delta: dict[str, float] = field(default_factory=dict)
    comparison: DeltaComparison = field(default_factory=_new_comparison)
    group_comparisons: dict[str, DeltaComparison] = field(default_factory=dict)
    # (matrix kind, group) -> shares; kind is "matrix" (one matrix per plain 2-D
    # tensor and per expert slice) or "stacked" (fused expert tensors as one
    # matrix each).
    overlaps: dict[tuple[str, str], OverlapShares] = field(default_factory=dict)

    def group_summary(self, group: str) -> DeltaSummary:
        return self.groups.setdefault(group, DeltaSummary())

    def group_comparison(self, group: str) -> DeltaComparison:
        return self.group_comparisons.setdefault(group, DeltaComparison(PairGeometry()))

    def overlap_shares(self, kind: str, group: str) -> OverlapShares:
        return self.overlaps.setdefault((kind, group), OverlapShares())


def tensor_blocks(array: np.ndarray, name: str, component: Component) -> list[tuple[int | None, np.ndarray]]:
    """(expert key, slice) pairs of one tensor, exactly :func:`checkpoint_blocks`' split."""
    if component is Component.ROUTED_EXPERTS:
        if array.ndim != 3:
            raise ValueError(f"routed-expert tensor {name!r} has shape {array.shape}; expected [E, out, in]")
        return [(expert, array[expert]) for expert in range(array.shape[0])]
    return [(None, array)]


class BasisSide(StrEnum):
    """Which singular directions span a matrix's subspace.

    ``RIGHT`` uses the input-side (column) directions of every matrix, WP9's
    definition. ``MIN`` uses the smaller side's directions, transposing wide
    matrices; for a tall matrix the two coincide.
    """

    RIGHT = "right"
    MIN = "min"


def oriented(matrix: np.ndarray, side: BasisSide) -> np.ndarray:
    """``matrix`` laid out so its subspace is spanned by right singular directions."""
    if side is BasisSide.MIN and matrix.shape[0] < matrix.shape[1]:
        return matrix.T
    return matrix


@dataclass(frozen=True)
class TensorBases:
    """One tensor's reference bases: one per block (expert slice or whole matrix), plus the stacked matrix's."""

    blocks: list[ReferenceBases | None]
    stacked: ReferenceBases | None


def tensor_reference_bases(
    name: str,
    component: Component,
    base_array: np.ndarray,
    reference_array: np.ndarray,
    *,
    seed: int,
    side: BasisSide,
    pool: ThreadPoolExecutor,
) -> TensorBases | None:
    """The reference delta's bases for every matrix of one tensor, computed once for all compared models.

    Each matrix draws its nulls from its own keyed generator, so the matrices are
    computed concurrently in ``pool`` (eigendecompositions release the GIL) with
    results that do not depend on scheduling. ``None`` for 1-D tensors, which
    carry no subspace.
    """
    if reference_array.ndim < 2:
        return None
    pairs = list(
        zip(
            tensor_blocks(base_array, name, component),
            tensor_blocks(reference_array, name, component),
            strict=True,
        )
    )
    stacked = None
    if component is Component.ROUTED_EXPERTS:
        stacked_reference = (reference_array - base_array).reshape(-1, reference_array.shape[-1])
        stacked = pool.submit(
            reference_bases,
            oriented(stacked_reference, side),
            generator=matrix_generator(seed, f"{name}:stacked", None),
        )
    blocks = [
        pool.submit(
            reference_bases,
            oriented(reference_slice - base_slice, side),
            generator=matrix_generator(seed, name, expert),
        )
        for (expert, base_slice), (_expert, reference_slice) in pairs
    ]
    return TensorBases(
        blocks=[future.result() for future in blocks], stacked=stacked.result() if stacked is not None else None
    )


def fold_tensor(
    report: ModelReport,
    *,
    name: str,
    component: Component,
    group: str,
    base_array: np.ndarray,
    model_array: np.ndarray,
    reference_array: np.ndarray | None,
    bases: TensorBases | None,
    side: BasisSide,
) -> tuple[DeltaSummary, float]:
    """Fold one tensor of one model into ``report``.

    Returns the tensor's own summary and max absolute delta, which feed the
    per-tensor check against the result JSON. With ``reference_array`` set, the
    tensor also folds into the model's E-comparison and — for plain 2-D tensors
    and per-expert expert slices — its overlap record against ``bases``, the
    reference's precomputed bases. The stacked convention adds one more matrix per
    fused expert tensor, after its experts.
    """
    blocks = tensor_blocks(model_array, name, component)
    base_blocks = tensor_blocks(base_array, name, component)
    reference_blocks = tensor_blocks(reference_array, name, component) if reference_array is not None else None

    tensor_summary = DeltaSummary()
    tensor_max = 0.0
    for index, (expert, model_slice) in enumerate(blocks):
        base_slice = base_blocks[index][1]
        block = block_summary(Block(name, component, expert, base_slice), Block(name, component, expert, model_slice))
        tensor_summary.merge(block)
        report.summary.merge(block)
        report.group_summary(group).merge(block)

        delta = model_slice - base_slice
        tensor_max = max(tensor_max, float(np.max(np.abs(delta))))

        if reference_blocks is None:
            continue
        reference_delta = reference_blocks[index][1] - base_slice
        comparison = block_comparison(
            Block(name, component, expert, delta), Block(name, component, expert, reference_delta)
        )
        report.comparison.merge(comparison)
        report.group_comparison(group).merge(comparison)
        if delta.ndim == 2:
            record = overlap_against(oriented(delta, side), bases.blocks[index])
            # A plain 2-D tensor is one matrix under both conventions; only fused
            # expert tensors differ (per-expert slices here, one stacked matrix
            # below).
            kinds = ("matrix", "stacked") if component is not Component.ROUTED_EXPERTS else ("matrix",)
            for kind in kinds:
                report.overlap_shares(kind, group).add(record)
                report.overlap_shares(kind, "ALL").add(record)

    if reference_blocks is not None and component is Component.ROUTED_EXPERTS:
        stacked_delta = (model_array - base_array).reshape(-1, model_array.shape[-1])
        record = overlap_against(oriented(stacked_delta, side), bases.stacked)
        report.overlap_shares("stacked", group).add(record)
        report.overlap_shares("stacked", "ALL").add(record)
    return tensor_summary, tensor_max


def check_per_tensor(
    label: str,
    name: str,
    tensor_summary: DeltaSummary,
    tensor_max: float,
    theirs: TheirResults,
) -> list[ComparisonRow]:
    """Compare one tensor's recomputed metrics with the result JSON's entry."""
    theirs_entry = theirs.per_tensor[name]
    return [
        ComparisonRow("b2", name, label, tensor_summary.base_sq, float(theirs_entry["b2"])),
        ComparisonRow("d2", name, label, tensor_summary.delta_sq, float(theirs_entry["d2"])),
        ComparisonRow("changed", name, label, float(tensor_summary.changed), float(theirs_entry["changed"])),
        ComparisonRow("dmax", name, label, tensor_max, float(theirs_entry["dmax"])),
    ]


def parse_model_specs(specs: list[str]) -> dict[str, str]:
    """``label=path`` strings to an ordered label → path mapping."""
    models: dict[str, str] = {}
    for spec in specs:
        label, separator, path = spec.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"--model must be label=path, got {spec!r}")
        if label in models:
            raise ValueError(f"model label {label!r} given twice")
        models[label] = path
    return models


def run_pass(
    base_path: str,
    models: dict[str, str],
    reference_label: str,
    their: dict[str, TheirResults],
    *,
    tensor_filter: set[str] | None,
    full_stage: bool,
    seed: int,
    workers: int,
    side: BasisSide = BasisSide.RIGHT,
) -> tuple[list[ComparisonRow], list[str], dict[str, ModelReport]]:
    """Stream every checkpoint once and compare per-tensor metrics with the JSONs.

    Returns the comparison rows, the per-tensor failure messages, and the
    per-model reports. A non-empty failure list means a name, upcast or slicing
    bug, and the caller stops before the summary and overlap stages.
    """
    base_checkpoint = SafetensorsCheckpoint(base_path)
    checkpoints = {label: SafetensorsCheckpoint(path) for label, path in models.items()}
    for label, checkpoint in checkpoints.items():
        if checkpoint.names() != base_checkpoint.names():
            raise ValueError(f"{label}'s tensor list differs from B's")
    for label, results in their.items():
        if set(results.per_tensor) != set(base_checkpoint.names()):
            raise ValueError(f"the {label} result JSON and B cover different tensors")

    names = base_checkpoint.names()
    if tensor_filter is not None:
        unknown = tensor_filter - set(names)
        if unknown:
            raise ValueError(f"--tensors names no tensor of B: {sorted(unknown)[:5]}")
        names = [name for name in names if name in tensor_filter]

    reports = {label: ModelReport() for label in models}
    rows: list[ComparisonRow] = []
    failures: list[str] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, name in enumerate(names, start=1):
            component = component_of(name)
            group = group_9920(name)
            base_array = base_checkpoint.read(name).astype(np.float32)
            arrays = {label: checkpoints[label].read(name).astype(np.float32) for label in models}
            bases = (
                tensor_reference_bases(
                    name, component, base_array, arrays[reference_label], seed=seed, side=side, pool=pool
                )
                if full_stage
                else None
            )
            for label in models:
                reference_array = arrays[reference_label] if full_stage and label != reference_label else None
                tensor_summary, tensor_max = fold_tensor(
                    reports[label],
                    name=name,
                    component=component,
                    group=group,
                    base_array=base_array,
                    model_array=arrays[label],
                    reference_array=reference_array,
                    bases=bases if reference_array is not None else None,
                    side=side,
                )
                reports[label].per_tensor[name] = tensor_summary
                reports[label].max_abs_delta[name] = tensor_max
                tensor_rows = check_per_tensor(label, name, tensor_summary, tensor_max, their[label])
                rows.extend(tensor_rows)
                failures.extend(
                    f"{row.model} {row.component} {row.metric}: ours {row.ours!r} theirs {row.theirs!r} "
                    f"rel err {row.relative_error:.3e}"
                    for row in tensor_rows
                    if not row.within
                )
            if index % 25 == 0 or index == len(names):
                logger.info("tensor %d/%d %s (%.0f s elapsed)", index, len(names), name, time.time() - started)
    return rows, failures, reports


def summary_rows(
    models: dict[str, str],
    reference_label: str,
    their: dict[str, TheirResults],
    reports: dict[str, ModelReport],
) -> tuple[list[ComparisonRow], list[str]]:
    """Per-component share and cosine rows, then the overlap and sign rows.

    Share and cosine rows count as pass/fail; the published-table rows never do
    (they only say which overlap convention reproduces #9920).
    """
    rows: list[ComparisonRow] = []
    failures: list[str] = []

    def judge(new_rows: list[ComparisonRow], *, counts: bool) -> None:
        rows.extend(new_rows)
        if not counts:
            return
        failures.extend(
            f"{row.model} {row.component} {row.metric}: ours {row.ours!r} theirs {row.theirs!r} "
            f"rel err {row.relative_error:.3e}"
            for row in new_rows
            if not row.within
        )

    for label in models:
        report = reports[label]
        theirs = their[label]
        if report.summary.elements != int(theirs.summary["ALL"]["n"]):
            raise ValueError(
                f"{label}: {report.summary.elements} elements against the result JSON's "
                f"{int(theirs.summary['ALL']['n'])}"
            )
        new_rows: list[ComparisonRow] = []
        for group in ("ALL", *(name for name in theirs.summary if name != "ALL")):
            summary = report.summary if group == "ALL" else report.groups.get(group)
            if summary is None:
                raise ValueError(f"{label}: no accumulated summary for group {group!r}")
            new_rows.append(
                ComparisonRow(
                    "rel_frobenius", group, label, summary.relative_norm, float(theirs.summary[group]["rel_frobenius"])
                )
            )
            new_rows.append(
                ComparisonRow(
                    "frac_changed",
                    group,
                    label,
                    summary.changed_fraction,
                    float(theirs.summary[group]["frac_changed"]),
                )
            )
            if label != reference_label and group in theirs.cos_vs_ref0:
                comparison = report.comparison if group == "ALL" else report.group_comparisons[group]
                new_rows.append(ComparisonRow("cos_vs_ref0", group, label, comparison.cosine, theirs.cos_vs_ref0[group]))
        judge(new_rows, counts=True)

    for label in models:
        if label == reference_label or label not in PUBLISHED_OVERLAP:
            # The comment publishes overlap and sign-agreement rows only for the
            # three transfer checkpoints.
            continue
        report = reports[label]
        published: list[ComparisonRow] = [
            ComparisonRow(
                "published_sign_agreement",
                "ALL",
                label,
                report.comparison.sign_agreement,
                PUBLISHED_SIGN_AGREEMENT[label],
            )
        ]
        for kind, pooling in OVERLAP_CONVENTIONS:
            tag = "published_overlap" if kind == "matrix" else "published_overlap_stacked"
            overlap, null = report.overlap_shares(kind, "ALL").pooled(pooling)
            published.append(ComparisonRow(tag, "ALL", label, overlap, PUBLISHED_OVERLAP[label][0]))
            published.append(ComparisonRow(f"{tag}_random", "ALL", label, null, PUBLISHED_OVERLAP[label][1]))
            for component, table in PUBLISHED_OVERLAP_MULTIPLE.items():
                component_overlap, component_null = report.overlap_shares(kind, component).pooled(pooling)
                published.append(
                    ComparisonRow(f"{tag}_multiple", component, label, component_overlap / component_null, table[label])
                )
        judge(published, counts=False)
    return rows, failures


def write_csv(rows: list[ComparisonRow], path: str) -> None:
    """Every compared number, as the plan's six columns."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "component", "model", "ours", "theirs", "relative_error"))
        for row in rows:
            writer.writerow(
                (row.metric, row.component, row.model, repr(row.ours), repr(row.theirs), repr(row.relative_error))
            )


def print_overlap_report(reports: dict[str, ModelReport], reference_label: str) -> None:
    """The four conventions beside the published numbers, for the run log."""
    for label, report in sorted(reports.items()):
        if label == reference_label:
            continue
        print(f"\n{label}: overlap multiples over the random null, published value in parens")
        print(f"{'component':>15}" + "".join(f"  {kind}/{pooling:>16}" for kind, pooling in OVERLAP_CONVENTIONS))
        for component, table in PUBLISHED_OVERLAP_MULTIPLE.items():
            if ("matrix", component) not in report.overlaps:
                continue  # a --tensors subset scored no matrix of this component
            cells = []
            for kind, pooling in OVERLAP_CONVENTIONS:
                overlap, null = report.overlap_shares(kind, component).pooled(pooling)
                cells.append(f"{overlap / null:9.3f} ({table[label]:6.1f})")
            print(f"{component:>15}" + "".join(f"  {cell:>18}" for cell in cells))
        overlap, null = report.overlap_shares("matrix", "ALL").pooled("weighted")
        print(
            f"{'ALL':>15}  overlap {overlap:.3f} (published {PUBLISHED_OVERLAP[label][0]:.3f}), "
            f"random {null:.3f} (published {PUBLISHED_OVERLAP[label][1]:.3f})"
        )
        print(
            f"{'sign agree':>15}  {report.comparison.sign_agreement:.3f}"
            f" (published {PUBLISHED_SIGN_AGREEMENT[label]:.3f})"
        )


def main() -> None:
    configure_coreweave_s3()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="path to the B checkpoint (bf16 export)")
    parser.add_argument("--model", action="append", required=True, help="label=path, repeated; E must appear")
    parser.add_argument("--reference", default=REFERENCE_LABEL, help="which label is the expert delta E")
    parser.add_argument("--results-prefix", default=RESULTS_PREFIX, help="directory holding the wdelta-*.json results")
    parser.add_argument("--tensors", action="append", default=None, help="restrict to these tensor names")
    parser.add_argument("--stage", choices=("per-tensor", "full"), default="full")
    parser.add_argument("--seed", type=int, default=0, help="seed of the overlap nulls")
    parser.add_argument("--workers", type=int, default=8, help="threads computing one tensor's reference bases")
    parser.add_argument(
        "--basis-side",
        choices=[side.value for side in BasisSide],
        default=BasisSide.RIGHT.value,
        help="singular directions spanning each matrix's subspace",
    )
    parser.add_argument("--csv", default=None, help="where to write the comparison CSV")
    args = parser.parse_args()

    models = parse_model_specs(args.model)
    if args.reference not in models:
        raise ValueError(f"--reference {args.reference!r} is not among --model labels {sorted(models)}")
    their = {label: TheirResults.load(label, args.results_prefix, path) for label, path in models.items()}

    rows, failures, reports = run_pass(
        args.base,
        models,
        args.reference,
        their,
        tensor_filter=set(args.tensors) if args.tensors else None,
        full_stage=args.stage == "full",
        seed=args.seed,
        workers=args.workers,
        side=BasisSide(args.basis_side),
    )
    if failures:
        if args.csv:
            write_csv(rows, args.csv)
        raise SystemExit(
            f"{len(failures)} per-tensor mismatches against the #9920 results; a name, upcast or slicing "
            "bug — fix before the summary check. First failures:\n" + "\n".join(failures[:20])
        )
    if args.stage == "per-tensor":
        if args.csv:
            write_csv(rows, args.csv)
        print(f"per-tensor check passed for {len(rows) // 4} tensors x {len(models)} models")
        return

    if args.tensors:
        # The summaries compare whole-model totals, which a tensor subset cannot match;
        # the overlap multiples are still per component.
        print_overlap_report(reports, args.reference)
        if args.csv:
            write_csv(rows, args.csv)
        return
    summary, summary_failures = summary_rows(models, args.reference, their, reports)
    rows.extend(summary)
    print_overlap_report(reports, args.reference)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"wrote {len(rows)} comparison rows to {args.csv}")
    if summary_failures:
        raise SystemExit(
            f"{len(summary_failures)} summary mismatches against the #9920 results:\n" + "\n".join(summary_failures[:20])
        )


if __name__ == "__main__":
    main()

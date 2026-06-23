# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build roofline rows and totals from formulas, hardware, and observations."""

from __future__ import annotations

from collections import defaultdict

from marin.tools.roofline.attribution import AttributionRule, attribute_name, suggested_regex
from marin.tools.roofline.formulas import FormulaEstimate, formula_estimates
from marin.tools.roofline.hardware import Hardware
from marin.tools.roofline.model_spec import ModelSpec
from marin.tools.roofline.profile_ingest import ObservedProfileRow
from marin.tools.roofline.types import ObservedTimeBasis, RooflineRow, RooflineTotals, RowKind

SECONDS_PER_MICROSECOND = 1e-6
BITS_PER_BYTE = 8.0


def build_rows(
    model: ModelSpec,
    hardware: Hardware,
    observations: list[ObservedProfileRow],
    rules: list[AttributionRule],
) -> tuple[list[RooflineRow], list[dict[str, object]]]:
    formulas_by_op = {estimate.semantic_op: estimate for estimate in formula_estimates(model)}
    observed_by_op: dict[str, list[ObservedProfileRow]] = defaultdict(list)
    uncategorized_observations = []
    unattributed: list[dict[str, object]] = []
    for observation in observations:
        semantic_op = attribute_name(observation.match_text, rules)
        if semantic_op is None:
            uncategorized_observations.append(observation)
            unattributed.append(
                {
                    "name": observation.name,
                    "kernel_name": observation.kernel_name,
                    "duration": observation.total_time,
                    "basis": observation.basis,
                    "suggested_regex": suggested_regex(observation.match_text),
                }
            )
            continue
        observed_by_op[semantic_op].append(observation)

    rows = []
    for estimate in formulas_by_op.values():
        rows.append(_row_from_estimate(estimate, hardware, observed_by_op.get(estimate.semantic_op, [])))

    for semantic_op, semantic_observations in observed_by_op.items():
        if semantic_op in formulas_by_op:
            continue
        rows.append(_row_from_observations(semantic_op, semantic_observations))
    if uncategorized_observations:
        rows.append(_row_from_observations("uncategorized", uncategorized_observations))

    return (
        sorted(rows, key=lambda row: (row.kind.value, row.semantic_op)),
        sorted(
            unattributed,
            key=lambda row: -_unattributed_duration(row),
        )[:25],
    )


def build_totals(rows: list[RooflineRow]) -> RooflineTotals:
    compute_roofline = sum(row.ideal_time for row in rows if row.kind == RowKind.COMPUTE)
    comm_roofline = sum(row.ideal_time for row in rows if row.kind == RowKind.COMM)
    track_summed = sum(row.track_summed_observed_time or 0.0 for row in rows)
    exposed_values = [row.critical_path_observed_time for row in rows if row.critical_path_observed_time is not None]
    estimated = sum(row.estimated_time for row in rows)
    return RooflineTotals(
        compute_roofline_time=compute_roofline,
        comm_roofline_time=comm_roofline,
        observed_track_summed_time=track_summed if track_summed > 0 else None,
        observed_exposed_time=sum(exposed_values) if exposed_values else None,
        estimated_scenario_time=estimated,
    )


def _row_from_estimate(
    estimate: FormulaEstimate, hardware: Hardware, observations: list[ObservedProfileRow]
) -> RooflineRow:
    ideal_time = _ideal_time(estimate, hardware)
    observed_total = sum(observation.total_time for observation in observations) if observations else None
    observed_count = sum(observation.count or 0 for observation in observations) if observations else None
    observed_avg = _observed_avg(observations)
    basis = _observed_basis(observations)
    track_summed = observed_total * SECONDS_PER_MICROSECOND if observed_total is not None else None
    efficiency = _default_efficiency(estimate.semantic_op, estimate.kind, hardware)
    return RooflineRow(
        semantic_op=estimate.semantic_op,
        display_name=estimate.semantic_op.replace("_", " "),
        kind=estimate.kind,
        estimated_flops=estimate.flops,
        estimated_bytes=estimate.bytes_accessed,
        ideal_time=ideal_time,
        user_efficiency=efficiency,
        estimated_time=ideal_time / efficiency if efficiency > 0 else ideal_time,
        profile_observed_time=track_summed,
        track_summed_observed_time=track_summed,
        observed_time_basis=basis,
        profile_achieved_pct=None,
        observed_count=observed_count,
        observed_avg_time=observed_avg * SECONDS_PER_MICROSECOND if observed_avg is not None else None,
        formula=estimate.formula,
        matched_name=observations[0].name if observations else None,
        observed_comparable_to_model=estimate.semantic_op != "expert_all_to_all",
        source="model_formula+xprof" if observations else "model_formula",
    )


def _row_from_observations(semantic_op: str, observations: list[ObservedProfileRow]) -> RooflineRow:
    observed_total = sum(observation.total_time for observation in observations)
    observed_count = sum(observation.count or 0 for observation in observations)
    observed_avg = _observed_avg(observations)
    return RooflineRow(
        semantic_op=semantic_op,
        display_name=semantic_op.replace("_", " "),
        kind=RowKind.MIXED,
        source="xprof",
        profile_observed_time=observed_total * SECONDS_PER_MICROSECOND,
        track_summed_observed_time=observed_total * SECONDS_PER_MICROSECOND,
        observed_time_basis=_observed_basis(observations),
        observed_count=observed_count,
        observed_avg_time=observed_avg * SECONDS_PER_MICROSECOND if observed_avg is not None else None,
        matched_name=observations[0].name,
    )


def _ideal_time(estimate: FormulaEstimate, hardware: Hardware) -> float:
    if estimate.kind == RowKind.COMPUTE:
        device_flops = hardware.bf16_peak_tflops_per_device * 1e12
        return estimate.flops / device_flops if device_flops > 0 else 0.0
    bandwidth = hardware.inter_host_collective_bandwidth_gbps * 1e9 / BITS_PER_BYTE
    return estimate.bytes_accessed / bandwidth if bandwidth > 0 else 0.0


def _default_efficiency(semantic_op: str, kind: RowKind, hardware: Hardware) -> float:
    if semantic_op in {"muon_ns_gram", "muon_ns_polynomial", "muon_ns_apply"}:
        return hardware.default_compute_efficiency.get("muon_ns", 1.0)
    if semantic_op in {"attention_fa4"}:
        return hardware.default_compute_efficiency.get("attention_fwd", 1.0)
    if semantic_op in {"moe_expert"}:
        return hardware.default_compute_efficiency.get("moe_gmm", 1.0)
    if semantic_op in {"xent"}:
        return hardware.default_compute_efficiency.get("xent", 1.0)
    if kind == RowKind.COMM:
        return hardware.default_comm_efficiency.get(semantic_op, 1.0)
    return hardware.default_compute_efficiency.get("optimizer_vector_ops", 1.0)


def _observed_avg(observations: list[ObservedProfileRow]) -> float | None:
    observed_count = sum(observation.count or 0 for observation in observations)
    observed_total = sum(observation.total_time for observation in observations)
    if observed_count > 0:
        return observed_total / observed_count
    avgs = [observation.avg_time for observation in observations if observation.avg_time is not None]
    return sum(avgs) / len(avgs) if avgs else None


def _observed_basis(observations: list[ObservedProfileRow]) -> ObservedTimeBasis:
    if not observations:
        return ObservedTimeBasis.NONE
    if any(observation.basis == ObservedTimeBasis.TRACK_SUMMED_XPROF_KERNEL_TIME.value for observation in observations):
        return ObservedTimeBasis.TRACK_SUMMED_XPROF_KERNEL_TIME
    if any(
        observation.basis == ObservedTimeBasis.TRACK_SUMMED_PROFILE_HOT_OP_TIME.value for observation in observations
    ):
        return ObservedTimeBasis.TRACK_SUMMED_PROFILE_HOT_OP_TIME
    if any(observation.basis == ObservedTimeBasis.TRACE_EMPTY_TRAIN_STEP_TIME.value for observation in observations):
        return ObservedTimeBasis.TRACE_EMPTY_TRAIN_STEP_TIME
    return ObservedTimeBasis.MANUAL


def _unattributed_duration(row: dict[str, object]) -> float:
    duration = row["duration"]
    assert isinstance(duration, (int, float))
    return float(duration)

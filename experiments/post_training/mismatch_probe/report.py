# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate a mismatch report from FineStore archives without a live trainer."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from finestore.mismatch_probe import MANIFEST_TABLE, PROBE_TABLE, SCORES_TABLE, ManifestRow, ProbeRow, ScoreRow
from finestore.reader import ReadView

from experiments.post_training.mismatch_probe.metrics import (
    comparison_metrics,
    prompt_cluster_bootstrap,
)

GENERATION_SCORER = "vllm.generate"
RESCORE_SCORER = "vllm.rescore"
TRAINER_SCORER = "trainer"
UPDATE_PREFIX = "update@"

ANALYSIS_VERSION = 2
HEADLINE_METRICS = ("abs_p99", "k3", "share_beyond_2x")
NATIVE_MODE = "native"
BOOTSTRAP_DRAWS = 1000
GENERATION_SCORING = f"{GENERATION_SCORER}@0"
TRAINER_SCORING_PREFIX = f"{TRAINER_SCORER}@"


def _trainer_scoring(update: int, mode: str) -> str:
    return f"{TRAINER_SCORING_PREFIX}{update}:{mode}"


def _rescore_scoring(update: int, cache_mode: str = "off") -> str:
    return f"vllm.rescore@{update}" if cache_mode == "off" else f"vllm.rescore@{update}:{cache_mode}"


NATIVE_SCORING = _trainer_scoring(0, NATIVE_MODE)


@dataclass(frozen=True)
class ArchiveData:
    manifest: ManifestRow
    probes: list[ProbeRow]
    scores: dict[str, dict[str, ScoreRow]]
    commit_token: str


@dataclass(frozen=True)
class ComparisonRows:
    target: list[list[float]]
    reference: list[list[float]]
    masks: list[list[bool]]

    def metrics(self, indices: list[int]) -> dict[str, float | int]:
        values = comparison_metrics(
            [self.target[index] for index in indices],
            [self.reference[index] for index in indices],
            [self.masks[index] for index in indices],
        )
        return {
            name: value
            for name, value in values.items()
            if name not in {"chi2_sample_moment", "token_ess_fraction_raw", "sequence_ess_fraction_raw"}
        }


def _comparison_rows(
    probes: list[ProbeRow], scores: dict[str, dict[str, ScoreRow]], target: str, reference: str
) -> ComparisonRows:
    return ComparisonRows(
        target=[scores[target][row.sample_id].logprobs for row in probes],
        reference=[scores[reference][row.sample_id].logprobs for row in probes],
        masks=[row.loss_mask for row in probes],
    )


def _score_label(row: ScoreRow) -> str:
    if row.scorer == TRAINER_SCORER:
        return _trainer_scoring(row.update, row.mode)
    if row.scorer == RESCORE_SCORER:
        return _rescore_scoring(row.update, row.cache_mode)
    return f"{row.scorer}@{row.update}"


def load_archive(uri: str) -> ArchiveData:
    view = ReadView(uri)
    manifests = [ManifestRow.model_validate(row) for row in view.scan(MANIFEST_TABLE).to_pylist()]
    if len(manifests) != 1 or manifests[0].status != "complete":
        raise ValueError(f"mismatch archive {uri} is incomplete or has no unique manifest")
    manifest = manifests[0]
    probes = [ProbeRow.model_validate(row) for row in view.scan(PROBE_TABLE).to_pylist()]
    probes.sort(key=lambda row: row.batch_position)
    if not probes or len({row.sample_id for row in probes}) != len(probes):
        raise ValueError("mismatch archive has no unique frozen samples")
    if any(row.probe_hash != manifest.probe_hash for row in probes):
        raise ValueError("mismatch archive probe rows disagree with its manifest hash")
    scores: dict[str, dict[str, ScoreRow]] = {}
    valid_ids = {row.sample_id for row in probes}
    for raw in view.scan(SCORES_TABLE).to_pylist():
        row = ScoreRow.model_validate(raw)
        if row.probe_hash != manifest.probe_hash or row.sample_id not in valid_ids:
            raise ValueError("mismatch archive has a scoring outside its frozen probe")
        sample_scores = scores.setdefault(_score_label(row), {})
        if row.sample_id in sample_scores:
            raise ValueError(f"duplicate scoring {_score_label(row)} for {row.sample_id}")
        sample_scores[row.sample_id] = row
    for name, sample_scores in scores.items():
        if set(sample_scores) != valid_ids:
            raise ValueError(f"scoring {name} does not cover every frozen sample")
        if (
            len({score.update for score in sample_scores.values()}) != 1
            or len({score.weights_hash for score in sample_scores.values()}) != 1
        ):
            raise ValueError(f"scoring {name} mixes updates or weight identities")
        for probe in probes:
            score = sample_scores[probe.sample_id]
            if len(score.logprobs) != len(probe.vllm_output_ids):
                raise ValueError(f"scoring {name} has incomplete tokens for {probe.sample_id}")
            if any(not math.isfinite(value) for value in score.logprobs):
                raise ValueError(f"scoring {name} has nonfinite tokens for {probe.sample_id}")
            if score.update == 0 and score.weights_hash != manifest.starting_weights_hash:
                raise ValueError(f"scoring {name} differs from the starting weight hash")
    return ArchiveData(manifest=manifest, probes=probes, scores=scores, commit_token=str(view.token))


def _require_same_weights_if_same_update(
    label: str, target: dict[str, ScoreRow], reference: dict[str, ScoreRow]
) -> None:
    """Reject purported same-weight comparisons before numerical analysis."""
    target_row = next(iter(target.values()))
    reference_row = next(iter(reference.values()))
    if target_row.update == reference_row.update and target_row.weights_hash != reference_row.weights_hash:
        raise ValueError(f"comparison {label} scores different weights at update {target_row.update}")


def _trainer_modes(scores: dict[str, dict[str, ScoreRow]]) -> list[str]:
    return sorted(
        {
            row.mode
            for rows in scores.values()
            for row in rows.values()
            if row.scorer == TRAINER_SCORER and row.mode not in {NATIVE_MODE, "repeat"}
        }
    )


def _comparison_definitions(scores: dict[str, dict[str, ScoreRow]]) -> dict[str, tuple[str, str]]:
    names = set(scores)
    comparisons: dict[str, tuple[str, str]] = {}

    def add(label: str, target: str, reference: str):
        if target in names and reference in names:
            comparisons[label] = target, reference

    def reread(update: int) -> str:
        uncached = _rescore_scoring(update)
        return uncached if uncached in names else _rescore_scoring(update, "on")

    add("implementation_mismatch", NATIVE_SCORING, GENERATION_SCORING)
    add("trainer_floor", _trainer_scoring(0, "repeat"), NATIVE_SCORING)
    for mode in _trainer_modes(scores):
        add(f"{mode}_vs_generation", _trainer_scoring(0, mode), GENERATION_SCORING)
    updates = sorted({row.update for rows in scores.values() for row in rows.values() if row.update > 0})
    for update in updates:
        add(f"trainer_drift_after_{update}", _trainer_scoring(update, NATIVE_MODE), NATIVE_SCORING)
        add(f"observed_gap_after_{update}", _trainer_scoring(update, NATIVE_MODE), GENERATION_SCORING)
        add(f"vllm_drift_after_{update}", reread(update), reread(0))
        add(f"mismatch_after_{update}", _trainer_scoring(update, NATIVE_MODE), reread(update))
        for mode in _trainer_modes(scores):
            add(f"{mode}_after_{update}", _trainer_scoring(update, mode), reread(update))
    return comparisons


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def _route_counts(probes: list[ProbeRow], sample_scores: dict[str, ScoreRow]) -> tuple[np.ndarray, np.ndarray]:
    counts, matches = [], []
    for probe in probes:
        source = np.frombuffer(probe.routed_experts, dtype=probe.routed_experts_dtype).reshape(
            probe.routed_experts_shape
        )
        score = sample_scores[probe.sample_id]
        if score.expert_choices_shape != list(source.shape):
            raise ValueError(f"route observation shape differs from capture for {probe.sample_id}")
        actual = np.frombuffer(score.expert_choices, dtype=score.expert_choices_dtype).reshape(source.shape)
        valid = np.asarray(probe.route_valid_mask, dtype=np.bool_) & np.asarray(probe.loss_mask, dtype=np.bool_)[:, None]
        if np.any(actual[valid] < 0):
            raise ValueError(f"missing trainer route observation for {probe.sample_id}")
        equal = (np.sort(source, axis=-1) == np.sort(actual, axis=-1)).all(axis=-1)
        counts.append(valid.sum(axis=0))
        matches.append((equal & valid).sum(axis=0))
    return np.asarray(counts), np.asarray(matches)


def _route_statistics(indices: list[int], counts: np.ndarray, matches: np.ndarray) -> dict[str, float]:
    layer_counts, layer_matches = counts[indices].sum(axis=0), matches[indices].sum(axis=0)
    total = int(layer_counts.sum())
    values = {"set_agreement": float(layer_matches.sum() / total) if total else math.nan}
    for layer, count in enumerate(layer_counts):
        values[f"layer_{layer}/set_agreement"] = float(layer_matches[layer] / count) if count else math.nan
    return values


def _route_diagnostics(probes: list[ProbeRow], scores: dict[str, dict[str, ScoreRow]], seed: int, draws: int) -> dict:
    """Compare expert sets on captured, unmasked token-and-layer rows."""
    result = {}
    if any(row.routed_experts is None for row in probes):
        return result
    for name, sample_scores in scores.items():
        if any(score.scorer != TRAINER_SCORER or score.expert_choices is None for score in sample_scores.values()):
            continue
        counts, matches = _route_counts(probes, sample_scores)
        bootstrap = prompt_cluster_bootstrap(
            [row.prompt_id for row in probes],
            lambda indices, counts=counts, matches=matches: _route_statistics(indices, counts, matches),
            seed=seed,
            draws=draws,
        )
        result[name] = {
            "metrics": {"set_agreement": bootstrap.point["set_agreement"]},
            "ci95": {"set_agreement": bootstrap.intervals.get("set_agreement")},
            "layers": {
                str(layer): {
                    "metrics": {"set_agreement": bootstrap.point[f"layer_{layer}/set_agreement"]},
                    "ci95": {"set_agreement": bootstrap.intervals.get(f"layer_{layer}/set_agreement")},
                }
                for layer in range(counts.shape[1])
            },
        }
    return result


def _timing_values(archive: ArchiveData) -> dict[str, float]:
    values = {
        name: seconds
        for name, seconds in json.loads(archive.manifest.timing_json).items()
        if isinstance(seconds, (float, int))
    }
    for update, metrics in json.loads(archive.manifest.step_metrics_json).items():
        if not update.startswith(UPDATE_PREFIX) or not isinstance(metrics, dict):
            continue
        for name, seconds in metrics.get("step_timings", {}).items():
            if isinstance(seconds, (float, int)):
                values[f"training/{update}/{name}"] = seconds
    return values


def analyze_archive(uri: str, *, bootstrap_draws: int = BOOTSTRAP_DRAWS) -> dict:
    archive = load_archive(uri)
    manifest, probes, scores = archive.manifest, archive.probes, archive.scores
    identity = {
        "samples": len(probes),
        "matching_responses": sum(row.vllm_output_ids == row.trainer_input_ids for row in probes),
        "matching_prompts": sum(row.prompt_token_ids == row.trainer_prompt_ids for row in probes),
    }
    identity["fraction"] = min(identity["matching_responses"], identity["matching_prompts"]) / len(probes)
    report = {
        "analysis_version": ANALYSIS_VERSION,
        "archive": uri,
        "manifest": manifest.model_dump(),
        "input_commit_token": archive.commit_token,
        "bootstrap": {"seed": manifest.bootstrap_seed, "draws": bootstrap_draws, "cluster": "prompt_id"},
        "token_identity": identity,
        "comparisons": {},
        "paired_improvements": {},
        "route_diagnostics": {},
        "checks": {},
        "timing": json.loads(manifest.timing_json),
        "step_metrics": json.loads(manifest.step_metrics_json),
    }
    if identity["fraction"] != 1.0:
        report["checks"]["token_identity"] = "failed"
        return _finite_json(report)

    report["route_diagnostics"] = _route_diagnostics(probes, scores, manifest.bootstrap_seed, bootstrap_draws)

    prompt_ids = [row.prompt_id for row in probes]
    definitions = _comparison_definitions(scores)
    sampled_metrics = {}
    for label, (target_name, reference_name) in definitions.items():
        _require_same_weights_if_same_update(label, scores[target_name], scores[reference_name])
        rows = _comparison_rows(probes, scores, target_name, reference_name)

        def calculate(indices, *, rows=rows):
            return rows.metrics(indices)

        bootstrap = prompt_cluster_bootstrap(prompt_ids, calculate, seed=manifest.bootstrap_seed, draws=bootstrap_draws)
        report["comparisons"][label] = {
            "target": target_name,
            "reference": reference_name,
            "metrics": bootstrap.point,
            "ci95": bootstrap.intervals,
            "reference_distribution": (
                "generation" if reference_name == GENERATION_SCORING else "diagnostic_re_read_or_trainer"
            ),
        }
        if target_name in report["route_diagnostics"]:
            route = report["route_diagnostics"][target_name]
            report["comparisons"][label]["route_set_agreement"] = {
                "value": route["metrics"]["set_agreement"],
                "ci95": route["ci95"].get("set_agreement"),
            }
        sampled_metrics[label] = bootstrap.draws

    baseline = "implementation_mismatch"
    for mode in _trainer_modes(scores):
        variant = f"{mode}_vs_generation"
        if baseline not in sampled_metrics or variant not in sampled_metrics:
            continue
        paired = {}
        for metric in HEADLINE_METRICS:
            draws = np.asarray(
                [
                    left[metric] - right[metric]
                    for left, right in zip(sampled_metrics[baseline], sampled_metrics[variant], strict=True)
                ]
            )
            point = (
                report["comparisons"][baseline]["metrics"][metric] - report["comparisons"][variant]["metrics"][metric]
            )
            paired[metric] = {"native_minus_mode": float(point), "ci95": np.percentile(draws, [2.5, 97.5]).tolist()}
        report["paired_improvements"][mode] = paired

    if "implementation_mismatch" in sampled_metrics:
        baseline_stats = report["comparisons"]["implementation_mismatch"]["metrics"]
        bootstrap_ratios = np.asarray([draw["mean_ratio"] for draw in sampled_metrics["implementation_mismatch"]])
        standard_error = float(bootstrap_ratios.std(ddof=1)) if len(bootstrap_ratios) > 1 else math.nan
        mean_ratio = baseline_stats["mean_ratio"]
        passes = math.isfinite(standard_error) and mean_ratio <= 1 + 3 * standard_error
        report["checks"]["generation_ratio_sanity"] = {
            "mean_ratio": mean_ratio,
            "bootstrap_standard_error": standard_error,
            "pass": passes,
        }
        if not passes:
            raise ValueError(
                f"generation-time sampling-distribution check failed: mean ratio {mean_ratio:.6g}, "
                f"bootstrap standard error {standard_error:.6g}"
            )
    return _finite_json(report)


def compare_archives(
    left_uri: str,
    right_uri: str,
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict:
    """Compare matched prompt groups from two runs with the same starting policy."""
    left = load_archive(left_uri)
    right = load_archive(right_uri)
    for archive in (left, right):
        if any(
            row.prompt_token_ids != row.trainer_prompt_ids or row.vllm_output_ids != row.trainer_input_ids
            for row in archive.probes
        ):
            raise ValueError("configuration A/B requires exact trainer and sampler token identity in both archives")
    if left.manifest.starting_weights_hash != right.manifest.starting_weights_hash:
        raise ValueError("configuration A/B requires the same starting weight hash")
    if left.manifest.vllm_enforce_eager != right.manifest.vllm_enforce_eager:
        raise ValueError("configuration A/B requires the same vLLM execution mode")
    left_tokenizer = left.manifest.tokenizer_fingerprint
    right_tokenizer = right.manifest.tokenizer_fingerprint
    if not left_tokenizer or left_tokenizer != right_tokenizer:
        raise ValueError("configuration A/B requires the same tokenizer fingerprint")
    left_by_id = {row.sample_id: row for row in left.probes}
    right_by_id = {row.sample_id: row for row in right.probes}
    if set(left_by_id) != set(right_by_id):
        raise ValueError("configuration A/B requires the same prompt and repetition IDs")
    aligned_right_probes = [right_by_id[row.sample_id] for row in left.probes]
    for left_row, right_row in zip(left.probes, aligned_right_probes, strict=True):
        if left_row.prompt_id != right_row.prompt_id or left_row.prompt_token_ids != right_row.prompt_token_ids:
            raise ValueError("configuration A/B requires identical prompt token IDs")
        if left_row.request_seed != right_row.request_seed:
            raise ValueError("configuration A/B requires paired request seeds")
    left_sampling = json.loads(left.manifest.config_json).get("generator", {}).get("sampling_params")
    right_sampling = json.loads(right.manifest.config_json).get("generator", {}).get("sampling_params")
    if left_sampling != right_sampling:
        raise ValueError("configuration A/B requires identical sampling parameters")
    if left.manifest.probe_hash != right.manifest.probe_hash:
        raise ValueError("configuration A/B requires the same frozen probe hash")
    left_cache_modes = {
        score.cache_mode for rows in left.scores.values() for score in rows.values() if score.scorer == RESCORE_SCORER
    }
    right_cache_modes = {
        score.cache_mode for rows in right.scores.values() for score in rows.values() if score.scorer == RESCORE_SCORER
    }
    if left_cache_modes != right_cache_modes:
        raise ValueError("configuration A/B requires the same prefix-cache modes")
    if any(
        left_row.vllm_output_ids != right_row.vllm_output_ids or left_row.loss_mask != right_row.loss_mask
        for left_row, right_row in zip(left.probes, aligned_right_probes, strict=True)
    ):
        raise ValueError("shared-token archives disagree with their probe hash")
    left_definitions = _comparison_definitions(left.scores)
    right_definitions = _comparison_definitions(right.scores)
    common = sorted(set(left_definitions) & set(right_definitions))
    if not common:
        raise ValueError("configuration A/B has no common scoring comparison")
    result = {
        "left": left_uri,
        "right": right_uri,
        "kind": "shared_tokens",
        "starting_weights_hash": left.manifest.starting_weights_hash,
        "probe_hashes": [left.manifest.probe_hash, right.manifest.probe_hash],
        "bootstrap": {"seed": left.manifest.bootstrap_seed, "draws": bootstrap_draws, "cluster": "prompt_id"},
        "comparisons": {},
    }
    prompt_ids = [row.prompt_id for row in left.probes]
    for label in common:
        left_target, left_reference = left_definitions[label]
        right_target, right_reference = right_definitions[label]
        _require_same_weights_if_same_update(label, left.scores[left_target], left.scores[left_reference])
        _require_same_weights_if_same_update(label, right.scores[right_target], right.scores[right_reference])
        left_rows = _comparison_rows(left.probes, left.scores, left_target, left_reference)
        right_rows = _comparison_rows(aligned_right_probes, right.scores, right_target, right_reference)

        def calculate(
            indices,
            *,
            left_rows=left_rows,
            right_rows=right_rows,
        ):
            left_values = left_rows.metrics(indices)
            right_values = right_rows.metrics(indices)
            return {f"{name}_left_minus_right": left_values[name] - right_values[name] for name in HEADLINE_METRICS}

        bootstrap = prompt_cluster_bootstrap(
            prompt_ids, calculate, seed=left.manifest.bootstrap_seed, draws=bootstrap_draws
        )
        result["comparisons"][label] = {"metrics": bootstrap.point, "ci95": bootstrap.intervals}
    left_timing, right_timing = _timing_values(left), _timing_values(right)
    result["timing"] = {
        name: {
            "left_seconds": left_timing.get(name),
            "right_seconds": right_timing.get(name),
            "right_minus_left_seconds": (
                right_timing[name] - left_timing[name] if name in left_timing and name in right_timing else None
            ),
        }
        for name in sorted(left_timing.keys() | right_timing.keys())
    }
    result["routes"] = {}
    if all(row.routed_experts is not None for row in left.probes + aligned_right_probes):
        for name in sorted(left.scores.keys() & right.scores.keys()):
            if any(
                score.scorer != TRAINER_SCORER or score.expert_choices is None
                for score in list(left.scores[name].values()) + list(right.scores[name].values())
            ):
                continue
            left_counts, left_matches = _route_counts(left.probes, left.scores[name])
            right_counts, right_matches = _route_counts(aligned_right_probes, right.scores[name])

            def route_difference(
                indices,
                *,
                left_counts=left_counts,
                left_matches=left_matches,
                right_counts=right_counts,
                right_matches=right_matches,
            ):
                before = _route_statistics(indices, left_counts, left_matches)
                after = _route_statistics(indices, right_counts, right_matches)
                return {key: after[key] - before[key] for key in before}

            bootstrap = prompt_cluster_bootstrap(
                prompt_ids, route_difference, seed=left.manifest.bootstrap_seed, draws=bootstrap_draws
            )
            result["routes"][name] = {"right_minus_left": bootstrap.point, "ci95": bootstrap.intervals}
    return _finite_json(result)


def write_plots(report: dict, output_dir: Path) -> None:
    """Write static figures from the same archive-derived numbers as the report."""
    plt.switch_backend("Agg")

    routes = report.get("route_diagnostics", {})
    if routes:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, item in routes.items():
            layer_values = [
                (int(layer), details["metrics"]["set_agreement"])
                for layer, details in sorted(item["layers"].items(), key=lambda entry: int(entry[0]))
            ]
            if layer_values:
                ax.plot(
                    [index for index, _ in layer_values],
                    [np.nan if isinstance(value, dict) else value for _, value in layer_values],
                    marker="o",
                    label=name,
                )
        ax.set(xlabel="MoE layer", ylabel="expert-set agreement", ylim=(0, 1))
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(output_dir / "routes.png", dpi=160)
        plt.close(fig)


def _display_metric(value, *, percent_digits: int | None = None, scale: float = 1.0) -> str:
    if isinstance(value, dict) and "nonfinite" in value:
        return f"nonfinite ({value['nonfinite']})"
    if not isinstance(value, (int, float)):
        return "unavailable"
    value *= scale
    if not math.isfinite(value):
        return f"nonfinite ({value})"
    return f"{value:.{percent_digits}%}" if percent_digits is not None else f"{value:.5g}"


def _display_interval(value, *, percent_digits: int | None = None, scale: float = 1.0) -> str:
    if value is None:
        return "-"
    low, high = value
    return (
        f"[{_display_metric(low, percent_digits=percent_digits, scale=scale)}, "
        f"{_display_metric(high, percent_digits=percent_digits, scale=scale)}]"
    )


def render_markdown(report: dict) -> str:
    identity = report["token_identity"]
    lines = [
        "# Mismatch probe",
        "",
        f"Archive: `{report['archive']}`",
        "",
        f"Token identity: {identity['matching_responses']}/{identity['samples']} responses, "
        f"{identity['matching_prompts']}/{identity['samples']} prompts ({identity['fraction']:.1%}).",
        "",
        "## Checks",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]
    for name, check in report["checks"].items():
        status = ("passed" if check["pass"] else "failed") if isinstance(check, dict) else check
        details = json.dumps(check, sort_keys=True) if isinstance(check, dict) else ""
        lines.append(f"| {name} | {status} | {details} |")
    lines.append("")
    if report["comparisons"]:
        lines.extend(
            [
                "Δ is target minus reference log probability on masked response tokens. "
                "P99 is the 99th percentile of its absolute value.",
                "Replay uses routes captured during the original generation; later updates measure stale-route replay.",
                f"95% intervals resample whole prompts {report['bootstrap']['draws']} times "
                f"with seed {report['bootstrap']['seed']}.",
                "",
                "| Comparison | p99 abs Δ | 95% CI | k3 | 95% CI | beyond 2x | 95% CI | route agreement | 95% CI |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, item in report["comparisons"].items():
            metrics, ci = item["metrics"], item["ci95"]
            route = item.get("route_set_agreement")
            route_text = "-" if route is None else _display_metric(route["value"], percent_digits=1)
            route_ci = "-" if route is None else _display_interval(route["ci95"], percent_digits=1)
            lines.append(
                f"| {name} | {_display_metric(metrics['abs_p99'])} | {_display_interval(ci.get('abs_p99'))} | "
                f"{_display_metric(metrics['k3'])} | {_display_interval(ci.get('k3'))} | "
                f"{_display_metric(metrics['share_beyond_2x'], percent_digits=3)} | "
                f"{_display_interval(ci.get('share_beyond_2x'), percent_digits=3)} | {route_text} | {route_ci} |"
            )
        lines.extend(
            [
                "",
                "## Numerical details",
                "",
                "| Comparison | tokens | min abs Δ | mean abs Δ | p50 | p75 | p90 | p99.9 | max abs Δ | signed mean Δ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, item in report["comparisons"].items():
            values = item["metrics"]
            keys = ("abs_min", "abs_mean", "abs_p50", "abs_p75", "abs_p90", "abs_p999", "abs_max", "delta_mean")
            numbers = " | ".join(_display_metric(values[key]) for key in keys)
            lines.append(f"| {name} | {values['tokens']} | {numbers} |")
        lines.extend(
            [
                "",
                "## Paired trainer-mode improvements",
                "",
                "Positive native minus mode means a smaller mismatch in that mode.",
                "",
                "| Mode | p99 improvement | 95% paired CI |",
                "|---|---:|---:|",
            ]
        )
        for mode, item in report["paired_improvements"].items():
            effect = item["abs_p99"]
            lines.append(
                f"| {mode} | {_display_metric(effect['native_minus_mode'])} | {_display_interval(effect['ci95'])} |"
            )
        lines.append("")
    if report["route_diagnostics"]:
        lines.extend(
            ["## Routes by layer", "", "| Scorer | Layer | Expert-set agreement | 95% CI |", "|---|---:|---:|---:|"]
        )
        for name, item in report["route_diagnostics"].items():
            for layer, details in item["layers"].items():
                lines.append(
                    f"| {name} | {layer} | {_display_metric(details['metrics']['set_agreement'], percent_digits=1)} | "
                    f"{_display_interval(details['ci95'].get('set_agreement'), percent_digits=1)} |"
                )
        lines.append("")
    if report["timing"]:
        lines.extend(
            [
                "## Timing",
                "",
                "Scoring timers include dispatch and compilation; "
                "these observations do not estimate total RL step overhead.",
                "",
                "| Timer | seconds |",
                "|---|---:|",
            ]
        )
        for name, seconds in sorted(report["timing"].items()):
            if isinstance(seconds, (int, float)):
                lines.append(f"| {name} | {_display_metric(seconds)} |")
        lines.append("")
    step_timings = [
        (step, name, value)
        for step, metrics in sorted(report["step_metrics"].items())
        if isinstance(metrics, dict) and step.startswith(UPDATE_PREFIX)
        for name, value in sorted(metrics.get("step_timings", {}).items())
        if isinstance(value, (int, float))
    ]
    if step_timings:
        lines.extend(["## Training timing", "", "| Probe update | Timer | seconds |", "|---|---|---:|"])
        for step, name, value in step_timings:
            lines.append(f"| {step} | {name} | {_display_metric(value)} |")
        lines.append("")
    return "\n".join(lines)


def render_archive_comparison(comparison: dict) -> str:
    lines = [
        "## Configuration comparison",
        "",
        f"Left archive: `{comparison['left']}`",
        "",
        f"Right archive: `{comparison['right']}`",
        "",
        "The archives contain the same frozen responses.",
        "",
        "| Comparison | p99 abs Δ difference (left minus right) | 95% paired CI |",
        "|---|---:|---:|",
    ]
    for label, item in comparison["comparisons"].items():
        key = "abs_p99_left_minus_right"
        value = item["metrics"][key]
        lo, hi = item["ci95"][key]
        lines.append(f"| {label} | {value:.5g} | [{lo:.5g}, {hi:.5g}] |")
    lines.extend(
        [
            "",
            "## Timing changes",
            "",
            "| Timer | Left seconds | Right seconds | Right minus left seconds |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, item in comparison["timing"].items():
        values = " | ".join(
            _display_metric(item[key]) for key in ("left_seconds", "right_seconds", "right_minus_left_seconds")
        )
        lines.append(f"| {name} | {values} |")
    lines.extend(
        [
            "",
            "## Route changes",
            "",
            "| Scorer | Layer | Right minus left agreement (pp) | 95% paired CI (pp) |",
            "|---|---|---:|---:|",
        ]
    )
    for name, item in comparison["routes"].items():
        for key, value in item["right_minus_left"].items():
            layer = "all" if key == "set_agreement" else key.split("/", 1)[0]
            lines.append(
                f"| {name} | {layer} | {_display_metric(value, scale=100)} | "
                f"{_display_interval(item['ci95'].get(key), scale=100)} |"
            )
    return "\n".join(lines) + "\n"


def _write_single_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "report.md").write_text(render_markdown(report))
    write_plots(report, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", help="FineStore URIs of completed mismatch probes")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [analyze_archive(uri, bootstrap_draws=args.bootstrap_draws) for uri in args.archives]
    if len(reports) == 1:
        _write_single_report(reports[0], args.output_dir)
        return
    comparisons = [
        compare_archives(args.archives[0], uri, bootstrap_draws=args.bootstrap_draws) for uri in args.archives[1:]
    ]
    for index, report in enumerate(reports):
        run_dir = args.output_dir / f"run-{index}"
        _write_single_report(report, run_dir)
    combined = {
        "analysis_version": ANALYSIS_VERSION,
        "archives": args.archives,
        "configuration_comparisons": comparisons,
    }
    (args.output_dir / "report.json").write_text(json.dumps(combined, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(
        "# Mismatch probe configuration comparisons\n\n"
        + "\n".join(render_archive_comparison(item) for item in comparisons)
    )


if __name__ == "__main__":
    main()

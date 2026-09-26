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
from finestore.mismatch import MANIFEST_TABLE, PROBE_TABLE, SCORES_TABLE, ManifestRow, ProbeRow, ScoreRow
from finestore.reader import ReadView

from experiments.post_training.mismatch_probe.metrics import comparison_metrics, prompt_cluster_bootstrap

ANALYSIS_VERSION = 1
HEADLINE_METRICS = ("abs_p99", "k3", "share_beyond_2x", "token_ess_fraction_raw")
PROBABILITY_BINS = ((0.0, 0.001), (0.001, 0.01), (0.01, 0.1), (0.1, 1.0))
REPLAY_MODES = ("router_replay", "router_replay_filtered")
BOOTSTRAP_DRAWS = 1000
GENERATION_SCORING = "vllm.generate@0"
NATIVE_SCORING = "trainer@0:native"


@dataclass(frozen=True)
class ArchiveData:
    manifest: ManifestRow
    probes: list[ProbeRow]
    scores: dict[str, dict[str, ScoreRow]]


def _clip_thresholds(manifest: ManifestRow) -> tuple[float, float]:
    algorithm = json.loads(manifest.config_json).get("trainer", {}).get("algorithm", {})
    low = float(algorithm.get("eps_clip_low", 0.2))
    high = float(algorithm.get("eps_clip_high", 0.2))
    if not (0 <= low < 1 and high >= 0 and math.isfinite(low) and math.isfinite(high)):
        raise ValueError("mismatch archive has invalid PPO clipping thresholds")
    return low, high


def _probability_bucket(logprob: float) -> str:
    probability = math.exp(logprob)
    return next(
        (f"{lower:g}-{upper:g}" for lower, upper in PROBABILITY_BINS if lower <= probability < upper),
        "1",
    )


def _layer_numbers(metrics: dict) -> list[int]:
    return sorted(
        int(key.split("/")[0][6:]) for key in metrics if key.startswith("layer_") and key.endswith("/set_agreement")
    )


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
        sample_scores = scores.setdefault(row.scoring, {})
        if row.sample_id in sample_scores:
            raise ValueError(f"duplicate scoring {row.scoring} for {row.sample_id}")
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
    return ArchiveData(manifest=manifest, probes=probes, scores=scores)


def _require_same_weights_if_same_update(
    label: str, target: dict[str, ScoreRow], reference: dict[str, ScoreRow]
) -> None:
    """Reject purported same-weight comparisons before numerical analysis."""
    target_row = next(iter(target.values()))
    reference_row = next(iter(reference.values()))
    if target_row.update == reference_row.update and target_row.weights_hash != reference_row.weights_hash:
        raise ValueError(f"comparison {label} scores different weights at update {target_row.update}")


def _comparison_definitions(scores: dict[str, dict[str, ScoreRow]]) -> dict[str, tuple[str, str]]:
    names = set(scores)
    comparisons: dict[str, tuple[str, str]] = {}

    def add(label: str, target: str, reference: str):
        if target in names and reference in names:
            comparisons[label] = target, reference

    def reread(update: int) -> str:
        uncached = f"vllm.rescore@{update}"
        return uncached if uncached in names else f"{uncached}:on"

    add("implementation_mismatch", NATIVE_SCORING, GENERATION_SCORING)
    add("trainer_floor", "trainer@0:repeat", NATIVE_SCORING)
    add("generate_vs_reread", reread(0), GENERATION_SCORING)
    add("reread_mismatch", NATIVE_SCORING, reread(0))
    add("prefix_cache_effect_at_0", "vllm.rescore@0:on", "vllm.rescore@0")
    for mode in REPLAY_MODES:
        add(f"{mode}_vs_generation", f"trainer@0:{mode}", GENERATION_SCORING)
        add(f"{mode}_vs_reread", f"trainer@0:{mode}", reread(0))
    updates = sorted({row.update for rows in scores.values() for row in rows.values() if row.update > 0})
    for update in updates:
        add(f"trainer_drift_after_{update}", f"trainer@{update}:native", NATIVE_SCORING)
        add(f"observed_gap_after_{update}", f"trainer@{update}:native", GENERATION_SCORING)
        add(f"vllm_drift_after_{update}", reread(update), reread(0))
        add(f"mismatch_after_{update}", f"trainer@{update}:native", reread(update))
        add(f"prefix_cache_effect_at_{update}", f"vllm.rescore@{update}:on", f"vllm.rescore@{update}")
        for mode in REPLAY_MODES:
            add(f"{mode}_vs_reread_after_{update}", f"trainer@{update}:{mode}", reread(update))
    return comparisons


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def _route_diagnostics(probes: list[ProbeRow], scores: dict[str, dict[str, ScoreRow]], seed: int, draws: int) -> dict:
    """Compare archived Megatron choices with the sampler's captured expert IDs."""
    prompt_ids = [row.prompt_id for row in probes]
    result = {}
    if any(row.routed_experts is None for row in probes):
        return result
    captured = [
        np.frombuffer(row.routed_experts, dtype=row.routed_experts_dtype).reshape(row.routed_experts_shape)
        for row in probes
    ]
    for name, sample_scores in scores.items():
        if not name.startswith("trainer@") or any(sample_scores[row.sample_id].expert_choices is None for row in probes):
            continue
        selected = []
        replaced = []
        for probe, routes in zip(probes, captured, strict=True):
            score = sample_scores[probe.sample_id]
            if score.expert_choices_shape != list(routes.shape):
                raise ValueError(f"route observation shape differs from capture for {name}/{probe.sample_id}")
            choices = np.frombuffer(score.expert_choices, dtype=score.expert_choices_dtype).reshape(routes.shape)
            change = np.frombuffer(score.replacement_mask, dtype=np.bool_).reshape(routes.shape)
            selected.append(choices)
            replaced.append(change)

        def calculate(indices, *, selected=selected, replaced=replaced, name=name):
            token_count = valid_count = set_match = slot_match = any_disagreement = replacements = 0
            differing_layer_total = 0
            layer_valid = np.zeros(captured[0].shape[1], dtype=np.int64)
            layer_set = np.zeros_like(layer_valid)
            layer_slot = np.zeros_like(layer_valid)
            layer_replaced = np.zeros_like(layer_valid)
            topk = captured[0].shape[-1]
            for index in indices:
                source = captured[index]
                actual = selected[index]
                changed = replaced[index]
                loss_mask = np.asarray(probes[index].loss_mask, dtype=np.bool_)
                route_valid = np.asarray(probes[index].route_valid_mask, dtype=np.bool_) & loss_mask[:, None]
                valid_tokens = route_valid.any(axis=-1)
                if np.any(actual[route_valid] < 0):
                    raise ValueError(f"missing trainer route observation for {name}/{probes[index].sample_id}")
                token_count += int(loss_mask.sum())
                valid_count += int(valid_tokens.sum())
                exact = (source == actual).all(axis=-1)
                sets = (np.sort(source, axis=-1) == np.sort(actual, axis=-1)).all(axis=-1)
                set_match += int(sets[route_valid].sum())
                slot_match += int(exact[route_valid].sum())
                any_disagreement += int(((~sets & route_valid).any(axis=-1) & valid_tokens).sum())
                differing_layer_total += int((~sets & route_valid).sum())
                replacements += int(changed[route_valid].sum())
                layer_valid += route_valid.sum(axis=0)
                layer_set += (sets & route_valid).sum(axis=0)
                layer_slot += (exact & route_valid).sum(axis=0)
                layer_replaced += (changed & route_valid[:, :, None]).sum(axis=(0, 2))
            denominator = int(layer_valid.sum())
            metrics = {
                "valid_coverage": valid_count / token_count if token_count else math.nan,
                "set_agreement": set_match / denominator if denominator else math.nan,
                "exact_slot_agreement": slot_match / denominator if denominator else math.nan,
                "any_layer_disagreement": any_disagreement / valid_count if valid_count else math.nan,
                "mean_differing_layers": differing_layer_total / valid_count if valid_count else math.nan,
                "replacement_fraction": replacements / (denominator * topk) if denominator else math.nan,
                "valid_tokens": float(valid_count),
                "loss_tokens": float(token_count),
            }
            for layer in range(len(layer_valid)):
                count = layer_valid[layer]
                metrics[f"layer_{layer}/set_agreement"] = float(layer_set[layer] / count) if count else math.nan
                metrics[f"layer_{layer}/exact_slot_agreement"] = float(layer_slot[layer] / count) if count else math.nan
                metrics[f"layer_{layer}/replacement_fraction"] = (
                    float(layer_replaced[layer] / (count * topk)) if count else math.nan
                )
            return metrics

        bootstrap = prompt_cluster_bootstrap(prompt_ids, calculate, seed=seed, draws=draws)
        result[name] = {"metrics": bootstrap.point, "ci95": bootstrap.intervals}
    return result


def _breakdowns(probes: list[ProbeRow], target: dict[str, ScoreRow], reference: dict[str, ScoreRow]) -> dict:
    buckets: dict[str, dict[str, list[float]]] = {
        axis: {} for axis in ("reference_probability", "response_position", "answer_length")
    }
    for row in probes:
        mask = np.asarray(row.loss_mask, dtype=np.bool_)
        chosen = np.asarray(target[row.sample_id].logprobs, dtype=np.float64)
        baseline = np.asarray(reference[row.sample_id].logprobs, dtype=np.float64)
        for position in np.flatnonzero(mask):
            difference = float(chosen[position] - baseline[position])
            probability_key = _probability_bucket(float(baseline[position]))
            position_key = "1-4" if position < 4 else "5-16" if position < 16 else "17-64" if position < 64 else "65+"
            length = int(mask.sum())
            length_key = "1-16" if length <= 16 else "17-64" if length <= 64 else "65+"
            for axis, key in (
                ("reference_probability", probability_key),
                ("response_position", position_key),
                ("answer_length", length_key),
            ):
                buckets[axis].setdefault(key, []).append(difference)
    return {
        axis: {
            key: {
                "tokens": len(values),
                "abs_p99": float(np.percentile(np.abs(values), 99)),
                "mean_squared_delta": float(np.mean(np.square(values))),
                "signed_mean": float(np.mean(values)),
            }
            for key, values in groups.items()
        }
        for axis, groups in buckets.items()
    }


def _drift_scale(probes: list[ProbeRow], scores: dict[str, dict[str, ScoreRow]], updates: list[int]) -> dict:
    names = (GENERATION_SCORING, NATIVE_SCORING)
    if not all(name in scores for name in names):
        return {}
    result = {"probability_buckets": {}, "equivalent_updates": {"status": "unavailable"}}
    drift_mean_squares = []
    baseline_squared = []
    for update in updates:
        drift_name = f"trainer@{update}:native"
        if drift_name not in scores:
            continue
        per_bucket: dict[str, dict[str, list[float]]] = {}
        drift_squared = []
        for row in probes:
            sample = row.sample_id
            for position in np.flatnonzero(row.loss_mask):
                generation = scores[GENERATION_SCORING][sample].logprobs[position]
                initial = scores[NATIVE_SCORING][sample].logprobs[position]
                current = scores[drift_name][sample].logprobs[position]
                key = _probability_bucket(generation)
                bucket = per_bucket.setdefault(key, {"mismatch": [], "drift": []})
                mismatch_sq = (initial - generation) ** 2
                drift_sq = (current - initial) ** 2
                bucket["mismatch"].append(mismatch_sq)
                bucket["drift"].append(drift_sq)
                baseline_squared.append(mismatch_sq)
                drift_squared.append(drift_sq)
        result["probability_buckets"][str(update)] = {
            key: {
                "tokens": len(values["drift"]),
                "mismatch_mean_squared": float(np.mean(values["mismatch"])),
                "drift_mean_squared": float(np.mean(values["drift"])),
                "mismatch_over_drift": (
                    float(np.mean(values["mismatch"]) / np.mean(values["drift"]))
                    if np.mean(values["drift"]) > 0
                    else None
                ),
            }
            for key, values in per_bucket.items()
        }
        if drift_squared and np.mean(drift_squared) > 0:
            drift_mean_squares.append((update, float(np.mean(drift_squared))))
    if len(drift_mean_squares) >= 3 and baseline_squared:
        steps, values = zip(*drift_mean_squares, strict=True)
        exponent, log_coefficient = np.polyfit(np.log(steps), np.log(values), 1)
        if exponent > 0:
            mismatch_mean_squared = float(np.mean(baseline_squared))
            equivalent = math.exp((math.log(mismatch_mean_squared) - log_coefficient) / exponent)
            result["equivalent_updates"] = {
                "status": "estimated",
                "steps": float(equivalent),
                "exponent": float(exponent),
                "fit_updates": list(steps),
            }
    return result


def analyze_archive(uri: str, *, bootstrap_draws: int = BOOTSTRAP_DRAWS, tis_cap: float | None = None) -> dict:
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
        "input_commit_token": str(ReadView(uri).token),
        "bootstrap": {"seed": manifest.bootstrap_seed, "draws": bootstrap_draws, "cluster": "prompt_id"},
        "token_identity": identity,
        "comparisons": {},
        "paired_improvements": {},
        "breakdowns": {},
        "drift_scale": {},
        "route_diagnostics": {},
        "checks": {},
        "timing": json.loads(manifest.timing_json),
        "step_metrics": json.loads(manifest.step_metrics_json),
    }
    if identity["fraction"] != 1.0:
        report["checks"]["token_identity"] = "failed"
        return _finite_json(report)

    eps_clip_low, eps_clip_high = _clip_thresholds(manifest)
    report["route_diagnostics"] = _route_diagnostics(probes, scores, manifest.bootstrap_seed, bootstrap_draws)

    prompt_ids = [row.prompt_id for row in probes]
    masks = [row.loss_mask for row in probes]
    advantages = [row.advantage for row in probes]
    definitions = _comparison_definitions(scores)
    sampled_metrics = {}
    for label, (target_name, reference_name) in definitions.items():
        _require_same_weights_if_same_update(label, scores[target_name], scores[reference_name])
        target = [scores[target_name][row.sample_id].logprobs for row in probes]
        reference = [scores[reference_name][row.sample_id].logprobs for row in probes]

        def calculate(indices, *, target=target, reference=reference):
            return comparison_metrics(
                [target[index] for index in indices],
                [reference[index] for index in indices],
                [masks[index] for index in indices],
                advantages=[advantages[index] for index in indices],
                tis_cap=tis_cap,
                eps_clip_low=eps_clip_low,
                eps_clip_high=eps_clip_high,
            )

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
        report["breakdowns"][label] = _breakdowns(probes, scores[target_name], scores[reference_name])
        sampled_metrics[label] = bootstrap.draws

    baseline = "implementation_mismatch" if "implementation_mismatch" in sampled_metrics else "reread_mismatch"
    for mode in REPLAY_MODES:
        variant = f"{mode}_vs_generation" if baseline == "implementation_mismatch" else f"{mode}_vs_reread"
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

    positive_updates = sorted({score.update for rows in scores.values() for score in rows.values() if score.update > 0})
    for update in positive_updates:
        names = (f"trainer@{update}:native", NATIVE_SCORING, GENERATION_SCORING)
        if not all(name in scores for name in names):
            continue
        maximal_error = 0.0
        for probe in probes:
            current, initial, generated = (
                np.asarray(scores[name][probe.sample_id].logprobs, dtype=np.float64) for name in names
            )
            error = np.max(np.abs((current - generated) - ((current - initial) + (initial - generated))))
            maximal_error = max(maximal_error, float(error))
        report["checks"][f"drift_identity_after_{update}"] = {
            "max_abs_error": maximal_error,
            "pass": maximal_error <= 1e-12,
        }

    report["drift_scale"] = _drift_scale(probes, scores, positive_updates)

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
    tis_cap: float | None = None,
) -> dict:
    """Compare matched prompt groups from two runs with the same starting policy."""
    left = load_archive(left_uri)
    right = load_archive(right_uri)
    left_clip_low, left_clip_high = _clip_thresholds(left.manifest)
    right_clip_low, right_clip_high = _clip_thresholds(right.manifest)
    if left.manifest.starting_weights_hash != right.manifest.starting_weights_hash:
        raise ValueError("configuration A/B requires the same starting weight hash")
    if left.manifest.vllm_enforce_eager != right.manifest.vllm_enforce_eager:
        raise ValueError("configuration A/B requires the same vLLM execution mode")
    left_tokenizer = json.loads(left.manifest.software_json).get("tokenizer_fingerprint")
    right_tokenizer = json.loads(right.manifest.software_json).get("tokenizer_fingerprint")
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
    shared_tokens = left.manifest.probe_hash == right.manifest.probe_hash
    if shared_tokens and any(
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
        "kind": "shared_tokens" if shared_tokens else "independent_generation",
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

        def calculate(
            indices,
            *,
            left_target=left_target,
            left_reference=left_reference,
            right_target=right_target,
            right_reference=right_reference,
        ):
            def metrics(probes, scores, target_name, reference_name, clip_low, clip_high):
                return comparison_metrics(
                    [scores[target_name][probes[index].sample_id].logprobs for index in indices],
                    [scores[reference_name][probes[index].sample_id].logprobs for index in indices],
                    [probes[index].loss_mask for index in indices],
                    advantages=[probes[index].advantage for index in indices],
                    tis_cap=tis_cap,
                    eps_clip_low=clip_low,
                    eps_clip_high=clip_high,
                )

            left_values = metrics(left.probes, left.scores, left_target, left_reference, left_clip_low, left_clip_high)
            right_values = metrics(
                aligned_right_probes, right.scores, right_target, right_reference, right_clip_low, right_clip_high
            )
            return {f"{name}_left_minus_right": left_values[name] - right_values[name] for name in HEADLINE_METRICS}

        bootstrap = prompt_cluster_bootstrap(
            prompt_ids, calculate, seed=left.manifest.bootstrap_seed, draws=bootstrap_draws
        )
        result["comparisons"][label] = {"metrics": bootstrap.point, "ci95": bootstrap.intervals}
    return _finite_json(result)


def write_plots(report: dict, output_dir: Path) -> None:
    """Write static figures from the same archive-derived numbers as the report."""
    plt.switch_backend("Agg")

    comparisons = report.get("comparisons", {})
    if comparisons:
        names = list(comparisons)
        values = [comparisons[name]["metrics"]["abs_p99"] for name in names]
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 4))
        ax.bar(range(len(names)), values)
        ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
        ax.set_ylabel("p99 absolute log-probability gap")
        fig.tight_layout()
        fig.savefig(output_dir / "mismatch.png", dpi=160)
        plt.close(fig)
    routes = report.get("route_diagnostics", {})
    if routes:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, item in routes.items():
            layer_values = [
                (layer, item["metrics"][f"layer_{layer}/set_agreement"]) for layer in _layer_numbers(item["metrics"])
            ]
            if layer_values:
                ax.plot(
                    [index for index, _ in layer_values], [value for _, value in layer_values], marker="o", label=name
                )
        ax.set(xlabel="MoE layer", ylabel="expert-set agreement", ylim=(0, 1))
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(output_dir / "routes.png", dpi=160)
        plt.close(fig)


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
    ]
    if report["comparisons"]:
        lines.extend(
            [
                "| Comparison | p99 | 95% CI | k3 | share beyond 2x | route set agreement |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, item in report["comparisons"].items():
            metrics, ci = item["metrics"], item["ci95"]
            interval = ci.get("abs_p99", [math.nan, math.nan])
            route = item.get("route_set_agreement")
            route_text = "-" if route is None else f"{route['value']:.1%}"
            if route is not None and route["ci95"] is not None:
                lo, hi = route["ci95"]
                route_text += f" [{lo:.1%}, {hi:.1%}]"
            lines.append(
                f"| {name} | {metrics['abs_p99']:.5g} | [{interval[0]:.5g}, {interval[1]:.5g}] | "
                f"{metrics['k3']:.5g} | {metrics['share_beyond_2x']:.3%} | {route_text} |"
            )
        lines.append("")
        lines.extend(
            [
                "## Numerical details",
                "",
                "| Comparison | tokens | p50 | p75 | p90 | p99.9 | max | signed mean | "
                "raw token ESS | raw sequence ESS |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, item in report["comparisons"].items():
            metrics = item["metrics"]
            lines.append(
                f"| {name} | {metrics['tokens']} | {metrics['abs_p50']:.5g} | {metrics['abs_p75']:.5g} | "
                f"{metrics['abs_p90']:.5g} | {metrics['abs_p999']:.5g} | {metrics['abs_max']:.5g} | "
                f"{metrics['delta_mean']:.5g} | {metrics['token_ess_fraction_raw']:.3f} | "
                f"{metrics['sequence_ess_fraction_raw']:.3f} |"
            )
        lines.append("")
    if report.get("drift_scale"):
        lines.extend(["## Drift relative to mismatch", ""])
        equivalent = report["drift_scale"]["equivalent_updates"]
        if equivalent["status"] == "estimated":
            lines.append(f"Equivalent updates: {equivalent['steps']:.3g} from updates {equivalent['fit_updates']}.")
        else:
            lines.append("Equivalent updates: unavailable with fewer than three nonzero drift measurements.")
        lines.append("")
    if report["paired_improvements"]:
        lines.extend(["## Paired mode effects", ""])
        for mode, metrics in report["paired_improvements"].items():
            item = metrics["abs_p99"]
            lo, hi = item["ci95"]
            lines.append(f"- {mode}: native minus mode p99 |Δ| = {item['native_minus_mode']:.5g} [{lo:.5g}, {hi:.5g}].")
        lines.append("")
    if report["route_diagnostics"]:
        lines.extend(
            [
                "## Routing",
                "",
                "| Scoring | Valid coverage | Expert set agreement | Replacement rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, item in report["route_diagnostics"].items():
            metrics = item["metrics"]
            lines.append(
                f"| {name} | {metrics['valid_coverage']:.1%} | {metrics['set_agreement']:.1%} | "
                f"{metrics['replacement_fraction']:.1%} |"
            )
        lines.append("")
        lines.extend(
            [
                "| Scoring | Layer | Expert set agreement | Exact slot agreement | Replacement rate |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, item in report["route_diagnostics"].items():
            metrics = item["metrics"]
            for layer in _layer_numbers(metrics):
                lines.append(
                    f"| {name} | {layer} | {metrics[f'layer_{layer}/set_agreement']:.1%} | "
                    f"{metrics[f'layer_{layer}/exact_slot_agreement']:.1%} | "
                    f"{metrics[f'layer_{layer}/replacement_fraction']:.1%} |"
                )
        lines.append("")
    if report["timing"]:
        lines.extend(["## Timing", ""])
        for name, seconds in sorted(report["timing"].items()):
            if isinstance(seconds, (int, float)):
                lines.append(f"- {name}: {seconds:.3f} s")
        lines.append("")
    if report.get("step_metrics"):
        lines.extend(["## Step metrics and data movement", ""])
        for step, metrics in sorted(report["step_metrics"].items()):
            if not isinstance(metrics, dict) or not step.startswith("update@"):
                continue
            route_bytes = metrics.get("route_bytes", 0)
            token_count = metrics.get("token_count", 0)
            lines.append(
                f"- {step}: {token_count} valid tokens, {route_bytes} captured route bytes, "
                f"{metrics.get('optimizer_steps', 0)} optimizer steps."
            )
        lines.append("")
    return "\n".join(lines)


def render_archive_comparison(comparison: dict) -> str:
    lines = [
        "## Configuration comparison",
        "",
        f"{comparison['kind']} on matched prompt groups.",
        "",
        "| Comparison | p99 difference (left - right) | 95% CI |",
        "|---|---:|---:|",
    ]
    for label, item in comparison["comparisons"].items():
        key = "abs_p99_left_minus_right"
        value = item["metrics"][key]
        lo, hi = item["ci95"][key]
        lines.append(f"| {label} | {value:.5g} | [{lo:.5g}, {hi:.5g}] |")
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
    parser.add_argument("--tis-cap", type=float)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [analyze_archive(uri, bootstrap_draws=args.bootstrap_draws, tis_cap=args.tis_cap) for uri in args.archives]
    if len(reports) == 1:
        _write_single_report(reports[0], args.output_dir)
        return
    comparisons = [
        compare_archives(args.archives[0], uri, bootstrap_draws=args.bootstrap_draws, tis_cap=args.tis_cap)
        for uri in args.archives[1:]
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

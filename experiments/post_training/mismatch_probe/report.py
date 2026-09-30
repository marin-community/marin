# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate a mismatch report from FineStore archives without a live trainer."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Collection
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
# A second cache-off re-read in the same job: vLLM's own run-to-run floor.
RESCORE_AGAIN_SCORER = "vllm.rescore_again"
# A reused source archive's update-0 cache-off re-read, copied verbatim as the frozen prefill reference.
FROZEN_RESCORE_SCORER = "vllm.rescore_frozen"
TRAINER_SCORER = "trainer"
UPDATE_PREFIX = "update@"
TRAINING_PASS_PREFIX = "training_pass@"
CACHE_OFF = "off"

ANALYSIS_VERSION = 3
HEADLINE_METRICS = ("abs_p99", "k3", "abs_mean", "share_beyond_2x", "byte_equal_fraction")
PERCENT_METRICS = frozenset({"share_beyond_2x", "byte_equal_fraction"})
METRIC_TITLES = {
    "abs_p99": "p99 abs Δ",
    "k3": "k3",
    "abs_mean": "mean abs Δ",
    "share_beyond_2x": "beyond 2x",
    "byte_equal_fraction": "byte-equal",
}
# Metrics on which a candidate must not be worse than the kept stack on the prefill metric.
KEEP_RULE_METRICS = ("abs_mean", "abs_p99", "k3")
NATIVE_MODE = "native"
REREAD_REPLAY_MODE = "reread_replay"
REPLAY_MODE = "router_replay"
BOOTSTRAP_DRAWS = 1000
GENERATION_SCORING = f"{GENERATION_SCORER}@0"
RESCORE_AGAIN_SCORING = f"{RESCORE_AGAIN_SCORER}@0"
FROZEN_RESCORE_SCORING = f"{FROZEN_RESCORE_SCORER}@0"
TRAINER_SCORING_PREFIX = f"{TRAINER_SCORER}@"


def _trainer_scoring(update: int, mode: str) -> str:
    return f"{TRAINER_SCORING_PREFIX}{update}:{mode}"


def _vllm_scoring(scorer: str, update: int, cache_mode: str | None) -> str:
    label = f"{scorer}@{update}"
    return label if cache_mode in (None, CACHE_OFF) else f"{label}:{cache_mode}"


def _rescore_scoring(update: int, cache_mode: str = CACHE_OFF) -> str:
    return _vllm_scoring(RESCORE_SCORER, update, cache_mode)


NATIVE_SCORING = _trainer_scoring(0, NATIVE_MODE)
RESCORE_SCORING = _rescore_scoring(0)


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
    return _vllm_scoring(row.scorer, row.update, row.cache_mode)


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


def _update_zero_trainer_modes(scores: dict[str, dict[str, ScoreRow]]) -> list[str]:
    return sorted(
        {
            row.mode
            for rows in scores.values()
            for row in rows.values()
            if row.scorer == TRAINER_SCORER and row.update == 0
        }
    )


def _prefill_reference(names: Collection[str]) -> str | None:
    """Return the frozen source re-read when the job reuses a probe, else this job's cache-off re-read."""
    for label in (FROZEN_RESCORE_SCORING, RESCORE_SCORING):
        if label in names:
            return label
    return None


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
    add("trainer_determinism", _trainer_scoring(0, "native_again"), NATIVE_SCORING)
    add("replay_layout_floor", _trainer_scoring(0, "repeat_replay"), _trainer_scoring(0, "router_replay"))
    add("prompt_replay_effect", _trainer_scoring(0, "router_replay"), _trainer_scoring(0, "router_replay_response"))
    for mode in _trainer_modes(scores):
        add(f"{mode}_vs_generation", _trainer_scoring(0, mode), GENERATION_SCORING)
    prefill = _prefill_reference(names)
    if prefill is not None:
        for mode in _update_zero_trainer_modes(scores):
            add(f"{mode}_vs_reread", _trainer_scoring(0, mode), prefill)
    add("reread_noise", RESCORE_AGAIN_SCORING, RESCORE_SCORING)
    add("reread_vs_frozen", RESCORE_SCORING, FROZEN_RESCORE_SCORING)
    updates = sorted({row.update for rows in scores.values() for row in rows.values() if row.update > 0})
    for update in updates:
        add(f"trainer_drift_after_{update}", _trainer_scoring(update, NATIVE_MODE), NATIVE_SCORING)
        add(f"observed_gap_after_{update}", _trainer_scoring(update, NATIVE_MODE), GENERATION_SCORING)
        add(f"vllm_drift_after_{update}", reread(update), reread(0))
        add(f"mismatch_after_{update}", _trainer_scoring(update, NATIVE_MODE), reread(update))
        for mode in _trainer_modes(scores):
            add(f"{mode}_after_{update}", _trainer_scoring(update, mode), reread(update))
            # Same mode before and after the update: under replay the routes are fixed, so this isolates
            # the weight change from routing flips.
            add(f"{mode}_drift_after_{update}", _trainer_scoring(update, mode), _trainer_scoring(0, mode))
            if update > 1:
                add(f"{mode}_step_drift_{update}", _trainer_scoring(update, mode), _trainer_scoring(update - 1, mode))
    return comparisons


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def _decoded_routes(score: ScoreRow) -> np.ndarray:
    return np.frombuffer(score.expert_choices, dtype=score.expert_choices_dtype).reshape(score.expert_choices_shape)


def _same_expert_sets(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (np.sort(left, axis=-1) == np.sort(right, axis=-1)).all(axis=-1)


def _has_routes(sample_scores: dict[str, ScoreRow]) -> bool:
    return all(score.expert_choices is not None for score in sample_scores.values())


def _route_counts(probes: list[ProbeRow], sample_scores: dict[str, ScoreRow]) -> tuple[np.ndarray, np.ndarray]:
    counts, matches = [], []
    for probe in probes:
        source = np.frombuffer(probe.routed_experts, dtype=probe.routed_experts_dtype).reshape(
            probe.routed_experts_shape
        )
        score = sample_scores[probe.sample_id]
        if score.expert_choices_shape != list(source.shape):
            raise ValueError(f"route observation shape differs from capture for {probe.sample_id}")
        actual = _decoded_routes(score)
        valid = np.asarray(probe.route_valid_mask, dtype=np.bool_) & np.asarray(probe.loss_mask, dtype=np.bool_)[:, None]
        if np.any(actual[valid] < 0):
            raise ValueError(f"missing trainer route observation for {probe.sample_id}")
        equal = _same_expert_sets(source, actual)
        counts.append(valid.sum(axis=0))
        matches.append((equal & valid).sum(axis=0))
    return np.asarray(counts), np.asarray(matches)


def _reread_routes(score: ScoreRow, probe: ProbeRow) -> np.ndarray:
    """Decode a re-read's ``[prompt + response - 1, layer, expert]`` routes, one row per input position."""
    routes = _decoded_routes(score)
    positions = len(probe.prompt_token_ids) + len(probe.vllm_output_ids) - 1
    if routes.ndim != 3 or routes.shape[0] != positions:
        raise ValueError(f"re-read routes for {probe.sample_id} do not cover prompt and response inputs")
    return routes


def _reread_response_route_counts(
    probes: list[ProbeRow], trainer_scores: dict[str, ScoreRow], reread_scores: dict[str, ScoreRow]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count per-layer expert-set matches on response inputs and each token's first disagreeing layer.

    Re-read rows ``P .. P + R - 2`` hold response tokens ``0 .. R - 2`` as inputs, the indexing of the
    trainer's ``[R, layer, expert]`` observations; the final response token is never an input.
    The last column of the first-disagreement counts holds tokens whose sets agree at every layer.
    """
    counts, matches, first = [], [], []
    for probe in probes:
        reread = _reread_routes(reread_scores[probe.sample_id], probe)
        prompt_length, response_length = len(probe.prompt_token_ids), len(probe.vllm_output_ids)
        layers = reread.shape[1]
        score = trainer_scores[probe.sample_id]
        if score.expert_choices_shape != [response_length, *reread.shape[1:]]:
            raise ValueError(f"trainer route observation shape differs from the re-read for {probe.sample_id}")
        actual = _decoded_routes(score)[: response_length - 1]
        valid = np.asarray(probe.loss_mask[: response_length - 1], dtype=np.bool_)
        if np.any(actual[valid] < 0):
            raise ValueError(f"missing trainer route observation for {probe.sample_id}")
        disagreeing = ~_same_expert_sets(reread[prompt_length:], actual)[valid]
        first_layer = np.where(disagreeing.any(axis=1), disagreeing.argmax(axis=1), layers)
        counts.append(np.full(layers, int(valid.sum())))
        matches.append((~disagreeing).sum(axis=0))
        first.append(np.bincount(first_layer, minlength=layers + 1))
    return np.asarray(counts), np.asarray(matches), np.asarray(first)


def _vllm_route_counts(
    probes: list[ProbeRow], target_scores: dict[str, ScoreRow], reference_scores: dict[str, ScoreRow]
) -> tuple[np.ndarray, np.ndarray]:
    """Count per-layer expert-set matches between two re-reads at every input position."""
    counts, matches = [], []
    for probe in probes:
        target = _reread_routes(target_scores[probe.sample_id], probe)
        reference = _reread_routes(reference_scores[probe.sample_id], probe)
        if target.shape != reference.shape:
            raise ValueError(f"re-read route shapes differ for {probe.sample_id}")
        counts.append(np.full(target.shape[1], target.shape[0]))
        matches.append(_same_expert_sets(target, reference).sum(axis=0))
    return np.asarray(counts), np.asarray(matches)


def _route_statistics(indices: list[int], counts: np.ndarray, matches: np.ndarray) -> dict[str, float]:
    layer_counts, layer_matches = counts[indices].sum(axis=0), matches[indices].sum(axis=0)
    total = int(layer_counts.sum())
    values = {"set_agreement": float(layer_matches.sum() / total) if total else math.nan}
    for layer, count in enumerate(layer_counts):
        values[f"layer_{layer}/set_agreement"] = float(layer_matches[layer] / count) if count else math.nan
    return values


def _route_agreement(probes: list[ProbeRow], counts: np.ndarray, matches: np.ndarray, seed: int, draws: int) -> dict:
    bootstrap = prompt_cluster_bootstrap(
        [row.prompt_id for row in probes],
        lambda indices: _route_statistics(indices, counts, matches),
        seed=seed,
        draws=draws,
    )
    return {
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


def _route_diagnostics(probes: list[ProbeRow], scores: dict[str, dict[str, ScoreRow]], seed: int, draws: int) -> dict:
    """Compare expert sets on captured, unmasked token-and-layer rows."""
    result = {}
    if any(row.routed_experts is None for row in probes):
        return result
    for name, sample_scores in scores.items():
        if any(score.scorer != TRAINER_SCORER for score in sample_scores.values()) or not _has_routes(sample_scores):
            continue
        counts, matches = _route_counts(probes, sample_scores)
        result[name] = _route_agreement(probes, counts, matches, seed, draws)
    return result


def _reread_route_diagnostics(
    probes: list[ProbeRow], scores: dict[str, dict[str, ScoreRow]], seed: int, draws: int
) -> dict:
    """Compare each update-0 trainer mode's response routes with the prefill reference's routes."""
    result = {}
    reference = _prefill_reference(scores)
    if reference is None or not _has_routes(scores[reference]):
        return result
    for mode in _update_zero_trainer_modes(scores):
        name = _trainer_scoring(0, mode)
        if not _has_routes(scores[name]):
            continue
        counts, matches, first = _reread_response_route_counts(probes, scores[name], scores[reference])
        totals = first.sum(axis=0)
        result[name] = {
            "reference": reference,
            **_route_agreement(probes, counts, matches, seed, draws),
            "first_disagreeing_layer": {
                "tokens": int(totals.sum()),
                "no_disagreement": int(totals[-1]),
                "layers": {str(layer): int(count) for layer, count in enumerate(totals[:-1])},
            },
        }
    return result


def _vllm_route_diagnostics(
    probes: list[ProbeRow], scores: dict[str, dict[str, ScoreRow]], seed: int, draws: int
) -> dict:
    """Compare vLLM re-read routes with each other at every prompt and response input position."""
    result = {}
    for label, target, reference in (
        ("reread_noise", RESCORE_AGAIN_SCORING, RESCORE_SCORING),
        ("reread_vs_frozen", RESCORE_SCORING, FROZEN_RESCORE_SCORING),
    ):
        if target not in scores or reference not in scores:
            continue
        if not (_has_routes(scores[target]) and _has_routes(scores[reference])):
            continue
        counts, matches = _vllm_route_counts(probes, scores[target], scores[reference])
        result[label] = {
            "target": target,
            "reference": reference,
            **_route_agreement(probes, counts, matches, seed, draws),
        }
    return result


def _paired_intervals(
    comparisons: dict, sampled_metrics: dict, baseline: str, candidate: str, metrics: tuple[str, ...]
) -> dict[str, dict]:
    """Return baseline minus candidate per metric with a paired 95% interval.

    Both comparisons resample the same prompt clusters with the same seed, so draw ``i`` pairs them.
    """
    paired = {}
    for metric in metrics:
        draws = np.asarray(
            [
                left[metric] - right[metric]
                for left, right in zip(sampled_metrics[baseline], sampled_metrics[candidate], strict=True)
            ]
        )
        point = comparisons[baseline]["metrics"][metric] - comparisons[candidate]["metrics"][metric]
        # Draws without a scorable token are NaN and carry no information about the difference.
        paired[metric] = {
            "baseline_minus_candidate": float(point),
            "ci95": np.nanpercentile(draws, [2.5, 97.5]).tolist(),
        }
    return paired


def _vllm_stable_masks(probes: list[ProbeRow], scores: dict[str, dict[str, ScoreRow]]) -> list[list[bool]] | None:
    """Loss masks limited to tokens where every update-0 cache-off vLLM re-read is bit-identical.

    vLLM's re-read is not reproducible on every token; these tokens isolate the trainer's own mismatch.
    Returns ``None`` when the archive holds fewer than two re-reads.
    """
    names = [
        name
        for name in (
            _rescore_scoring(0),
            _vllm_scoring(RESCORE_AGAIN_SCORER, 0, CACHE_OFF),
            _vllm_scoring(FROZEN_RESCORE_SCORER, 0, CACHE_OFF),
        )
        if name in scores
    ]
    if len(names) < 2:
        return None
    masks = []
    for row in probes:
        values = [np.asarray(scores[name][row.sample_id].logprobs, dtype=np.float32) for name in names]
        stable = np.all([value == values[0] for value in values[1:]], axis=0)
        masks.append((np.asarray(row.loss_mask, dtype=np.bool_) & stable).tolist())
    return masks


def _reference_distribution(reference: str, prefill: str | None) -> str:
    if reference == GENERATION_SCORING:
        return "generation"
    if reference == prefill:
        return "prefill_re_read"
    return "diagnostic_re_read_or_trainer"


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


def analyze_archive(uri: str, *, bootstrap_draws: int = BOOTSTRAP_DRAWS, kept_numerics: str | None = None) -> dict:
    """Analyze one archive; ``kept_numerics`` adds paired tables against replay under that numerics set."""
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
        "prefill_reference": _prefill_reference(scores),
        "comparisons": {},
        "paired_improvements": {},
        "paired_vs_reread": {},
        "paired_vs_generation": {},
        "paired_vs_reread_stable": {},
        "vllm_stable_token_fraction": None,
        "route_diagnostics": {},
        "route_diagnostics_vs_reread": {},
        "reread_route_agreement": {},
        "checks": {},
        "timing": json.loads(manifest.timing_json),
        "step_metrics": json.loads(manifest.step_metrics_json),
    }
    if identity["fraction"] != 1.0:
        report["checks"]["token_identity"] = "failed"
        return _finite_json(report)

    report["route_diagnostics"] = _route_diagnostics(probes, scores, manifest.bootstrap_seed, bootstrap_draws)
    report["route_diagnostics_vs_reread"] = _reread_route_diagnostics(
        probes, scores, manifest.bootstrap_seed, bootstrap_draws
    )
    report["reread_route_agreement"] = _vllm_route_diagnostics(probes, scores, manifest.bootstrap_seed, bootstrap_draws)

    prefill = report["prefill_reference"]
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
            "reference_distribution": _reference_distribution(reference_name, prefill),
        }
        if label in report["reread_route_agreement"]:
            route, route_reference = report["reread_route_agreement"][label], "re-read"
        elif reference_name == prefill:
            route, route_reference = report["route_diagnostics_vs_reread"].get(target_name), "re-read"
        else:
            route, route_reference = report["route_diagnostics"].get(target_name), "generation"
        if route is not None:
            report["comparisons"][label]["route_set_agreement"] = {
                "value": route["metrics"]["set_agreement"],
                "ci95": route["ci95"].get("set_agreement"),
                "routes": route_reference,
            }
        sampled_metrics[label] = bootstrap.draws

    stable_masks = _vllm_stable_masks(probes, scores) if prefill is not None else None
    if stable_masks is not None and any(any(mask) for mask in stable_masks):
        scored = sum(sum(row.loss_mask) for row in probes)
        report["vllm_stable_token_fraction"] = sum(sum(mask) for mask in stable_masks) / scored
        for mode in _update_zero_trainer_modes(scores):
            target_name = _trainer_scoring(0, mode)
            rows = ComparisonRows(
                target=[scores[target_name][row.sample_id].logprobs for row in probes],
                reference=[scores[prefill][row.sample_id].logprobs for row in probes],
                masks=stable_masks,
            )
            point = rows.metrics(list(range(len(probes))))

            # A resample can draw only prompts with no stable token; it contributes no interval draw.
            def calculate(indices, *, rows=rows, keys=tuple(point)):
                if not any(any(rows.masks[index]) for index in indices):
                    return dict.fromkeys(keys, math.nan)
                return rows.metrics(indices)

            bootstrap = prompt_cluster_bootstrap(
                prompt_ids, calculate, seed=manifest.bootstrap_seed, draws=bootstrap_draws
            )
            label = f"{mode}_vs_reread_stable"
            report["comparisons"][label] = {
                "target": target_name,
                "reference": prefill,
                "metrics": bootstrap.point,
                "ci95": bootstrap.intervals,
                "reference_distribution": "prefill_re_read_vllm_stable_tokens",
            }
            sampled_metrics[label] = bootstrap.draws

    baseline = "implementation_mismatch"
    for mode in _trainer_modes(scores):
        variant = f"{mode}_vs_generation"
        if baseline not in sampled_metrics or variant not in sampled_metrics:
            continue
        report["paired_improvements"][mode] = _paired_intervals(
            report["comparisons"], sampled_metrics, baseline, variant, HEADLINE_METRICS
        )

    kept_mode = REREAD_REPLAY_MODE if f"{REREAD_REPLAY_MODE}_vs_reread" in sampled_metrics else NATIVE_MODE
    kept_suffix = f"+{kept_numerics}" if kept_numerics else ""
    for prefill_kept in dict.fromkeys((kept_mode, f"{REREAD_REPLAY_MODE}{kept_suffix}")):
        kept = f"{prefill_kept}_vs_reread"
        if kept not in sampled_metrics:
            continue
        candidates = {}
        for mode in _update_zero_trainer_modes(scores):
            if mode == prefill_kept or f"{mode}_vs_reread" not in sampled_metrics:
                continue
            paired = _paired_intervals(
                report["comparisons"], sampled_metrics, kept, f"{mode}_vs_reread", KEEP_RULE_METRICS
            )
            candidates[mode] = {
                "metrics": paired,
                # Worse than the kept stack: some interval of kept minus candidate lies entirely below zero.
                "regresses": any(item["ci95"][1] < 0 for item in paired.values()),
            }
        report["paired_vs_reread"][prefill_kept] = candidates
    if kept_suffix:
        generation_kept = f"{REPLAY_MODE}{kept_suffix}"
        kept = f"{generation_kept}_vs_generation"
        if kept in sampled_metrics:
            candidates = {}
            for mode in _update_zero_trainer_modes(scores):
                variant = f"{mode}_vs_generation"
                if mode == generation_kept or not mode.startswith(REPLAY_MODE) or variant not in sampled_metrics:
                    continue
                paired = _paired_intervals(report["comparisons"], sampled_metrics, kept, variant, KEEP_RULE_METRICS)
                candidates[mode] = {
                    "metrics": paired,
                    "regresses": any(item["ci95"][1] < 0 for item in paired.values()),
                }
            report["paired_vs_generation"][generation_kept] = candidates
    kept_stable = f"{kept_mode}_vs_reread_stable"
    if kept_stable in sampled_metrics:
        candidates = {}
        for mode in _update_zero_trainer_modes(scores):
            if mode == kept_mode:
                continue
            paired = _paired_intervals(
                report["comparisons"], sampled_metrics, kept_stable, f"{mode}_vs_reread_stable", KEEP_RULE_METRICS
            )
            candidates[mode] = {"metrics": paired, "regresses": any(item["ci95"][1] < 0 for item in paired.values())}
        report["paired_vs_reread_stable"][kept_mode] = candidates

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


def _training_pass_lines(timing: dict) -> list[str]:
    """Render timed forward+backward passes: median of the repetitions (slowest rank each) against native."""
    passes: dict[tuple[str, str], dict] = {}
    for name, value in timing.items():
        if not name.startswith(TRAINING_PASS_PREFIX):
            continue
        label, metric = name.rsplit("/", 1)
        update, mode = label.removeprefix(TRAINING_PASS_PREFIX).split(":", 1)
        passes.setdefault((update, mode), {})[metric] = value
    if not passes:
        return []
    lines = [
        "## Training-pass timing",
        "",
        "Forward and backward on the probe batch at zero learning rate, one warmup, median of the timed "
        "repetitions (each the slowest rank). Change is against `native` at the same update.",
        "",
        "| Update | Mode | seconds (median) | repetitions | change vs native | peak memory GiB |",
        "|---|---|---:|---|---:|---:|",
    ]
    for (update, mode), metrics in sorted(passes.items()):
        seconds = metrics.get("seconds", [])
        median = float(np.median(seconds)) if seconds else math.nan
        native = passes.get((update, "native"), {}).get("seconds", [])
        change = median / float(np.median(native)) - 1 if native else math.nan
        repetitions = ", ".join(f"{value:.4g}" for value in seconds)
        memory = metrics.get("peak_memory_bytes", math.nan) / 2**30
        lines.append(
            f"| {update} | {mode} | {_display_metric(median)} | {repetitions} | "
            f"{_display_metric(change, percent_digits=1)} | {_display_metric(memory)} |"
        )
    lines.append("")
    return lines


def _display_interval(value, *, percent_digits: int | None = None, scale: float = 1.0) -> str:
    if value is None:
        return "-"
    low, high = value
    return (
        f"[{_display_metric(low, percent_digits=percent_digits, scale=scale)}, "
        f"{_display_metric(high, percent_digits=percent_digits, scale=scale)}]"
    )


def _percent_digits(metric: str) -> int | None:
    return 3 if metric in PERCENT_METRICS else None


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
    if report["prefill_reference"] is not None:
        lines.extend([f"Prefill reference: `{report['prefill_reference']}`.", ""])
    if report["comparisons"]:
        headers = " | ".join(f"{METRIC_TITLES[metric]} | 95% CI" for metric in HEADLINE_METRICS)
        lines.extend(
            [
                "Δ is target minus reference log probability on masked response tokens. "
                "P99 is the 99th percentile of its absolute value.",
                "Replay uses routes captured during the original generation; later updates measure stale-route replay.",
                "Route agreement is against the re-read's routes for comparisons with a re-read reference "
                "and against generation routes otherwise.",
                f"95% intervals resample whole prompts {report['bootstrap']['draws']} times "
                f"with seed {report['bootstrap']['seed']}.",
                "",
                f"| Comparison | {headers} | route agreement | 95% CI |",
                "|---|" + "---:|---:|" * len(HEADLINE_METRICS) + "---:|---:|",
            ]
        )
        for name, item in report["comparisons"].items():
            metrics, ci = item["metrics"], item["ci95"]
            route = item.get("route_set_agreement")
            route_text = "-" if route is None else _display_metric(route["value"], percent_digits=1)
            route_ci = "-" if route is None else _display_interval(route["ci95"], percent_digits=1)
            cells = " | ".join(
                f"{_display_metric(metrics[metric], percent_digits=_percent_digits(metric))} | "
                f"{_display_interval(ci.get(metric), percent_digits=_percent_digits(metric))}"
                for metric in HEADLINE_METRICS
            )
            lines.append(f"| {name} | {cells} | {route_text} | {route_ci} |")
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
                "Positive native minus mode means a smaller mismatch in that mode against generation.",
                "",
                "| Mode | p99 improvement | 95% paired CI |",
                "|---|---:|---:|",
            ]
        )
        for mode, item in report["paired_improvements"].items():
            effect = item["abs_p99"]
            lines.append(
                f"| {mode} | {_display_metric(effect['baseline_minus_candidate'])} | "
                f"{_display_interval(effect['ci95'])} |"
            )
        lines.append("")
    paired_tables = [(kept, candidates, "") for kept, candidates in report["paired_vs_reread"].items()] + [
        (
            kept,
            candidates,
            f" Only tokens where every vLLM re-read is bit-identical count "
            f"({_display_metric(report['vllm_stable_token_fraction'], percent_digits=1)} of scored tokens).",
        )
        for kept, candidates in report["paired_vs_reread_stable"].items()
    ]
    paired_tables += [
        (kept, candidates, "generation") for kept, candidates in report.get("paired_vs_generation", {}).items()
    ]
    for kept, candidates, scope in paired_tables:
        if scope == "generation":
            headers = " | ".join(f"{METRIC_TITLES[metric]} | 95% paired CI" for metric in KEEP_RULE_METRICS)
            lines.extend(
                [
                    f"## Generation metric: `{kept}` minus candidate",
                    "",
                    "Both sides are scored against vLLM's generation-time log probabilities. "
                    f"Positive means the candidate is closer to vLLM than `{kept}`. "
                    "A candidate regresses when any interval lies entirely below zero.",
                    "",
                    f"| Candidate | {headers} | regresses |",
                    "|---|" + "---:|---:|" * len(KEEP_RULE_METRICS) + "---|",
                ]
            )
            for mode, item in candidates.items():
                cells = " | ".join(
                    f"{_display_metric(item['metrics'][metric]['baseline_minus_candidate'])} | "
                    f"{_display_interval(item['metrics'][metric]['ci95'])}"
                    for metric in KEEP_RULE_METRICS
                )
                lines.append(f"| {mode} | {cells} | {'**yes**' if item['regresses'] else 'no'} |")
            lines.append("")
            continue
        headers = " | ".join(f"{METRIC_TITLES[metric]} | 95% paired CI" for metric in KEEP_RULE_METRICS)
        lines.extend(
            [
                f"## Prefill metric{' on vLLM-stable tokens' if scope else ''}: `{kept}` minus candidate",
                "",
                f"Both sides are scored against `{report['prefill_reference']}`. "
                f"Positive means the candidate is closer to the re-read than `{kept}`. "
                f"A candidate regresses when any interval lies entirely below zero.{scope}",
                "",
                f"| Candidate | {headers} | regresses |",
                "|---|" + "---:|---:|" * len(KEEP_RULE_METRICS) + "---|",
            ]
        )
        for mode, item in candidates.items():
            cells = " | ".join(
                f"{_display_metric(item['metrics'][metric]['baseline_minus_candidate'])} | "
                f"{_display_interval(item['metrics'][metric]['ci95'])}"
                for metric in KEEP_RULE_METRICS
            )
            lines.append(f"| {mode} | {cells} | {'**yes**' if item['regresses'] else 'no'} |")
        lines.append("")
    for title, routes in (
        ("Routes by layer against generation", report["route_diagnostics"]),
        ("Routes by layer against the re-read (response inputs)", report["route_diagnostics_vs_reread"]),
        ("vLLM re-read route agreement (all inputs)", report["reread_route_agreement"]),
    ):
        if not routes:
            continue
        lines.extend([f"## {title}", "", "| Scorer | Layer | Expert-set agreement | 95% CI |", "|---|---:|---:|---:|"])
        for name, item in routes.items():
            layers = [("all", item), *item["layers"].items()]
            for layer, details in layers:
                lines.append(
                    f"| {name} | {layer} | {_display_metric(details['metrics']['set_agreement'], percent_digits=1)} | "
                    f"{_display_interval(details['ci95'].get('set_agreement'), percent_digits=1)} |"
                )
        lines.append("")
    if report["route_diagnostics_vs_reread"]:
        lines.extend(
            [
                "## First disagreeing layer against the re-read",
                "",
                "For each masked response input, the lowest layer whose expert set differs from the re-read's.",
                "",
                "| Scorer | First disagreeing layer | tokens | share |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, item in report["route_diagnostics_vs_reread"].items():
            histogram = item["first_disagreeing_layer"]
            bins = [(layer, count) for layer, count in histogram["layers"].items() if count]
            bins.append(("none", histogram["no_disagreement"]))
            for layer, count in bins:
                share = count / histogram["tokens"] if histogram["tokens"] else math.nan
                lines.append(f"| {name} | {layer} | {count} | {_display_metric(share, percent_digits=1)} |")
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
        lines.extend(_training_pass_lines(report["timing"]))
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
    parser.add_argument("--kept-numerics", help="Numerics set of the kept stack, e.g. compiled_stack.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [
        analyze_archive(uri, bootstrap_draws=args.bootstrap_draws, kept_numerics=args.kept_numerics)
        for uri in args.archives
    ]
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

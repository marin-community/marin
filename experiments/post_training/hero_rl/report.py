# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the historical Hero 4K numerical gate to a frozen mismatch archive."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from finestore.rl.mismatch_probe import ProbeRow, ScoreRow

from experiments.post_training.mismatch_probe.report import GENERATION_SCORING, ArchiveData, load_archive

REPLAY_SCORING = "trainer@0:router_replay"
MINIMUM_TOKENS = 50_000
MAXIMUM_OUTSIDE_FRACTION = 0.001
MINIMUM_RATIO = 0.8
MAXIMUM_RATIO = 1.2
MAXIMUM_CONTEXT = 4096
HERO_LAYERS = 48
HERO_EXPERTS_PER_TOKEN = 8
NUMERICAL_SCOPE = (
    "Initial-weight, sampled-vLLM versus captured-route trainer numerical qualification at 4K only. "
    "Actual PPO clipping, learning, publication, and checkpoint restore require separate live gates."
)


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INVALID = "INVALID"


@dataclass(frozen=True)
class GateProvenance:
    archive_commit_token: str
    probe_hash: str
    checkpoint_path: str
    runtime_commit: str | None
    tokenizer_fingerprint: str
    starting_global_step: int
    model_source: str | None
    model_source_identity: str | None


@dataclass(frozen=True)
class GateCounts:
    independent_responses: int
    prompts: int
    response_tokens: int
    ratios_outside_interval: int
    outside_fraction: float


@dataclass
class NumericalGateReport:
    status: GateStatus
    archive: str
    expected_model_source: str
    expected_starting_step: int
    reasons: list[str]
    provenance: GateProvenance | None = None
    counts: GateCounts | None = None
    scope: str = NUMERICAL_SCOPE
    minimum_tokens: int = MINIMUM_TOKENS
    maximum_outside_fraction: float = MAXIMUM_OUTSIDE_FRACTION
    ratio_interval: tuple[float, float] = (MINIMUM_RATIO, MAXIMUM_RATIO)


def _validate_routes(probe: ProbeRow, score: ScoreRow, mask: np.ndarray) -> None:
    if probe.routed_experts is None or probe.routed_experts_shape is None or score.expert_choices is None:
        raise ValueError(f"Missing captured or executed replay routes for {probe.sample_id}")
    if score.expert_choices_shape != probe.routed_experts_shape:
        raise ValueError(f"Captured and executed route shapes differ for {probe.sample_id}")
    if probe.routed_experts_shape[1:] != [HERO_LAYERS, HERO_EXPERTS_PER_TOKEN]:
        raise ValueError(f"Hero replay must capture all 48 layers and eight expert choices for {probe.sample_id}")
    captured = np.frombuffer(probe.routed_experts, dtype=probe.routed_experts_dtype).reshape(probe.routed_experts_shape)
    executed = np.frombuffer(score.expert_choices, dtype=score.expert_choices_dtype).reshape(score.expert_choices_shape)
    valid = np.asarray(probe.route_valid_mask, dtype=bool)
    if not captured.shape[1] or not captured.shape[2] or not valid[mask].all():
        raise ValueError(f"Incomplete captured routes on qualified tokens for {probe.sample_id}")
    if not np.array_equal(captured[mask], executed[mask]):
        raise ValueError(f"Executed routes differ from captured routes for {probe.sample_id}")


def _counts(archive: ArchiveData) -> GateCounts:
    if REPLAY_SCORING not in archive.scores:
        raise ValueError("Archive has no initial-weight captured-route trainer scoring")
    seen_requests = set()
    tokens = outside = 0
    lower, upper = math.log(MINIMUM_RATIO), math.log(MAXIMUM_RATIO)
    for probe in archive.probes:
        if probe.prompt_token_ids != probe.trainer_prompt_ids or probe.vllm_output_ids != probe.trainer_input_ids:
            raise ValueError(f"Trainer and sampled token identity differ for {probe.sample_id}")
        if len(probe.prompt_token_ids) + len(probe.vllm_output_ids) > MAXIMUM_CONTEXT:
            raise ValueError(f"Frozen response exceeds the 4K context for {probe.sample_id}")
        request = (tuple(probe.prompt_token_ids), probe.request_seed)
        if request in seen_requests:
            raise ValueError("Repeated prompt and request seed cannot count as an independent response")
        seen_requests.add(request)
        response_mask = np.asarray(probe.response_mask, dtype=bool)
        mask = np.asarray(probe.loss_mask, dtype=bool)
        if np.any(mask & ~response_mask):
            raise ValueError(f"Loss mask includes non-response tokens for {probe.sample_id}")
        replay = archive.scores[REPLAY_SCORING][probe.sample_id]
        generation = archive.scores[GENERATION_SCORING][probe.sample_id]
        for score in (replay, generation):
            if score.update != 0 or score.global_step != archive.manifest.starting_global_step:
                raise ValueError("Initial sampled and trainer scores differ from the manifest starting step")
        _validate_routes(probe, replay, mask)
        delta = np.asarray(replay.logprobs, dtype=np.float64) - np.asarray(generation.logprobs, dtype=np.float64)
        selected = delta[mask]
        tokens += int(mask.sum())
        outside += int(np.count_nonzero((selected < lower) | (selected > upper)))
    if not tokens:
        raise ValueError("Archive contains no loss-bearing response tokens")
    return GateCounts(
        independent_responses=len(seen_requests),
        prompts=len({tuple(probe.prompt_token_ids) for probe in archive.probes}),
        response_tokens=tokens,
        ratios_outside_interval=outside,
        outside_fraction=outside / tokens,
    )


def qualify_archive(uri: str, *, expected_model_source: str, expected_starting_step: int) -> NumericalGateReport:
    """Check complete initial-weight evidence without counting later or repeated scorings."""
    report = NumericalGateReport(GateStatus.INVALID, uri, expected_model_source, expected_starting_step, [])
    try:
        archive = load_archive(uri)
        manifest = archive.manifest
        config = json.loads(manifest.config_json)
        model = config["trainer"]["policy"]["model"]
        report.provenance = GateProvenance(
            archive.commit_token,
            manifest.probe_hash,
            manifest.checkpoint_path,
            manifest.runtime_commit,
            manifest.tokenizer_fingerprint,
            manifest.starting_global_step,
            model.get("source_uri"),
            model.get("source_identity"),
        )
        if manifest.starting_global_step != expected_starting_step:
            raise ValueError("Manifest starting step differs from the requested checkpoint step")
        if (
            dict(zip(manifest.scored_updates, manifest.scored_global_steps, strict=True)).get(0)
            != expected_starting_step
        ):
            raise ValueError("Manifest has no matching initial-weight scoring step")
        if model.get("source_uri") != expected_model_source or not model.get("source_identity"):
            raise ValueError("Manifest policy source differs from the requested completed export or lacks identity")
        if manifest.checkpoint_path != model.get("path"):
            raise ValueError("Manifest loaded checkpoint differs from the declared initial policy export")
        if manifest.vllm_enforce_eager or not manifest.tokenizer_fingerprint:
            raise ValueError("Numerical qualification requires non-eager vLLM and a tokenizer fingerprint")
        report.counts = _counts(archive)
    except (ValueError, OSError, KeyError, TypeError) as error:
        report.reasons.append(str(error))
        return report
    if report.counts.response_tokens < MINIMUM_TOKENS:
        report.reasons.append(
            f"Only {report.counts.response_tokens} qualified tokens; require at least {MINIMUM_TOKENS}"
        )
    if report.counts.outside_fraction >= MAXIMUM_OUTSIDE_FRACTION:
        report.reasons.append("Probability-ratio outlier fraction must be strictly below 0.001")
    report.status = GateStatus.FAIL if report.reasons else GateStatus.PASS
    return report


def render_markdown(report: NumericalGateReport) -> str:
    lines = [
        f"# Hero 4K numerical gate: {report.status}",
        "",
        report.scope,
        "",
        "Require at least 50,000 qualified response tokens and fewer than 0.1% of ratios outside [0.8, 1.2].",
        "",
        f"Archive: `{report.archive}`",
        "",
    ]
    if report.counts is not None:
        counts = report.counts
        lines.extend(
            [
                "| Evidence | Count |",
                "|---|---:|",
                f"| Independent responses | {counts.independent_responses} |",
                f"| Prompts | {counts.prompts} |",
                f"| Qualified response tokens | {counts.response_tokens} |",
                f"| Ratios outside [0.8, 1.2] | {counts.ratios_outside_interval} |",
                f"| Outlier fraction | {counts.outside_fraction:.8g} |",
                "",
            ]
        )
    lines.extend(f"- {reason}" for reason in report.reasons)
    if report.provenance is not None:
        lines.extend(["", "Provenance:", "", "```json", json.dumps(asdict(report.provenance), indent=2), "```"])
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--expected-model-source", required=True)
    parser.add_argument("--expected-starting-step", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = qualify_archive(
        args.archive,
        expected_model_source=args.expected_model_source,
        expected_starting_step=args.expected_starting_step,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(asdict(report), indent=2, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(render_markdown(report))
    return {GateStatus.PASS: 0, GateStatus.FAIL: 1, GateStatus.INVALID: 2}[report.status]


if __name__ == "__main__":
    raise SystemExit(main())

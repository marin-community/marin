# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize task-eval signal-to-noise ratios for the two-phase many-domain study."""

from __future__ import annotations

import io
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.two_phase_many_observed_runs import load_original_qsplit240_with_core_baselines

ROOT = Path(__file__).resolve().parents[4]
EXPLORATORY_DIR = ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many"
OUTPUT_CSV = EXPLORATORY_DIR / "eval_signal_to_noise_ranked.csv"

COMMON_SWARM_PATH = EXPLORATORY_DIR / "two_phase_many.csv"

QSPLIT240_OVERLAP_RESULTS_GLOB = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_qsplit240_olmo_base_easy_overlap_rerun/**/collect_results*/results.csv"
)
RUN00097_OVERLAP_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_run00097_seed_study_olmo_base_easy_overlap_rerun/collect_results-83df1e/results.csv"
)
QSPLIT240_SL_VERB_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_qsplit240_mmlu_sl_verb_rerun/collect_results-ef2602/results.csv"
)
RUN00097_STANDARD_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_seed_study/collect_results-ca72ba/results.csv"
)
RUN00097_STANDARD_BACKFILL_RESULTS_CSV = EXPLORATORY_DIR / "run00097_seed_study_backfill" / "results.csv"
RUN00097_SL_VERB_RESULTS_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_run00097_seed_study_mmlu_sl_verb_rerun/collect_results-34269c/results.csv"
)

UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
MMLU_SL_VERB_BPB_METRIC = "lm_eval/mmlu_sl_verb_5shot/bpb"


@dataclass(frozen=True)
class SignalNoiseMetricSpec:
    """One metric to include in the ranked SNR table."""

    eval_name: str
    metric: str
    source: str
    primary_metric_kind: str


OVERLAP_PRIMARY_METRICS = (
    SignalNoiseMetricSpec("olmo_base_easy_overlap_macro", "lm_eval/olmo_base_easy_overlap/macro_bpb", "overlap", "bpb"),
    SignalNoiseMetricSpec("mmlu_5shot", "lm_eval/mmlu_5shot/bpb", "overlap", "bpb"),
    SignalNoiseMetricSpec("mmlu_stem_5shot", "lm_eval/mmlu_stem_5shot/bpb", "overlap", "bpb"),
    SignalNoiseMetricSpec("mmlu_humanities_5shot", "lm_eval/mmlu_humanities_5shot/bpb", "overlap", "bpb"),
    SignalNoiseMetricSpec(
        "mmlu_social_sciences_5shot",
        "lm_eval/mmlu_social_sciences_5shot/bpb",
        "overlap",
        "bpb",
    ),
    SignalNoiseMetricSpec("mmlu_other_5shot", "lm_eval/mmlu_other_5shot/bpb", "overlap", "bpb"),
    SignalNoiseMetricSpec("arc_easy_5shot", "lm_eval/arc_easy_5shot/bpb", "overlap", "bpb"),
    SignalNoiseMetricSpec("arc_challenge_5shot", "lm_eval/arc_challenge_5shot/bpb", "overlap", "bpb"),
    SignalNoiseMetricSpec("csqa_5shot", "lm_eval/csqa_5shot/bpb", "overlap", "bpb"),
    SignalNoiseMetricSpec("hellaswag_5shot", "lm_eval/hellaswag_5shot/bpb", "overlap", "bpb"),
    SignalNoiseMetricSpec("winogrande_5shot", "lm_eval/winogrande_5shot/bpb", "overlap", "bpb"),
    SignalNoiseMetricSpec("piqa_5shot", "lm_eval/piqa_5shot/bpb", "overlap", "bpb"),
    SignalNoiseMetricSpec("lambada_0shot", "lm_eval/lambada_0shot/perplexity", "overlap", "perplexity"),
    SignalNoiseMetricSpec("medmcqa_5shot", "lm_eval/medmcqa_5shot/acc_norm", "overlap", "acc_norm"),
    SignalNoiseMetricSpec("sciq_5shot", "lm_eval/sciq_5shot/acc_norm", "overlap", "acc_norm"),
    SignalNoiseMetricSpec("socialiqa_5shot", "lm_eval/socialiqa_5shot/acc", "overlap", "acc"),
    SignalNoiseMetricSpec(
        "socialiqa_5shot_choice_logprob",
        "lm_eval/socialiqa_5shot/choice_logprob",
        "overlap",
        "choice_logprob",
    ),
)

EXTRA_PRIMARY_METRICS = (SignalNoiseMetricSpec("mmlu_sl_verb_5shot", MMLU_SL_VERB_BPB_METRIC, "non_overlap", "bpb"),)


def _read_gcs_csv(uri: str) -> pd.DataFrame:
    data = subprocess.check_output(["gsutil", "cat", uri], text=True)
    return pd.read_csv(io.StringIO(data))


def _read_csv(path_or_uri: Path | str) -> pd.DataFrame:
    if isinstance(path_or_uri, Path):
        return pd.read_csv(path_or_uri)
    if path_or_uri.startswith("gs://"):
        return _read_gcs_csv(path_or_uri)
    return pd.read_csv(path_or_uri)


def _list_gcs_paths(pattern: str) -> list[str]:
    output = subprocess.check_output(["gsutil", "ls", pattern], text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _load_overlap_signal_frame() -> pd.DataFrame:
    shard_uris = sorted(_list_gcs_paths(QSPLIT240_OVERLAP_RESULTS_GLOB))
    if len(shard_uris) != 8:
        raise ValueError(f"Expected 8 overlap shard results, found {len(shard_uris)}: {shard_uris}")

    frame = pd.concat([_read_gcs_csv(uri) for uri in shard_uris], ignore_index=True)
    expected_names = {run.run_name for run in load_original_qsplit240_with_core_baselines()}
    actual_names = set(frame["run_name"])
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"Overlap signal run set mismatch. Missing={missing[:8]} extra={extra[:8]}")
    if len(frame) != len(expected_names):
        raise ValueError(f"Expected {len(expected_names)} overlap signal rows, found {len(frame)}")

    return frame.sort_values("run_id").reset_index(drop=True)


def _load_standard_signal_frame() -> pd.DataFrame:
    frame = pd.read_csv(COMMON_SWARM_PATH)
    expected_names = {run.run_name for run in load_original_qsplit240_with_core_baselines()}
    filtered = frame.loc[frame["run_name"].isin(expected_names)].copy()
    actual_names = set(filtered["run_name"])
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"Standard signal run set mismatch. Missing={missing[:8]} extra={extra[:8]}")
    if len(filtered) != len(expected_names):
        raise ValueError(f"Expected {len(expected_names)} standard signal rows, found {len(filtered)}")
    return filtered.sort_values("run_id").reset_index(drop=True)


def _snr_row(
    *,
    spec: SignalNoiseMetricSpec,
    signal_values: pd.Series,
    noise_values: pd.Series,
) -> dict[str, float | int | str]:
    signal = signal_values.dropna()
    noise = noise_values.dropna()
    if len(signal) < 2:
        raise ValueError(f"{spec.metric}: need at least 2 signal values, got {len(signal)}")
    if len(noise) < 2:
        raise ValueError(f"{spec.metric}: need at least 2 noise values, got {len(noise)}")

    signal_scale = float(signal.std(ddof=1))
    noise_scale = float(noise.std(ddof=1))
    return {
        "eval_name": spec.eval_name,
        "metric": spec.metric,
        "source": spec.source,
        "primary_metric_kind": spec.primary_metric_kind,
        "signal_n": len(signal),
        "noise_n": len(noise),
        "signal_scale": signal_scale,
        "noise_scale": noise_scale,
        "signal_range": float(signal.max() - signal.min()),
        "signal_to_noise": signal_scale / noise_scale,
    }


def _eval_bpb_metric_specs(*, signal_frame: pd.DataFrame, noise_frame: pd.DataFrame) -> list[SignalNoiseMetricSpec]:
    metrics = [
        column
        for column in signal_frame.columns
        if column.startswith("eval/")
        and (column.endswith("/bpb") or column.endswith("_bpb") or column in {"eval/bpb", "eval/macro_bpb"})
        and column in noise_frame.columns
    ]
    metrics = sorted(set(metrics))

    specs: list[SignalNoiseMetricSpec] = []
    for metric in metrics:
        if metric == "eval/bpb":
            eval_name = "eval_total"
        elif metric == "eval/macro_bpb":
            eval_name = "eval_macro"
        else:
            eval_name = metric.removeprefix("eval/").removesuffix("/bpb").removesuffix("_bpb").replace("/", "::")
        specs.append(
            SignalNoiseMetricSpec(
                eval_name=eval_name,
                metric=metric,
                source="eval_bpb",
                primary_metric_kind="bpb",
            )
        )
    return specs


def build_signal_to_noise_table() -> pd.DataFrame:
    overlap_signal = _load_overlap_signal_frame()
    overlap_noise = _read_gcs_csv(RUN00097_OVERLAP_RESULTS_URI)
    sl_verb_signal = _read_gcs_csv(QSPLIT240_SL_VERB_RESULTS_URI)
    sl_verb_noise = _read_gcs_csv(RUN00097_SL_VERB_RESULTS_URI)
    standard_noise_path: Path | str = (
        RUN00097_STANDARD_BACKFILL_RESULTS_CSV
        if RUN00097_STANDARD_BACKFILL_RESULTS_CSV.exists()
        else RUN00097_STANDARD_RESULTS_URI
    )
    standard_noise = _read_csv(standard_noise_path)
    standard_seed_noise = standard_noise.loc[standard_noise["cohort"] == "seed_sweep"].reset_index(drop=True)
    standard_signal = _load_standard_signal_frame()
    eval_bpb_specs = _eval_bpb_metric_specs(signal_frame=standard_signal, noise_frame=standard_seed_noise)

    rows = [
        _snr_row(
            spec=spec,
            signal_values=overlap_signal[spec.metric],
            noise_values=overlap_noise[spec.metric],
        )
        for spec in OVERLAP_PRIMARY_METRICS
    ]
    rows.extend(
        [
            _snr_row(
                spec=spec,
                signal_values=standard_signal[spec.metric],
                noise_values=standard_seed_noise[spec.metric],
            )
            for spec in eval_bpb_specs
        ]
    )
    rows.extend(
        [
            _snr_row(
                spec=EXTRA_PRIMARY_METRICS[0],
                signal_values=sl_verb_signal[MMLU_SL_VERB_BPB_METRIC],
                noise_values=sl_verb_noise[MMLU_SL_VERB_BPB_METRIC],
            ),
        ]
    )

    return pd.DataFrame(rows).sort_values("signal_to_noise", ascending=False).reset_index(drop=True)


def main() -> None:
    frame = build_signal_to_noise_table()
    frame.to_csv(OUTPUT_CSV, index=False)
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

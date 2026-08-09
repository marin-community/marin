# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Freeze the WSD80 batch-size and simulated-repetition intervention panel."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

DESIGN_VERSION = "2026-08-04-v1"
SEQ_LEN = 2048
PHASE_0_FRACTION = Fraction(4, 5)
MIXTURE_BLOCK_SIZE = 2048
MATERIALIZED_TOKENS = 7_408_189_440
BASE_BATCH_SIZE = 128
BASE_TOTAL_STEPS = 28_260
BASE_TARGET_BUDGET = 5_729_908_864_777
BASE_MUON_LR = 0.02
BASE_ADAM_LR = 0.008
WARMUP_FRACTION = Fraction(1, 100)
MODEL_HIDDEN_SIZE = 640
MODEL_TOTAL_PARAMETERS = 210_052_480
MODEL_NON_EMBEDDING_PARAMETERS = 45_884_800
MODEL_NUM_LAYERS = 7
MODEL_NUM_HEADS = 5
MODEL_FLOPS_PER_TOKEN = 292_833_280.0
STARCODER_SOURCE_TOKENS = 216_567_300_822
NEMOTRON_SOURCE_TOKENS = BASE_TARGET_BUDGET
PAIR_SEEDS = tuple(range(20_260_811, 20_260_817))
EXPECTED_CONDITIONS = 8
EXPECTED_POLICIES = 3
EXPECTED_RUN_COUNT = EXPECTED_CONDITIONS * EXPECTED_POLICIES * len(PAIR_SEEDS)

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "starcoder_wsd80_batch_repetition_design_20260804.json"
ARTIFACT_DIR = Path(__file__).parent / "reference_outputs" / "starcoder_wsd80_batch_repetition_design_20260804"
CSV_PATH = ARTIFACT_DIR / "run_manifest.csv"
REPORT_PATH = ARTIFACT_DIR / "report.md"


@dataclass(frozen=True)
class ConditionSpec:
    """One optimizer or simulated-repetition condition."""

    condition_id: str
    family: str
    batch_size: int
    learning_rate_scale: Fraction
    target_budget_multiplier: Fraction
    interpretation: str


@dataclass(frozen=True)
class PolicySpec:
    """One fixed policy used in every condition."""

    policy_id: str
    role: str
    phase_0_starcoder: Fraction
    phase_1_starcoder: Fraction


CONDITIONS = (
    ConditionSpec("base", "baseline", 128, Fraction(1), Fraction(1), "historical optimizer and pool"),
    ConditionSpec("b064_fixed", "batch_fixed_peak_lr", 64, Fraction(1), Fraction(1), "smaller batch at fixed peak LR"),
    ConditionSpec("b256_fixed", "batch_fixed_peak_lr", 256, Fraction(1), Fraction(1), "larger batch at fixed peak LR"),
    ConditionSpec(
        "b064_intlr",
        "batch_integrated_lr_control",
        64,
        Fraction(1, 2),
        Fraction(1),
        "smaller batch with approximately preserved integrated LR and eta over batch",
    ),
    ConditionSpec(
        "b256_intlr",
        "batch_integrated_lr_control",
        256,
        Fraction(2),
        Fraction(1),
        "larger batch with approximately preserved integrated LR and eta over batch",
    ),
    ConditionSpec(
        "target025",
        "simulated_repetition",
        128,
        Fraction(1),
        Fraction(1, 4),
        "four-times larger unique pool and one-quarter simulated epochs",
    ),
    ConditionSpec(
        "target050",
        "simulated_repetition",
        128,
        Fraction(1),
        Fraction(1, 2),
        "two-times larger unique pool and one-half simulated epochs",
    ),
    ConditionSpec(
        "target200",
        "simulated_repetition",
        128,
        Fraction(1),
        Fraction(2),
        "one-half unique pool and twice simulated epochs",
    ),
)

POLICIES = (
    PolicySpec("A_phase", "confirmed_two_phase_candidate", Fraction(1, 50), Fraction(41, 50)),
    PolicySpec("B_agg018", "aggregate_matched_tied", Fraction(9, 50), Fraction(9, 50)),
    PolicySpec("C_tied070", "best_observed_tied", Fraction(7, 10), Fraction(7, 10)),
)


def canonical_sha256(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _rounded_fraction(value: int, multiplier: Fraction) -> int:
    numerator = value * multiplier.numerator
    return (numerator + multiplier.denominator // 2) // multiplier.denominator


def _schedule(condition: ConditionSpec) -> dict[str, int | float]:
    tokens_per_step = condition.batch_size * SEQ_LEN
    if MATERIALIZED_TOKENS % tokens_per_step != 0:
        raise ValueError(f"{condition.condition_id}: materialized tokens are not divisible by tokens per step")
    total_steps = MATERIALIZED_TOKENS // tokens_per_step
    boundary_numerator = total_steps * PHASE_0_FRACTION.numerator
    if boundary_numerator % PHASE_0_FRACTION.denominator != 0:
        raise ValueError(f"{condition.condition_id}: 80/20 boundary is not integral")
    boundary_step = boundary_numerator // PHASE_0_FRACTION.denominator
    if (boundary_step * condition.batch_size) % MIXTURE_BLOCK_SIZE != 0:
        raise ValueError(f"{condition.condition_id}: phase boundary is not mixture-block aligned")
    warmup_steps = total_steps * WARMUP_FRACTION.numerator // WARMUP_FRACTION.denominator
    eval_interval_steps = total_steps // 10
    if eval_interval_steps * 10 != total_steps:
        raise ValueError(f"{condition.condition_id}: endpoint is not divisible into ten eval intervals")
    lr_scale = float(condition.learning_rate_scale)
    batch_ratio = condition.batch_size / BASE_BATCH_SIZE
    return {
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "boundary_step": boundary_step,
        "warmup_steps": warmup_steps,
        "decay_steps": total_steps - boundary_step,
        "eval_interval_steps": eval_interval_steps,
        "materialized_tokens": total_steps * tokens_per_step,
        "muon_learning_rate": BASE_MUON_LR * lr_scale,
        "adam_learning_rate": BASE_ADAM_LR * lr_scale,
        "integrated_lr_proxy_relative": lr_scale * total_steps / BASE_TOTAL_STEPS,
        "eta_over_batch_proxy_relative": lr_scale / batch_ratio,
    }


def _policy_row(policy: PolicySpec) -> dict[str, float | str]:
    phase_0 = float(policy.phase_0_starcoder)
    phase_1 = float(policy.phase_1_starcoder)
    aggregate = float(PHASE_0_FRACTION) * phase_0 + (1.0 - float(PHASE_0_FRACTION)) * phase_1
    return {
        "policy_id": policy.policy_id,
        "role": policy.role,
        "phase_0_starcoder": phase_0,
        "phase_1_starcoder": phase_1,
        "aggregate_starcoder": aggregate,
    }


def _epoch_summary(condition: ConditionSpec, policy: PolicySpec) -> dict[str, float]:
    target_budget = _rounded_fraction(BASE_TARGET_BUDGET, condition.target_budget_multiplier)
    phase_0 = float(policy.phase_0_starcoder)
    phase_1 = float(policy.phase_1_starcoder)
    beta_0 = float(PHASE_0_FRACTION)
    beta_1 = 1.0 - beta_0
    return {
        "starcoder_phase_0_epochs": beta_0 * phase_0 * target_budget / STARCODER_SOURCE_TOKENS,
        "starcoder_phase_1_epochs": beta_1 * phase_1 * target_budget / STARCODER_SOURCE_TOKENS,
        "nemotron_phase_0_epochs": beta_0 * (1.0 - phase_0) * target_budget / NEMOTRON_SOURCE_TOKENS,
        "nemotron_phase_1_epochs": beta_1 * (1.0 - phase_1) * target_budget / NEMOTRON_SOURCE_TOKENS,
    }


def build_payload() -> dict[str, Any]:
    """Build and validate the complete immutable design payload."""
    model = CompletedAdamHHeuristic()._build_model_config(MODEL_HIDDEN_SIZE, seq_len=SEQ_LEN)
    observed_model = {
        "hidden_size": MODEL_HIDDEN_SIZE,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "total_parameters": model.total_trainable_params(llama3_tokenizer_vocab_size),
        "non_embedding_parameters": model.total_trainable_params(0),
        "flops_per_token": float(model.flops_per_token(llama3_tokenizer_vocab_size, SEQ_LEN)),
    }
    expected_model = {
        "hidden_size": MODEL_HIDDEN_SIZE,
        "num_layers": MODEL_NUM_LAYERS,
        "num_heads": MODEL_NUM_HEADS,
        "total_parameters": MODEL_TOTAL_PARAMETERS,
        "non_embedding_parameters": MODEL_NON_EMBEDDING_PARAMETERS,
        "flops_per_token": MODEL_FLOPS_PER_TOKEN,
    }
    if observed_model != expected_model:
        raise ValueError(f"Model geometry drifted: {observed_model} != {expected_model}")

    condition_rows = []
    for condition in CONDITIONS:
        schedule = _schedule(condition)
        multiplier = float(condition.target_budget_multiplier)
        condition_rows.append(
            {
                "condition_id": condition.condition_id,
                "family": condition.family,
                "batch_size": condition.batch_size,
                "learning_rate_scale": float(condition.learning_rate_scale),
                "target_budget_multiplier": multiplier,
                "target_budget": _rounded_fraction(BASE_TARGET_BUDGET, condition.target_budget_multiplier),
                "unique_pool_scale_relative": 1.0 / multiplier,
                "simulated_epoch_scale_relative": multiplier,
                "interpretation": condition.interpretation,
                **schedule,
            }
        )

    policy_rows = [_policy_row(policy) for policy in POLICIES]
    if abs(float(policy_rows[0]["aggregate_starcoder"]) - float(policy_rows[1]["aggregate_starcoder"])) > 1e-12:
        raise ValueError("Policies A and B are not aggregate matched")

    rows = []
    for condition, condition_row in zip(CONDITIONS, condition_rows, strict=True):
        for policy in POLICIES:
            policy_row = _policy_row(policy)
            epochs = _epoch_summary(condition, policy)
            for seed in PAIR_SEEDS:
                run_name = f"brm_{condition.condition_id}_{policy.policy_id}_s{seed}"
                rows.append(
                    {
                        "run_name": run_name,
                        "condition_id": condition.condition_id,
                        "condition_family": condition.family,
                        "policy_id": policy.policy_id,
                        "policy_role": policy.role,
                        "pair_seed": seed,
                        "data_seed": seed,
                        "simulated_epoch_subset_seed": seed,
                        "phase_0_starcoder": policy_row["phase_0_starcoder"],
                        "phase_1_starcoder": policy_row["phase_1_starcoder"],
                        "aggregate_starcoder": policy_row["aggregate_starcoder"],
                        "batch_size": condition_row["batch_size"],
                        "total_steps": condition_row["total_steps"],
                        "boundary_step": condition_row["boundary_step"],
                        "warmup_steps": condition_row["warmup_steps"],
                        "decay_steps": condition_row["decay_steps"],
                        "eval_interval_steps": condition_row["eval_interval_steps"],
                        "materialized_tokens": condition_row["materialized_tokens"],
                        "muon_learning_rate": condition_row["muon_learning_rate"],
                        "adam_learning_rate": condition_row["adam_learning_rate"],
                        "target_budget": condition_row["target_budget"],
                        "target_budget_multiplier": condition_row["target_budget_multiplier"],
                        "unique_pool_scale_relative": condition_row["unique_pool_scale_relative"],
                        "simulated_epoch_scale_relative": condition_row["simulated_epoch_scale_relative"],
                        **epochs,
                    }
                )

    if len(rows) != EXPECTED_RUN_COUNT or len({row["run_name"] for row in rows}) != EXPECTED_RUN_COUNT:
        raise ValueError("Intervention run manifest has the wrong cardinality")
    for condition in CONDITIONS:
        condition_runs = [row for row in rows if row["condition_id"] == condition.condition_id]
        if len(condition_runs) != EXPECTED_POLICIES * len(PAIR_SEEDS):
            raise ValueError(f"{condition.condition_id}: incomplete policy-by-seed block")

    payload: dict[str, Any] = {
        "design_version": DESIGN_VERSION,
        "description": "Fixed-arm WSD80 batch-size and simulated-repetition mechanism interventions.",
        "expected_condition_count": EXPECTED_CONDITIONS,
        "expected_policy_count": EXPECTED_POLICIES,
        "expected_run_count": EXPECTED_RUN_COUNT,
        "pair_seeds": list(PAIR_SEEDS),
        "training_environment": {
            "tpu_type": "v5p-8",
            "tpu_region": "us-central1",
            "tpu_zone": "us-central1-a",
            "marin_prefix": "gs://marin-us-central1",
        },
        "fixed_training": {
            "sequence_length": SEQ_LEN,
            "phase_0_fraction": float(PHASE_0_FRACTION),
            "mixture_block_size": MIXTURE_BLOCK_SIZE,
            "materialized_tokens": MATERIALIZED_TOKENS,
            "base_target_budget": BASE_TARGET_BUDGET,
            "model": observed_model,
        },
        "policies": policy_rows,
        "conditions": condition_rows,
        "analysis": {
            "primary_metric": "eval/paloma/dolma_100_programing_languages/bpb",
            "primary_estimand": "delta_order = loss(B_agg018) - loss(A_phase)",
            "treatment_estimand": "gamma_j = delta_order(j) - delta_order(base)",
            "batch_multiplicity_family": [
                "b064_fixed",
                "b256_fixed",
                "b064_intlr",
                "b256_intlr",
            ],
            "repetition_multiplicity_family": ["target025", "target050", "target200"],
            "multiplicity_control": "two-sided paired t-tests with Holm correction separately within each family",
            "secondary_estimands": [
                "delta_aggregate = loss(B_agg018) - loss(C_tied070)",
                "delta_global = loss(C_tied070) - loss(A_phase)",
            ],
            "selection_policy": (
                "No Bayesian optimization or adaptive acquisition; all coordinates and contrasts are frozen."
            ),
        },
        "caveats": [
            (
                "Linear LR scaling approximately preserves integrated LR and eta-over-batch; it is not a pure "
                "gradient-noise intervention."
            ),
            (
                "At target-budget multiplier 2, C_tied070 reaches about 37 StarCoder epochs and may show aggregate "
                "repetition damage."
            ),
            "The panel measures gain magnitude at one aggregate and contrast; it does not locate a new optimum.",
        ],
        "runs": rows,
    }
    payload["design_sha256"] = canonical_sha256(payload)
    return payload


def _write_report(payload: dict[str, Any]) -> None:
    condition_lines = []
    for row in payload["conditions"]:
        condition_lines.append(
            "| {condition_id} | {family} | {batch_size} | {total_steps} | {muon_learning_rate:.4f} | "
            "{target_budget_multiplier:.2f} | {unique_pool_scale_relative:.2f} | "
            "{simulated_epoch_scale_relative:.2f} |".format(**row)
        )
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# StarCoder WSD80 batch and repetition intervention design",
                "",
                "This is a frozen, nonadaptive 144-run mechanism screen at the confirmed high-token cell: "
                "N=210,052,480 parameters and D=7,408,189,440 materialized tokens.",
                "",
                "## Fixed policies",
                "",
                "- A: `(phase0 code, phase1 code)=(0.02,0.82)`, the fresh-seed-confirmed two-phase candidate.",
                "- B: `(0.18,0.18)`, the exact aggregate-matched tied control for A.",
                "- C: `(0.70,0.70)`, the best observed tied comparator in the selected cell.",
                "",
                (
                    "Every condition runs all three policies on seeds 20260811 through 20260816. "
                    "No historical endpoint is reused."
                ),
                "",
                "## Conditions",
                "",
                (
                    "| condition | family | batch | steps | Muon LR | target-budget multiplier | unique-pool scale | "
                    "epoch scale |"
                ),
                "|---|---|---:|---:|---:|---:|---:|---:|",
                *condition_lines,
                "",
                "The target-budget multiplier and unique-pool scale are reciprocals. Increasing the target-budget "
                "multiplier "
                "shrinks the simulated unique pool and increases materialized epochs.",
                "",
                "## Frozen analysis",
                "",
                "The primary metric is Paloma Dolma 100 Programming Languages BPB; lower is better. Define "
                "`delta_order = L(B)-L(A)` and `gamma_j = delta_order(j)-delta_order(base)`. Test four batch conditions "
                "and three repetition conditions as separate Holm-corrected, two-sided paired families. "
                "`L(B)-L(C)` and `L(C)-L(A)` are secondary decomposition diagnostics.",
                "",
                "The fixed-peak-LR branch changes batch size, optimizer steps, integrated LR, and eta-over-batch. "
                "The integrated-LR control scales both Muon and Adam peak learning rates linearly with batch size, "
                "approximately preserving integrated LR and eta-over-batch. It is a schedule-normalized control, "
                "not a pure noise arm.",
                "",
                "At twice the target budget, policy C reaches roughly 37 StarCoder epochs; aggregate repetition damage "
                "there is preregistered as a secondary interpretation rather than phase-order evidence.",
                "",
                f"Design SHA-256: `{payload['design_sha256']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    payload = build_payload()
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["runs"][0]))
        writer.writeheader()
        writer.writerows(payload["runs"])
    _write_report(payload)
    print(
        json.dumps(
            {
                "design_path": str(OUTPUT_PATH),
                "manifest_path": str(CSV_PATH),
                "report_path": str(REPORT_PATH),
                "run_count": len(payload["runs"]),
                "design_sha256": payload["design_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

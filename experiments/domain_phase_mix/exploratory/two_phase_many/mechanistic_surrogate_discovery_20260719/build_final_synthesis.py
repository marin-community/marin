# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "tabulate>=0.9",
# ]
# ///

"""Build the final synthesis for the second mechanistic-surrogate drive."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import plotly.express as px

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[4]
OUTPUT_ROOT = TWO_PHASE_ROOT / "reference_outputs" / "mechanistic_surrogate_discovery_20260719"
FINAL_DIR = OUTPUT_ROOT / "final_synthesis"
FROZEN_DIR = OUTPUT_ROOT / "frozen_gate"
PRIOR_FINAL = (
    TWO_PHASE_ROOT
    / "reference_outputs"
    / "mechanistic_surrogate_discovery_20260717"
    / "final_synthesis"
    / "final_report.md"
)
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


STATUS_STAGE = {
    "rejected_prior_drive": "prior-drive terminal decision",
    "blocked_before_starcoder": "algebraic or materiality audit",
    "blocked_before_nested_starcoder": "StarCoder nested-ablation gate",
    "blocked_before_multi_swarm": "StarCoder shape or raw-optimum gate",
    "blocked_before_adversarial": "multi-swarm or historical pre-adversarial gate",
    "blocked_round1": "round-one historical or cross-scale gate",
    "blocked_boundary_round1": "round-one identifiability or boundary gate",
    "legacy_result_rejected": "StarCoder stability gate",
    "theoretical_equivalence_block": "algebraic observational-equivalence theorem",
    "theoretical_reachable_set_block": "algebraic reachable-set theorem",
    "descriptive_only_not_admissible": "mechanistic admissibility gate",
}


def markdown_table(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    selected = frame if columns is None else frame[list(columns)]
    return selected.to_markdown(index=False, floatfmt=".5f")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_inventory() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(SCRIPT_DIR.glob("*.py")):
        source = path.read_text()
        module = ast.parse(source)
        docstring = ast.get_docstring(module) or ""
        rows.append(
            {
                "path": path.relative_to(TWO_PHASE_ROOT).as_posix(),
                "sha256": file_sha256(path),
                "module_purpose": docstring.splitlines()[0] if docstring else "",
                "line_count": len(source.splitlines()),
                "has_main_entry_point": 'if __name__ == "__main__"' in source,
                "has_pep723_metadata": "# /// script" in source,
            }
        )
    return pd.DataFrame(rows)


def pareto_mask(frame: pd.DataFrame, metrics: list[str]) -> pd.Series:
    values = frame[metrics].astype(float).to_numpy()
    keep = []
    for index, row in enumerate(values):
        dominated = any(
            other_index != index and (other <= row).all() and (other < row).any()
            for other_index, other in enumerate(values)
        )
        keep.append(not dominated)
    return pd.Series(keep, index=frame.index)


def build_acceptance_table(registry: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    evidence_paths: dict[str, str] = {}
    for row in ledger.itertuples(index=False):
        candidate_ids = {part.strip() for part in re.split(r"[,:+]", str(row.candidate_id))}
        for candidate_id in candidate_ids:
            if candidate_id and str(row.evidence_path).strip():
                evidence_paths[candidate_id] = str(row.evidence_path)
    rows = []
    for route in registry.itertuples(index=False):
        rows.append(
            {
                "route_id": route.id,
                "family": route.family,
                "source_drive": "20260717" if str(route.id).startswith("prior_") else "20260719",
                "terminal_status": route.status,
                "furthest_gate": STATUS_STAGE[route.status],
                "full_adversarial_gate_reached": False,
                "raw_optimum_gate_pass": False,
                "all_required_gates_pass": False,
                "evidence_path": evidence_paths.get(str(route.id), ""),
                "blocking_evidence": route.status_evidence,
            }
        )
    return pd.DataFrame(rows)


def build_pareto_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    archive = metrics[
        metrics["swarm"].eq("delphi_3e18") & metrics["policy"].eq("two_phase") & metrics["split"].eq("heldout_all")
    ].copy()
    archive["calibration_slope_error"] = (archive["calibration_slope_observed_on_predicted"].astype(float) - 1.0).abs()
    metric_columns = [
        "rmse",
        "calibration_slope_error",
        "regret_at_1",
        "optimism_gt_0p05_count",
        "worst_optimism",
    ]
    archive["pareto"] = False
    for _target, group in archive.groupby("target"):
        archive.loc[group.index, "pareto"] = pareto_mask(group, metric_columns)

    adversarial = metrics[
        metrics["swarm"].eq("delphi_3e18")
        & metrics["policy"].eq("two_phase")
        & metrics["split"].isin(["adversarial_candidate_uncheatable", "adversarial_candidate_table9"])
    ].copy()
    candidate_target = adversarial["split"].str.removeprefix("adversarial_candidate_")
    target_matched = adversarial[adversarial["target"].eq(candidate_target)].copy()
    cross_target = adversarial[~adversarial["target"].eq(candidate_target)].copy()
    for frame in (target_matched, cross_target):
        frame["candidate_target"] = frame["split"].str.removeprefix("adversarial_candidate_")
        frame["calibration_slope_error"] = (frame["calibration_slope_observed_on_predicted"].astype(float) - 1.0).abs()
    return archive, target_matched, cross_target


def write_tradeoff_plots(
    archive: pd.DataFrame,
    target_matched: pd.DataFrame,
    restriction_metrics: pd.DataFrame,
) -> None:
    archive_plot = archive.copy()
    archive_plot["marker_size"] = archive_plot["optimism_gt_0p05_count"].astype(float) + 5
    figure = px.scatter(
        archive_plot,
        x="rmse",
        y="regret_at_1",
        color="calibration_slope_error",
        size="marker_size",
        symbol="pareto",
        hover_name="model",
        facet_col="target",
        color_continuous_scale="RdYlGn_r",
        title="710-run / 690-coordinate Delphi archive: no model dominates calibration, error, and selection regret",
        labels={
            "rmse": "Heldout RMSE",
            "regret_at_1": "Regret@1",
            "calibration_slope_error": "|calibration slope - 1|",
        },
    )
    figure.update_layout(template="plotly_white", height=570, width=1120)
    figure.write_html(
        FINAL_DIR / "heldout_pareto_tradeoffs.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )

    adversarial_plot = target_matched.copy()
    adversarial_plot["marker_size"] = adversarial_plot["regret_at_1"].astype(float) + 0.003
    figure = px.scatter(
        adversarial_plot,
        x="rmse",
        y="calibration_slope_observed_on_predicted",
        color="regret_at_1",
        size="marker_size",
        hover_name="model",
        facet_col="target",
        color_continuous_scale="RdYlGn_r",
        title="Exposed adversarial panel: low RMSE can coexist with response compression",
        labels={
            "rmse": "Target-matched RMSE",
            "calibration_slope_observed_on_predicted": "Observed-on-predicted slope",
            "regret_at_1": "Regret@1",
        },
    )
    figure.add_hline(y=1.0, line_dash="dash", line_color="#4d5963")
    figure.update_layout(template="plotly_white", height=570, width=1120)
    figure.write_html(
        FINAL_DIR / "adversarial_compression_tradeoffs.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )

    historical = restriction_metrics[restriction_metrics["segment"].eq("historical_one_phase_heldout")].copy()
    pivot = historical.pivot_table(
        index=["target", "model"],
        columns="fit_mode",
        values="rmse",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    tied = "algebraic_restriction_of_two_phase_fit"
    direct = "independent_one_phase_refit"
    figure = px.scatter(
        pivot,
        x=direct,
        y=tied,
        hover_name="model",
        facet_col="target",
        color="target",
        title="One-phase restriction is an estimand choice, not a harmless implementation detail",
        labels={direct: "Independent one-phase refit RMSE", tied: "Tied two-phase-fit RMSE"},
    )
    bounds = [0, 1.05 * max(pivot[direct].max(), pivot[tied].max())]
    figure.add_shape(type="line", x0=bounds[0], y0=bounds[0], x1=bounds[1], y1=bounds[1], line_dash="dash")
    figure.update_xaxes(range=bounds)
    figure.update_yaxes(range=bounds)
    figure.update_layout(template="plotly_white", height=570, width=1120, showlegend=False)
    figure.write_html(
        FINAL_DIR / "one_phase_restriction_transfer.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def future_confirmation_payload() -> dict[str, object]:
    launcher = REPO_ROOT / "experiments/domain_phase_mix/launch_delphi_augmented_swarm_3e18.py"
    resolved_manifest = TWO_PHASE_ROOT / "reference_outputs/delphi_augmented_swarm_3e18_20260714/run_specs.json"
    domain_config = REPO_ROOT / "experiments/domain_phase_mix/two_phase_dolma3_dolmino_top_level.py"
    token_counts = REPO_ROOT / "experiments/domain_phase_mix/dolma3_dolmino_top_level_domains.py"
    tokenizer_config = REPO_ROOT / "experiments/llama.py"
    return {
        "status": "inactive_until_a_future_candidate_passes_every_development_gate",
        "purpose": (
            "Confirm a frozen surrogate's phase-transition law and selected optimum on new policies "
            "that cannot be used for fitting or model selection."
        ),
        "activation_requirements": [
            "One candidate form, fitting procedure, hyperparameters, and nested ablations are frozen.",
            "The candidate passes algebraic, both StarCoder, grouped-CV, one-phase, cross-scale, historical, and exposed-adversarial development gates.",
            "The candidate's raw and deployment-regularized optima are materialized and checksummed before any training outcome is read.",
        ],
        "phase_parameterization": {
            "aggregate": "bar_w = gamma0 * w0 + gamma1 * w1",
            "contrast": "d = w1 - w0",
            "reconstruction": ["w0 = bar_w - gamma1 * d", "w1 = bar_w + gamma0 * d"],
            "phase_fractions": [0.8, 0.2],
        },
        "training_configuration": {
            "contract": (
                "Use the exact Delphi 3e18 fit-swarm training and evaluation configuration below. "
                "No architecture, optimizer, schedule, tokenizer, data-pool, simulated-epoch, seed, "
                "phase-boundary, or evaluator field may change after activation."
            ),
            "target_flops": 3e18,
            "model": {
                "architecture": "Qwen3 decoder",
                "reference_checkpoint": "Qwen/Qwen3-0.6B",
                "total_trainable_parameters": 358_304_128,
                "non_embedding_parameters": 128_469_376,
                "hidden_dim": 896,
                "intermediate_dim": 3584,
                "num_layers": 10,
                "num_attention_heads": 7,
                "num_kv_heads": 7,
                "activation": "silu",
                "tie_word_embeddings": False,
                "rope": {
                    "theta": 500_000.0,
                    "factor": 8.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192,
                },
            },
            "tokens_and_batching": {
                "sequence_length": 4096,
                "global_batch_size": 128,
                "train_steps": 3007,
                "expected_final_checkpoint_step": 3006,
                "realized_train_tokens": 1_576_534_016,
                "tokens_per_parameter": 1_576_534_016 / 358_304_128,
            },
            "optimizer": {
                "name": "AdamH",
                "learning_rate": 0.01,
                "adam_learning_rate": 0.0011683387056186526,
                "weight_decay": 0.1,
                "beta1": 0.9,
                "beta2": 0.9998000100000001,
                "epsilon": 1.0387398741167107e-08,
                "max_grad_norm": 0.1,
                "lr_schedule": "linear",
                "warmup_fraction": 0.1,
                "decay_fraction": 0.2,
                "minimum_lr_ratio": 0.0,
                "nesterov": False,
            },
            "numerics": {
                "parameter_dtype": "float32",
                "compute_dtype": "bfloat16",
                "tensor_parallel_size": 1,
            },
            "schedule": {
                "type": "two-phase WSD-compatible varying mixture",
                "phase_boundary_fraction": 0.8,
                "phase_fractions": [0.8, 0.2],
                "mixture_block_size": 2048,
            },
            "data": {
                "tokenizer": "meta-llama/Meta-Llama-3.1-8B",
                "bucket_count": 39,
                "pool": "Dolma 3 and Dolmino top-level domains in us-east5",
                "available_top_level_tokens": 6_986_431_605_135,
                "simulated_epoch_target_budget": 6_325_183_647_689,
                "shuffle": True,
                "simulated_epoch_subset_seed": None,
            },
            "seed_rule": (
                "Keep trainer_seed=0. Assign a unique deterministic data_seed to every policy/replicate "
                "from the checksummed launch manifest; decisive repeats vary data_seed only."
            ),
            "evaluation": {
                "training_time_smooth_evals": ["Paloma", "Uncheatable eval"],
                "steps_per_eval": 1000,
                "native_table9_request_set": "raw/eval-datasets/olmo_base_eval_table9/v2",
                "required_final_metrics": ["eval/uncheatable_eval/bpb", "Table-9 macro BPB"],
            },
            "provenance": {
                "launcher_path": launcher.relative_to(REPO_ROOT).as_posix(),
                "launcher_sha256": file_sha256(launcher),
                "resolved_manifest_path": resolved_manifest.relative_to(REPO_ROOT).as_posix(),
                "resolved_manifest_sha256": file_sha256(resolved_manifest),
                "domain_config_path": domain_config.relative_to(REPO_ROOT).as_posix(),
                "domain_config_sha256": file_sha256(domain_config),
                "token_count_config_path": token_counts.relative_to(REPO_ROOT).as_posix(),
                "token_count_config_sha256": file_sha256(token_counts),
                "tokenizer_config_path": tokenizer_config.relative_to(REPO_ROOT).as_posix(),
                "tokenizer_config_sha256": file_sha256(tokenizer_config),
                "scaling_fit_path": (
                    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
                    "delphi_baseline_mixtures_issue6607_20260623/analysis-af9355/"
                    "isoflop_analysis_result.json"
                ),
                "scaling_fit_sha256": "097328aada40b0beb8b38c765ae0b30bf1767623a2b2eacd6c5c02a77af49f2b",
                "source_fit_panel_path": (
                    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
                    "delphi_augmented_swarm_3e18_20260714/source/"
                    "fit_panel_table9_macro-4f283bacb4ef269c.csv"
                ),
                "source_fit_panel_sha256": "4f283bacb4ef269c396277cbd518ef74212a51741c909a1e1e9ace040751d507",
            },
        },
        "anchors_per_target": [
            "proportional policy",
            "independently fitted one-phase optimum of the frozen candidate",
            "best observed development one-phase policy, fixed before confirmation",
        ],
        "contrasts_per_anchor": {
            "tied_control": 1,
            "candidate_direction_scales": [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0],
            "fixed_family_balanced_direction_pairs": 3,
            "direction_seed": 20260719,
            "simplex_safety_fraction": 0.90,
            "candidate_direction_rule": (
                "Compute d*=w1*-w0* from the frozen raw optimum; subtract sum(d*)/K for numerical centering; "
                "reject activation if its L1 norm is zero; otherwise set u=d*/||d*||_1."
            ),
            "family_direction_rule": (
                "Using NumPy PCG64 seed 20260719, draw three iid standard-normal bucket vectors. Within each "
                "predeclared family of size >=2 subtract that family's mean; set singleton-family entries to zero; "
                "normalize each nonzero result to L1 norm one. Reuse the same three directions for every target and anchor."
            ),
            "signed_ray_rule": (
                "For zero-sum L1-normalized u and sign q, let v=sign(q)u and "
                "a_max=min(min_{v_i>0} bar_w_i/(gamma1*v_i), min_{v_i<0} bar_w_i/(-gamma0*v_i)). "
                "Set d(q)=|q|*0.90*a_max*v, w0=bar_w-gamma1*d(q), w1=bar_w+gamma0*d(q)."
            ),
            "validation_rule": (
                "Do not clip or renormalize coordinates. Assert sum(d)=0, sum(w0)=sum(w1)=1, and minimum "
                "weight >= -1e-12; persist the family-map checksum and every pre-deduplication policy checksum."
            ),
        },
        "additional_policies": [
            "raw two-phase optimum for each target",
            "exact aggregate-matched tied counterfactual of the raw optimum for each target",
            "deployment-regularized two-phase optimum for each target",
            "exact aggregate-matched tied counterfactual of the deployment-regularized optimum for each target",
        ],
        "maximum_unique_policy_count_before_deduplication": 86,
        "decisive_repeats_per_arm": 15,
        "maximum_training_runs_before_deduplication": 170,
        "policy_count_arithmetic": {
            "anchor_policies": "2 targets * 3 anchors * (1 tied + 6 candidate-ray + 3*2 family-ray) = 78",
            "additional_optimum_policies": "2 targets * (raw + raw tied + regularized + regularized tied) = 8",
            "repeat_rows_beyond_single_seed": "2 targets * 3 decisive policies * (15-1 extra seeds) = 84",
        },
        "repeat_plan": (
            "Use 15 independent training seeds for each target's raw optimum, aggregate-matched tied control, "
            "and incumbent frontier; all other policies use one seed. The count powers the first Holm threshold "
            "at alpha=0.025 under the upper-95% Table-9 nuisance bound."
        ),
        "single_seed_policy_use": (
            "Signed contrast rays and regularized optima are descriptive surface checks only. They cannot replace "
            "a failed raw optimum, define a new winner, or alter the frozen acceptance rule after outcomes are unsealed."
        ),
        "evaluation": [
            "Evaluate both Uncheatable BPB and Table-9 macro BPB for every checkpoint.",
            "Report target-matched and cross-target results separately.",
            "Preserve policy, anchor, direction, scale, seed, proposer, and checksum provenance.",
        ],
        "primary_acceptance": [
            "No target-matched optimism error exceeds 0.05 BPB.",
            "Within each anchor, predicted versus observed contrast deltas have Spearman >= 0.5 and observed-on-predicted slope in [0.7, 1.3] on both targets.",
            "Target-matched Regret@1 across the untouched policies is <= 0.005 BPB.",
            "The frozen optimum matches the incumbent frontier within +0.002 BPB at the upper 95% confidence bound on both targets and improves at least one target by >= 0.005 BPB at the point estimate.",
            "Across the two frozen raw-optimum versus tied-control superiority tests, apply Holm family-wise control at alpha=0.05; the phase-varying optimum must pass on at least one target and must not regress the other target by more than 0.002 BPB.",
        ],
        "sealing": [
            "Commit the generator, candidate weights, manifest, and acceptance JSON before launch.",
            "Do not inspect partial outcomes; unseal only after all planned training and both evaluations complete.",
            "Any candidate change, fit change, hyperparameter change, or dropped policy invalidates confirmation and requires a new panel.",
        ],
    }


def build_future_confirmation_report(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Future untouched confirmation preregistration",
            "",
            "**Status:** inactive. No candidate from this drive is eligible for confirmation.",
            "",
            "This design becomes active only after a future candidate passes every development gate. "
            "It is deliberately a paired aggregate/contrast experiment: the aggregate mixture is held fixed while "
            "phase contrast varies, so evidence about phase order cannot be supplied by aggregate reweighting.",
            "",
            "For phase fractions \\(\\gamma_0=0.8\\) and \\(\\gamma_1=0.2\\), define",
            "",
            "$$\\bar w=\\gamma_0w^{(0)}+\\gamma_1w^{(1)},\\qquad d=w^{(1)}-w^{(0)},$$",
            "",
            "so \\(w^{(0)}=\\bar w-\\gamma_1d\\) and \\(w^{(1)}=\\bar w+\\gamma_0d\\).",
            "",
            "## Frozen panel",
            "",
            "For each target, use three aggregate anchors: proportional, the independently fitted one-phase optimum, "
            "and the best observed development one-phase policy. At each anchor, include the tied control, six signed "
            "scales of the candidate's predicted phase direction, and three fixed family-balanced random direction pairs. "
            "Add each target's raw and regularized optima with exact aggregate-matched tied controls. The resulting panel "
            "has at most 86 unique policies and 170 training runs before deduplication.",
            "",
            "Every contrast direction is centered to sum zero and normalized to unit \\(L_1\\) norm. For a signed "
            "direction \\(v\\), compute the exact positive simplex ray length",
            "",
            "$$a_{max}=\\min\\left\\{\\min_{v_i>0}\\frac{\\bar w_i}{\\gamma_1v_i}, "
            "\\min_{v_i<0}\\frac{\\bar w_i}{-\\gamma_0v_i}\\right\\},$$",
            "",
            "then use \\(d(q)=|q|\\,0.90\\,a_{max}v\\). Candidate-ray scales are "
            "\\(q\\in\\{-1,-0.5,-0.25,0.25,0.5,1\\}\\). The three fixed family-balanced directions are "
            "drawn once with NumPy PCG64 seed 20260719, centered separately within each non-singleton family, "
            "normalized to unit \\(L_1\\), and reused across targets and anchors. No coordinate is clipped or renormalized.",
            "",
            "Use 15 independent training seeds for each target's raw optimum, tied counterfactual, and incumbent frontier. "
            "Evaluate both targets for every checkpoint and preserve all direction, anchor, seed, proposer, and checksum metadata.",
            "Signed contrast rays and regularized optima are descriptive surface checks only. They cannot replace a failed "
            "raw optimum or be searched post hoc for a new winner.",
            "",
            "## Frozen training configuration",
            "",
            "Every policy uses the exact Delphi \\(3\\times10^{18}\\) fit-swarm contract: a 358,304,128-parameter "
            "Qwen3 decoder trained for 3,007 steps at global batch 128 and sequence length 4,096 "
            "(1,576,534,016 realized tokens), with 80/20 phase fractions, the Llama-3.1 tokenizer, and the "
            "completed-AdamH configuration recorded in the JSON preregistration. The launcher, resolved fit-swarm "
            "manifest, domain definition, token-count definition, scaling fit, tokenizer definition, and source panel "
            "are pinned by path and SHA-256. Repeats vary only the deterministic data seed; trainer seed remains zero. "
            "Every checkpoint receives both Uncheatable and native Table-9 evaluation.",
            "",
            "## Acceptance",
            "",
            *[f"- {item}" for item in payload["primary_acceptance"]],
            "",
            "## Seal",
            "",
            *[f"- {item}" for item in payload["sealing"]],
            "",
            "Any candidate or procedure change invalidates the panel as confirmation evidence.",
        ]
    )


def decisive_table(archive: pd.DataFrame, target_matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in archive.groupby("target"):
        rmse = group.loc[group["rmse"].astype(float).idxmin()]
        regret = group.loc[group["regret_at_1"].astype(float).idxmin()]
        calibration = group.loc[group["calibration_slope_error"].astype(float).idxmin()]
        rows.append(
            {
                "panel": "710-run / 690-coordinate archive",
                "target": target,
                "best_rmse_model": rmse["model"],
                "best_rmse": float(rmse["rmse"]),
                "best_regret_model": regret["model"],
                "best_regret_at_1": float(regret["regret_at_1"]),
                "best_calibrated_model": calibration["model"],
                "best_slope": float(calibration["calibration_slope_observed_on_predicted"]),
            }
        )
    for target, group in target_matched.groupby("target"):
        rmse = group.loc[group["rmse"].astype(float).idxmin()]
        regret = group.loc[group["regret_at_1"].astype(float).idxmin()]
        calibration = group.loc[group["calibration_slope_error"].astype(float).idxmin()]
        rows.append(
            {
                "panel": "exposed target-matched adversarial",
                "target": target,
                "best_rmse_model": rmse["model"],
                "best_rmse": float(rmse["rmse"]),
                "best_regret_model": regret["model"],
                "best_regret_at_1": float(regret["regret_at_1"]),
                "best_calibrated_model": calibration["model"],
                "best_slope": float(calibration["calibration_slope_observed_on_predicted"]),
            }
        )
    return pd.DataFrame(rows)


def adversarial_stratum_winner_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, stratum_type, stratum_value), group in metrics.groupby(
        ["target", "stratum_type", "stratum_value"], sort=True
    ):
        rmse = group.loc[group["rmse"].astype(float).idxmin()]
        regret = group.loc[group["regret_at_1"].astype(float).idxmin()]
        calibration_error = (group["calibration_slope_observed_on_predicted"].astype(float) - 1.0).abs()
        calibration = group.loc[calibration_error.idxmin()]
        rows.append(
            {
                "candidate_target": target,
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "n": int(group["n"].iloc[0]),
                "best_rmse_model": rmse["model"],
                "best_rmse": float(rmse["rmse"]),
                "best_regret_model": regret["model"],
                "best_regret_at_1": float(regret["regret_at_1"]),
                "best_calibrated_model": calibration["model"],
                "best_calibration_slope": float(calibration["calibration_slope_observed_on_predicted"]),
            }
        )
    return pd.DataFrame(rows)


def support_degradation_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare nearest- and farthest-support heldout quartiles."""
    rows = []
    for (target, model), group in metrics.groupby(["target", "model"], sort=True):
        nearest = group.loc[group["support_quartile"].eq("Q1 nearest")]
        farthest = group.loc[group["support_quartile"].eq("Q4 farthest")]
        if len(nearest) != 1 or len(farthest) != 1:
            raise ValueError(f"Expected one nearest and farthest support row for {target}/{model}")
        nearest_row = nearest.iloc[0]
        farthest_row = farthest.iloc[0]
        rows.append(
            {
                "target": target,
                "model": model,
                "nearest_rmse": float(nearest_row["rmse"]),
                "farthest_rmse": float(farthest_row["rmse"]),
                "rmse_ratio_farthest_over_nearest": float(farthest_row["rmse"] / nearest_row["rmse"]),
                "farthest_calibration_slope": float(farthest_row["calibration_slope_observed_on_predicted"]),
                "farthest_regret_at_1": float(farthest_row["regret_at_1"]),
                "farthest_optimism_gt_0p05_count": int(farthest_row["optimism_gt_0p05_count"]),
                "farthest_worst_optimism": float(farthest_row["worst_optimism"]),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    registry: pd.DataFrame,
    ledger: pd.DataFrame,
    archive: pd.DataFrame,
    target_matched: pd.DataFrame,
    cross_target: pd.DataFrame,
    adversarial_stratum_winners: pd.DataFrame,
    decisive: pd.DataFrame,
    restriction_metrics: pd.DataFrame,
    tail_metrics: pd.DataFrame,
    convergence: pd.DataFrame | None,
    design_metrics: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
    coordinate_metrics: pd.DataFrame,
    ledger_coverage: pd.DataFrame,
    variance_decomposition: pd.DataFrame,
    component_transfer: pd.DataFrame,
    phase_transfer_deattenuation: pd.DataFrame,
    cross_target_phase_state: pd.DataFrame,
    phase_attenuation_bootstrap: pd.DataFrame,
    stratum_robustness: pd.DataFrame,
    prior_identifiability: pd.DataFrame,
    prior_hyperparameter_stability: pd.DataFrame,
    prior_complexity: pd.DataFrame,
    prior_support: pd.DataFrame,
    prior_crossfit: pd.DataFrame,
    policy_class_winners: pd.DataFrame,
    policy_class_transfer: pd.DataFrame,
    support_degradation: pd.DataFrame,
    support_abstention_metrics: pd.DataFrame,
    maximum_safe_coverage: pd.DataFrame,
    exposure_pattern_summary: pd.DataFrame,
    frontier_phase_benefit: pd.DataFrame,
    confirmation_noise: pd.DataFrame,
    confirmation_power: pd.DataFrame,
    repeat_noise_influence: pd.DataFrame,
    phase_reversal_observability: pd.DataFrame,
    multiplicity_required_repeats: pd.DataFrame,
    metric_reproduction: pd.DataFrame,
    row_prediction_summary: pd.DataFrame,
) -> str:
    statuses = registry["status"].value_counts().rename_axis("status").reset_index(name="count")
    archive_pareto = archive[archive["pareto"]].copy()
    archive_pareto = archive_pareto.sort_values(["target", "rmse"])
    historical = restriction_metrics[restriction_metrics["segment"].eq("historical_one_phase_heldout")]
    restriction_winners = (
        historical.sort_values("rmse")
        .groupby(["target", "fit_mode"], as_index=False)
        .first()[["target", "fit_mode", "model", "n", "rmse", "regret_at_1", "worst_optimism"]]
    )
    hpr_tail = tail_metrics[tail_metrics["model"].eq("hierarchical_phase_bucket_replay")]
    hpr_tail = hpr_tail[
        [
            "target",
            "segment",
            "excluded_count",
            "rmse",
            "observed_on_predicted_slope",
            "regret_at_1",
            "optimism_gt_0p05_count",
            "worst_optimism",
        ]
    ]
    support_ratio_min = float(support_degradation["rmse_ratio_farthest_over_nearest"].min())
    support_ratio_max = float(support_degradation["rmse_ratio_farthest_over_nearest"].max())
    if convergence is None:
        numerical_section = [
            "The convergence-only Round 55b follow-up was still running when this report was generated; "
            "rerun this builder after it completes."
        ]
    else:
        numerical_section = [
            "### Raw-optimum numerical follow-up",
            "",
            markdown_table(
                convergence,
                [
                    "target",
                    "model",
                    "excluded_count",
                    "objective_delta",
                    "successful_starts",
                    "l1_from_original_optimum",
                    "phase_tv",
                    "max_simulated_epoch",
                ],
            ),
            "",
            "All 180 high-budget starts converged. Hierarchical phase replay's best policies moved by "
            "0.05--0.14 in policy \\(L_1\\) distance for only 0.00016--0.00142 BPB of surrogate "
            "improvement; separate-heads moved by at most 0.024. The original warnings were numerical, "
            "but resolving them exposes a flat or multimodal raw surface rather than a trustworthy optimum.",
        ]
    coordinate_winners = (
        coordinate_metrics.sort_values("rmse")
        .groupby(["target", "weighting"], as_index=False)
        .first()[
            [
                "target",
                "weighting",
                "model",
                "rmse",
                "calibration_slope_observed_on_predicted",
                "regret_at_1",
            ]
        ]
    )
    unstable_hyperparameters = int((~prior_hyperparameter_stability["cross_panel_stable"].astype(bool)).sum())
    boundary_hyperparameters = int(prior_hyperparameter_stability["boundary_selection_fraction"].ge(0.5).sum())
    optimum_stability = prior_crossfit.merge(
        prior_support[
            [
                "dataset",
                "policy",
                "distance_over_fit_p95",
                "convex_effective_support",
            ]
        ],
        on=["dataset", "policy"],
        validate="one_to_one",
    )
    rmse_policy_transfer = policy_class_transfer.loc[policy_class_transfer["metric"].eq("rmse")]
    cross_scale_frontier = frontier_phase_benefit.loc[
        ~frontier_phase_benefit["same_scale_selection"].astype(bool)
        & frontier_phase_benefit["slice"].isin(["top_10", "top_25", "top_50"])
    ]
    threshold_power = confirmation_power.loc[
        confirmation_power["effect_bpb"].eq(0.005) & confirmation_power["repeats_per_arm"].isin([3, 6])
    ]
    return "\n".join(
        [
            "# Mechanistic surrogate discovery II: final synthesis",
            "",
            "## Verdict",
            "",
            "**No investigated family passed the frozen acceptance gate. No headline surrogate is recommended.**",
            "",
            "The drive tested 58 materially distinct new routes on top of the 41 routes from the previous drive. "
            "All 99 registry entries now have explicit equations, latent states, transitions, limiting interpretations, "
            "terminal statuses, and blocking evidence. No candidate reached the full exposed-adversarial gate, so no "
            "candidate was promoted and independent Claude review was intentionally not invoked.",
            "",
            "This is not evidence that phase scheduling has no value. It is evidence that the available random-swarm "
            "designs do not identify a trustworthy raw two-phase optimum under any tested simple transition law.",
            "",
            "## Frozen boundary",
            "",
            "- Frozen gate manifest digest: `c4f711312423f038ef8610950d1ae6be30ffba588648177fbf5077e6931f93be`.",
            f"- Registry: {len(registry)} routes; prior-drive routes: {registry['id'].str.startswith('prior_').sum()}; new routes: {(~registry['id'].str.startswith('prior_')).sum()}.",
            f"- Append-only data-use ledger: {len(ledger)} rows.",
            "- Development archive: 710 fit-disjoint run rows representing 690 unique policy hashes; 12 exact fit-coordinate aliases are excluded.",
            "- The exposed 120-policy adversarial panel was used only in frozen batches or diagnostics recorded in the ledger.",
            "- The still-sealed confirmation evidence was not read, used for fitting, or used for any decision in this drive.",
            "- No new training job was submitted.",
            "",
            "## Search outcome",
            "",
            markdown_table(statuses),
            "",
            "The main new mechanisms included optimizer-specific matrix dynamics, stochastic batch-composition drift, "
            "deep factorization, metaplastic cascades, finite feature occupancy, gradient-Gram loss flow, and tied-counterfactual "
            "gradient-Gram transport. Most were rejected at the two-domain shape gate before expensive multi-swarm or adversarial evaluation.",
            "",
            "The closest new route was **orthogonal gradient-Gram transport**. It had an exact zero-phase null, its coupled "
            "transition beat the aggregate ablation globally and across folds, and its correlation parameter was interior and stable. "
            "It nevertheless missed the frozen StarCoder shape tolerance on both schedules, selected the wrong phase-order signature, "
            "and placed both raw optima far from the observed best policies. It was therefore blocked before multi-swarm evaluation.",
            "",
            "### Mechanism coverage",
            "",
            markdown_table(mechanism_summary),
            "",
            "All 58 new routes map exactly once to a primary mechanism class and a terminal gate. A lexical duplicate screen "
            "flagged only PWD/ESR; their state transitions are materially different and they have independent StarCoder failures. "
            "The negative result therefore does not come from repeatedly renaming a single GRP/DSP exposure curve.",
            "",
            "### Data-use ledger integrity",
            "",
            f"All {len(ledger_coverage)} new routes have an explicit original or provenance-reconciled ledger trail. "
            "Fifty-five empirical routes have both a freeze/preregistration edge and a terminal edge; the remaining "
            "three routes are complete one-edge theoretical or descriptive diagnostics. Seven incomplete historical "
            "edges were reconciled append-only without changing a model, hyperparameter, metric, or terminal status. "
            "Three malformed legacy batch markers remain preserved and are each superseded by a complete corrected "
            "preregistration. No new route was evaluated directly on exposed adversarial outcomes; the only direct "
            "adversarial reconstructions are the 11 pre-existing Pareto baselines.",
            "",
            "### Independent metric reproduction",
            "",
            markdown_table(metric_reproduction),
            "",
            "An independent implementation reproduced all 680 decision-critical archive and exposed target-matched "
            "adversarial scalar metrics from row-level predictions. The maximum absolute discrepancy was "
            "\\(6.7\\times10^{-16}\\), and integer counts matched exactly. The verdict is therefore not an artifact of "
            "a residual-sign, policy-mask, calibration, tail, or Regret@k implementation error.",
            "",
            "### Terminal row-level predictions",
            "",
            markdown_table(row_prediction_summary),
            "",
            "The terminal bundle directly exposes all 12,780 uniquely keyed archive predictions: 710 heldout runs, "
            "two targets, and nine Pareto-baseline models. Each row carries its immutable mixture hash, policy class, "
            "archive split, candidate-target relation, proposal stratum, residual, and optimism value. Aggregate "
            "claims can therefore be audited without reopening the Observatory application. The separate adversarial "
            "export contains 2,400 predictions: target-matched residuals for all 11 exposed-panel models and cross-target "
            "residuals for the nine models with archive-wide predictions.",
            "",
            "## Why no incumbent can be declared the answer",
            "",
            markdown_table(decisive),
            "",
            "Different models win different diagnostics. On the 710-run / 690-coordinate archive, compact retained state has the best "
            "Uncheatable RMSE while separate heads has the best Uncheatable Regret@1. Hierarchical phase replay has the best "
            "Table-9 RMSE while separate heads has the best Table-9 Regret@1. On the exposed target-matched adversarial panel, "
            "the RMSE and calibration winners change again. This is a genuine Pareto set, not a near-tie that can be resolved by ordinary OOF RMSE.",
            "",
            "### Archive Pareto baseline",
            "",
            markdown_table(
                archive_pareto,
                [
                    "target",
                    "model",
                    "parameter_count",
                    "n",
                    "rmse",
                    "calibration_slope_observed_on_predicted",
                    "regret_at_1",
                    "optimism_gt_0p05_count",
                    "worst_optimism",
                ],
            ),
            "",
            "### Exposed target-matched adversarial results",
            "",
            markdown_table(
                target_matched.sort_values(["target", "rmse"]),
                [
                    "target",
                    "model",
                    "parameter_count",
                    "n",
                    "rmse",
                    "spearman",
                    "calibration_slope_observed_on_predicted",
                    "regret_at_1",
                    "optimism_gt_0p05_count",
                    "worst_optimism",
                ],
            ),
            "",
            "Low RMSE is insufficient here: several target-matched fits have observed-on-predicted slopes far below one, "
            "which means the proposed frontier's response range is compressed even when catastrophic optimism is absent.",
            "",
            "### Exposed cross-target transfer",
            "",
            markdown_table(
                cross_target.sort_values(["target", "rmse"]),
                [
                    "candidate_target",
                    "target",
                    "model",
                    "n",
                    "rmse",
                    "spearman",
                    "calibration_slope_observed_on_predicted",
                    "regret_at_1",
                    "worst_optimism",
                ],
            ),
            "",
            "Cross-target rows are transfer evidence only: the policy was proposed for `candidate_target`, while `target` "
            "is the independently evaluated BPB. They are never pooled with target-matched rows for selection.",
            "",
            "### Proposal-stratum robustness",
            "",
            markdown_table(
                stratum_robustness.loc[stratum_robustness["stratum_type"].eq("selection_stratum")],
                [
                    "target",
                    "model",
                    "worst_rmse",
                    "minimum_calibration_slope",
                    "minimum_spearman",
                    "maximum_regret_at_1",
                ],
            ),
            "",
            "Every baseline has at least one selection stratum with calibration slope below 0.5 on both targets. "
            "Selection-stratum rank correlation becomes negative for 6/11 Uncheatable baselines and 8/11 Table-9 "
            "baselines. Proposal geometry therefore controls whether a model resolves the frontier response; pooled "
            "target-matched RMSE cannot establish a robust winner.",
            "",
            "### Adversarial policy/proposal strata",
            "",
            markdown_table(adversarial_stratum_winners),
            "",
            "This table reports winners separately for one-phase versus two-phase policies, candidate-generation origin, "
            "proposal-model series, and baseline/challenger/disagreement selection strata. Winner turnover across these "
            "predeclared slices is further evidence that no pooled adversarial metric identifies an incumbent.",
            "",
            "## Heldout provenance and weighting",
            "",
            markdown_table(coordinate_winners),
            "",
            "The append-only archive contains 690 unique policy hashes and 20 extra run rows across 11 repeated-coordinate groups. "
            "Ten groups are distinct-seed repeats; one five-run group has only two distinct data seeds and must not receive fivefold "
            "independent weight. Averaging observed and predicted BPB within each policy hash changes RMSE by at most 0.000632 BPB, "
            "changes observed-on-predicted slope by at most 0.011, changes no Regret@1 value, and preserves both RMSE winners. "
            "All 120 adversarial rows join one-to-one to their frozen target, policy class, selection stratum, and proposer metadata.",
            "",
            "## Identification results",
            "",
            "1. A compact-support bump construction proves that finite observations plus smoothness do not identify the raw optimum. "
            "Both dense StarCoder convex hulls contain large empty balls, so even interpolation inside the hull is insufficient without a falsified structural law.",
            "2. Models within 5%-15% of the best Delphi fit-panel OOF RMSE select different archive policies. On Table-9, "
            "fit-near-equivalent models incur selected regrets from 0.0022 to 0.1396 and can have zero top-5 overlap.",
            "3. Support distance predicts error more consistently than model disagreement, but neither is an admissible response term. "
            "They justify abstention and targeted interventions, not an ensemble or post-hoc calibration layer.",
            "4. The failure is radial extrapolation along estimable directions, not only null-space confounding, training randomness, "
            "evaluation randomness, or a few mislabeled tail points.",
            "",
            "### Support-stratified degradation",
            "",
            markdown_table(support_degradation),
            "",
            f"Across the frozen archive, the farthest-support quartile has {support_ratio_min:.2f}--{support_ratio_max:.2f} times "
            "the nearest-quartile RMSE for the audited "
            "models. Table-9 effective exposure also loses rank entirely in that quartile and incurs 36 severe optimism errors. "
            "This supports a deployment abstention rule, but support distance is not a permissible correction to the response law.",
            "",
            "### Support-abstention audit",
            "",
            markdown_table(maximum_safe_coverage),
            "",
            "A nearest-support abstention envelope removes all optimism errors above 0.05 BPB from most Uncheatable "
            "baselines while retaining 50%--75% of the archive. Table-9 is qualitatively different: every audited "
            "baseline must contract to the nearest 10% (71/710 rows) to satisfy the same rule. Even there, global-archive "
            "Regret@1 remains 0.039--0.072 BPB because the best observed policies can lie outside the retained region. "
            "Abstention is therefore a useful deployment warning but neither calibrates the response nor identifies the "
            "global optimum. Its thresholds were fixed diagnostically and were not fitted into any surrogate.",
            "",
            "### Worst-residual exposure patterns",
            "",
            markdown_table(exposure_pattern_summary),
            "",
            "Support distance and phase divergence have positive optimism correlations for every audited frozen baseline and target. "
            "Max epoch and aggregate tilt are usually negatively correlated with optimism, so the failure is not generic "
            "over-repetition. This is descriptive evidence for unsupported phase contrast, not permission to add residual-derived features.",
            "",
            "### Same-budget design audit",
            "",
            markdown_table(
                design_metrics.loc[design_metrics["block"].eq("joint")],
                [
                    "design",
                    "rows",
                    "columns",
                    "numerical_rank",
                    "stable_rank",
                    "condition_number",
                    "mean_canonical_correlation",
                    "max_canonical_correlation",
                    "contrast_energy_after_aggregate_residualization",
                ],
            ),
            "",
            "At the same 280-checkpoint budget, the observed random two-phase design has full joint rank but a maximum "
            "aggregate/contrast canonical correlation of 0.833 and preserves only 0.662 of standardized contrast energy "
            "after aggregate residualization. A diagnostic design with 140 tied policies and 70 signed phase-fiber pairs "
            "is exactly orthogonal to numerical precision and retains full 76-dimensional joint rank. This is a coordinate-only "
            "identification result: it supports a future intervention design but is not evidence for any current response model.",
            "",
            "### Phase-reversal observability",
            "",
            markdown_table(phase_reversal_observability),
            "",
            "An exact contrast reversal at fixed aggregate separates an odd order/recency effect from an even "
            "phase-variation effect without choosing a response law. The cosine StarCoder surface contains only one "
            "such triple and the WSD surface contains none. More strongly, none of the 238 non-tied Delphi policies "
            "has a feasible exact reversal under the actual 79.8/20.2 phase durations: reversing its large contrast "
            "would require a negative bucket weight. The random swarm therefore cannot identify this physically "
            "meaningful decomposition. Small signed simplex-fiber interventions are not just better conditioned; they "
            "supply a missing invariant that the present data do not contain.",
            "",
            "### Cross-scale variance decomposition",
            "",
            markdown_table(
                variance_decomposition,
                [
                    "target",
                    "scale",
                    "n",
                    "standard_deviation_aggregate",
                    "standard_deviation_phase_delta",
                    "phase_to_aggregate_sd_ratio",
                    "aggregate_phase_delta_correlation",
                ],
            ),
            "",
            markdown_table(
                component_transfer.loc[component_transfer["component"].eq("phase_delta")],
                [
                    "target",
                    "component",
                    "pearson",
                    "spearman",
                    "delphi_on_300m_slope",
                    "standard_deviation_ratio_delphi_over_300m",
                ],
            ),
            "",
            "On the 238 exactly matched policies, the phase correction has approximately the same standard deviation as "
            "the aggregate response at 300M, but only 0.48--0.54 times the aggregate standard deviation at Delphi. "
            "Aggregate response and phase correction are anticorrelated at both scales. The correction transfers moderately "
            "for Uncheatable and poorly for Table-9, with Delphi-on-300M slopes 0.417 and 0.286. Weaker two-phase transfer "
            "therefore contains a scale-dependent attenuation and cancellation problem, not merely larger stochastic variance.",
            "",
            "### Cross-scale measurement-error bound",
            "",
            markdown_table(phase_transfer_deattenuation),
            "",
            "Training and evaluation noise cannot explain the weak transfer. Under point nuisance estimates, deattenuation "
            "raises Table-9 phase-effect correlation only from 0.385 to 0.393; even upper-95% noise raises it only to "
            "0.410. The corresponding errors-in-variables slope is 0.305, still far from scale invariance. Uncheatable "
            "changes from 0.671 to at most 0.690. A valid cross-scale surrogate therefore needs a declared transition law "
            "for genuine phase-effect attenuation rather than more averaging or a scale-specific intercept.",
            "",
            "### Cross-target phase-state audit",
            "",
            markdown_table(cross_target_phase_state),
            "",
            markdown_table(phase_attenuation_bootstrap),
            "",
            "Phase effects share meaningful signal across targets, but a universal scalar phase state is too restrictive. "
            "About 21%--22% of policies reverse phase-benefit sign between objectives. Table-9's scale attenuation ratio "
            "also exceeds Uncheatable's by 0.121, with paired-bootstrap 95% interval [0.027, 0.221]. A future shared-state "
            "model therefore needs target-specific observation loadings and an independently identified transition; these "
            "results do not reopen the rejected joint-latent route, which selected full latent rank in most folds.",
            "",
            "### Frontier phase-benefit audit",
            "",
            markdown_table(
                cross_scale_frontier,
                [
                    "target",
                    "ranking_scale",
                    "evaluation_scale",
                    "slice",
                    "n",
                    "mean_phase_delta",
                    "bootstrap_mean_delta_low",
                    "bootstrap_mean_delta_high",
                    "fraction_two_phase_better",
                ],
            ),
            "",
            "Ranking one-phase policies at one scale and evaluating the paired phase effect at the other avoids the direct "
            "same-scale coupling induced by $\\Delta_{phase}=Y_{2p}-Y_{1p}$. Across the fixed top-10, top-25, and top-50 "
            "slices, the matched random two-phase realization is worse on average by 0.007--0.023 BPB; only 0.20--0.40 "
            "of those realizations improve. This is not a causal estimate of the best phase schedule because contrasts were "
            "not randomized conditionally around each aggregate. It is direct evidence that random two-phase sampling does "
            "not provide a symmetric local fiber around strong one-phase anchors, reinforcing the signed phase-fiber design.",
            "",
            "## Single-phase restriction audit",
            "",
            markdown_table(restriction_winners),
            "",
            "The algebraically tied restriction of a two-phase fit and an independently fitted one-phase model are different estimands. "
            "Direct one-phase refits often improve 238-row OOF metrics but can reverse badly on 93 historical one-phase policies. "
            "For example, the direct Table-9 canonical refit has OOF RMSE 0.0242 but historical RMSE 0.0705; the tied two-phase fit "
            "has historical RMSE 0.0216. A clean paper ablation must report both protocols rather than silently substituting one for the other.",
            "",
            "### Policy-class heldout robustness",
            "",
            markdown_table(policy_class_winners),
            "",
            markdown_table(
                policy_class_transfer,
                [
                    "source_panel",
                    "target",
                    "metric",
                    "common_models",
                    "one_vs_two_model_rank_spearman",
                    "one_phase_best_model",
                    "two_phase_best_model",
                    "same_best_model",
                ],
            ),
            "",
            f"One-phase versus two-phase RMSE model-rank correlation spans "
            f"{rmse_policy_transfer['one_vs_two_model_rank_spearman'].min():.3f}--"
            f"{rmse_policy_transfer['one_vs_two_model_rank_spearman'].max():.3f}. The best-RMSE model changes "
            "with policy class in three of four historical or target-matched target comparisons, and the best "
            "selection model changes even more often. Pooled archive metrics cannot establish a policy-class-robust incumbent.",
            "",
            "## Low-tail influence audit",
            "",
            markdown_table(hpr_tail),
            "",
            "Deleting the best 1, 3, 7, or 14 fit observations while freezing structural hyperparameters does not repair hierarchical "
            "phase replay: Regret@1 and the number of severe optimism errors remain unchanged on both targets. Separate heads is more "
            "influence-sensitive, but its catastrophic archive optimism also persists. The lowest-loss observations are not the root cause.",
            "",
            *numerical_section,
            "",
            "## Complexity, identifiability, and optimum stability",
            "",
            "These are provenance-checked carry-forward diagnostics from the prior drive, not a new model-selection round. "
            "Their source files and SHA-256 digests are recorded in `prior_stability_source_manifest.csv`.",
            "",
            "### Active-set effective complexity",
            "",
            markdown_table(
                prior_complexity,
                [
                    "dataset",
                    "model",
                    "nominal_parameter_count",
                    "active_parameter_count",
                    "effective_degrees_of_freedom",
                    "penalized_condition_number",
                    "phase_coefficient_active",
                    "phase_coefficient",
                ],
            ),
            "",
            "Effective degrees of freedom are active-set ridge hat traces in fitted-link space after nonlinear "
            "hyperparameters are frozen, so they are lower bounds. The values remain about 38--40 despite nominal "
            "counts of 57--58, while penalized condition numbers are \\(2.6\\times10^{12}\\)--\\(9.5\\times10^{12}\\). "
            "A small active-set count therefore does not imply a well-identified mechanistic transition.",
            "",
            "### Nonlinear and foldwise stability",
            "",
            markdown_table(
                prior_identifiability,
                [
                    "dataset",
                    "family",
                    "n_repeats",
                    "modal_config",
                    "modal_frequency",
                    "baseline_selection_frequency",
                    "unique_selected_configs",
                    "selected_extra_median",
                    "selected_extra_mad",
                ],
            ),
            "",
            f"Across all prior screened families, {unstable_hyperparameters}/{len(prior_hyperparameter_stability)} "
            "family-parameter pairs are not cross-panel stable and "
            f"{boundary_hyperparameters}/{len(prior_hyperparameter_stability)} select a grid boundary in at least half "
            "of panels. The complete 89-row table is included as `prior_hyperparameter_cross_panel_stability.csv`.",
            "",
            "### Cross-fit raw-optimum stability",
            "",
            markdown_table(
                optimum_stability,
                [
                    "dataset",
                    "policy",
                    "refits",
                    "mean_predicted_bpb",
                    "sd_predicted_bpb",
                    "mean_gap_below_observed_frontier",
                    "fraction_below_observed_frontier",
                    "distance_over_fit_p95",
                    "convex_effective_support",
                ],
            ),
            "",
            "Every one of 25 refits per target and policy class assigns its frozen raw optimum a value below the "
            "observed frontier, with only 0.002--0.003 BPB prediction SD. Yet all four optima are more than twice "
            "the fit-panel 95th-percentile support radius away. The fantasy is stable across refits: ordinary "
            "coefficient variance is not the explanation for the unsupported optimum.",
            "",
            "## Optimization conclusion",
            "",
            "No tested model has a raw optimum that is simultaneously support-audited, bootstrap-stable, correctly ordered on both "
            "StarCoder schedules, and competitive on the 710-run / 690-coordinate archive. Deployment KL or trust-region penalties can choose safer "
            "policies, but they are decision constraints, not evidence that the raw surface is correct. They therefore cannot rescue a headline model.",
            "",
            "## Scientific recommendation",
            "",
            "Retain the complete Pareto baseline for descriptive prediction and candidate ranking within observed support. Do not present "
            "any current form as a solved two-phase scaling law, and do not optimize an unconstrained raw surrogate into deployment. The "
            "next informative evidence is a paired aggregate/phase-contrast intervention: hold \\(\\bar w\\) fixed and vary "
            "\\(d=w^{(1)}-w^{(0)}\\) around independently identified one-phase anchors. That design identifies the phase transition "
            "without asking a random two-phase swarm to identify aggregate utility and phase order simultaneously.",
            "",
            "The preregistered future panel in `future_confirmation_preregistration.md` is inactive because no candidate is eligible. "
            "Its signed simplex-ray construction, deterministic family directions, 86-policy/170-run arithmetic, and "
            "no-clipping assertions are machine-validated. If a future model passes every development gate, that panel "
            "provides the untouched confirmation needed before any provisional recommendation becomes a paper claim.",
            "",
            "### Confirmation power audit",
            "",
            markdown_table(confirmation_noise),
            "",
            markdown_table(
                threshold_power,
                [
                    "target",
                    "repeats_per_arm",
                    "effect_bpb",
                    "power",
                    "minimum_detectable_effect_at_80pct_power",
                ],
            ),
            "",
            "Ten independent exact-policy repeat groups imply pooled within-policy SD 0.00082 on Uncheatable and "
            "0.00302 on Table-9. Three repeats per arm have only 0.514 power for the frozen 0.005-BPB Table-9 "
            "improvement threshold; six raise uncorrected point-estimate power to 0.847. That six-repeat result is "
            "superseded below by the multiplicity audit. The Table-9 "
            "variance interval remains wide, so any later variance-based adjustment must be blinded and frozen before "
            "treatment-arm outcomes are inspected.",
            "",
            "### Repeat-noise influence audit",
            "",
            markdown_table(repeat_noise_influence),
            "",
            "Leaving out each exact-policy repeat group changes pooled SD by at most 10.4%. The worst Table-9 "
            "leave-one-group-out estimate still gives 0.803 uncorrected power at six repeats per arm, so the nuisance "
            "estimate is not determined by one unusually noisy policy. This is a robustness check on the future design, not "
            "evidence for any surrogate.",
            "",
            "### Confirmation multiplicity audit",
            "",
            markdown_table(multiplicity_required_repeats),
            "",
            "Claiming superiority on either of two targets requires family-wise control. The frozen design uses Holm's "
            "two-target procedure; when only one target improves, its first threshold is alpha=0.025. Seven Table-9 "
            "repeats per arm provide 0.812 power at the pooled nuisance estimate, but 15 are required at its upper 95% "
            "bound. The inactive panel therefore fixes 15 repeats for each decisive arm. The two simultaneous frontier "
            "noninferiority conditions form an intersection-union test and retain one-sided 95% bounds. Single-seed rays "
            "are surface diagnostics and cannot be searched for a replacement winner after unsealing.",
            "",
            "## Deliverables",
            "",
            "- `executive_summary.md`: concise terminal decision, decisive evidence, present-use boundary, and next experiment.",
            "- `data_dictionary.md`: metric signs, policy/refit semantics, support conventions, and structural missingness.",
            "- `metric_reproduction_summary.csv` and `metric_reproduction.csv`: independent row-level reproduction of decision-critical metrics.",
            "- `all_3e18_row_predictions.csv`, `adversarial_row_predictions.csv`, and `row_prediction_summary.csv`: every baseline residual with policy and proposal provenance.",
            "- `deliverable_traceability.csv`: machine-checked map from every requested terminal deliverable to its evidence.",
            "- `../approach_registry.csv`: complete 99-route registry with exact terminal evidence.",
            "- `../data_use_ledger.csv`: append-only data-use ledger.",
            "- `acceptance_gate_evaluation.csv`: every route mapped to its terminal gate; no route passes all gates.",
            "- `heldout_pareto_baseline.csv`: comparable 710-run archive metrics.",
            "- `heldout_provenance_index.csv`, `excluded_coordinate_aliases.csv`, and `coordinate_repeat_groups.csv`: exact archive identity and alias accounting.",
            "- `adversarial_provenance.csv`: frozen target, policy, selection-stratum, and proposal metadata for all 120 rows.",
            "- `coordinate_balanced_metrics.csv` and `coordinate_balanced_comparison.csv`: 710-run versus 690-coordinate sensitivity.",
            "- `baseline_complexity.csv`: nominal parameter counts by swarm, target, policy class, and model.",
            "- `paired_bootstrap_comparisons.csv` and `calibration_bins.csv`: frozen uncertainty and calibration diagnostics.",
            "- `support_stratified_metrics.csv` and `support_degradation_summary.csv`: heldout error, calibration, and optimism by support quartile.",
            "- `support_abstention_metrics.csv` and `maximum_safe_coverage.csv`: fixed nearest-support coverage tradeoffs; operational diagnostic only.",
            "- `worst_exposure_feature_correlations.csv`, `worst_exposure_feature_summary.csv`, and `worst_optimism_rows.csv`: policy diagnostics for every frozen baseline's worst archive errors.",
            "- `frontier_phase_delta_summaries.csv`, `one_phase_rank_phase_delta_correlations.csv`, and `phase_benefit_sign_transitions.csv`: matched-policy frontier and phase-effect diagnostics.",
            "- `repeat_noise_estimates.csv`, `confirmation_power.csv`, and `required_repeats.csv`: empirical repeat noise and future-confirmation power.",
            "- `repeat_noise_leave_one_group_out.csv` and `repeat_noise_influence_summary.csv`: nuisance-variance influence audit for the future confirmation budget.",
            "- `adversarial_target_matched_metrics.csv`, `adversarial_cross_target_metrics.csv`, and `adversarial_proposal_strata_metrics.csv`.",
            "- `adversarial_stratum_winners.csv`: diagnostic winners for every candidate-target, policy-class, proposer, origin, and selection slice.",
            "- `one_phase_restriction_comparison.csv` and `low_tail_influence_metrics.csv`.",
            "- `policy_class_metrics.csv`, `policy_class_winners.csv`, and `policy_class_rank_transfer.csv`: "
            "frozen one-phase versus two-phase heldout diagnostics.",
            "- `two_stage_design_identifiability.csv`: equal-budget aggregate/phase-contrast design audit.",
            "- `mechanism_coverage.csv`: route counts by independent mechanism class.",
            "- `route_ledger_coverage.csv` and `superseded_batch_markers.csv`: append-only chronology and reconciliation audit.",
            "- `cross_scale_variance_decomposition.csv` and `cross_scale_component_transfer.csv`: aggregate/phase-effect variance and transfer.",
            "- `cross_scale_noise_inputs.csv` and `phase_transfer_deattenuation.csv`: errors-in-variables bound on phase-effect transfer.",
            "- `cross_target_component_transfer.csv` and `phase_attenuation_bootstrap.csv`: shared phase-signal and target-specific attenuation audit.",
            "- `adversarial_stratum_robustness.csv`: worst calibration, rank, error, and regret within each proposal stratum.",
            "- `prior_parameter_identifiability.csv`, `prior_hyperparameter_cross_panel_stability.csv`, "
            "`prior_candidate_active_set_complexity.csv`, `prior_raw_optimum_convex_support.csv`, and "
            "`prior_raw_optimum_crossfit_summary.csv`: provenance-checked inherited complexity and stability evidence.",
            "- `evidence_index.csv`: links to all surface, calibration, scale-transfer, residual, and optimum artifacts.",
            "- `future_confirmation_preregistration.md` and `.json`.",
            "- `confirmation_design_checks.csv`: machine-validated inactive panel specification and count arithmetic.",
            "- `source_inventory.csv`: SHA-256 inventory, purpose, entry-point status, and PEP 723 metadata for every rerunnable source file.",
            "- `phase_reversal_observability_summary.csv`: exact reflected-contrast support and odd/even identifiability.",
            "- `multiplicity_adjusted_required_repeats.csv` and `multiplicity_adjusted_power.csv`: Holm-adjusted confirmation power.",
            "- `heldout_pareto_tradeoffs.html`, `adversarial_compression_tradeoffs.html`, and `one_phase_restriction_transfer.html`.",
            "",
            "The [previous 41-route report](../../mechanistic_surrogate_discovery_20260717/final_synthesis/final_report.md) "
            "remains part of the evidence chain.",
        ]
    )


def build_executive_summary(decisive: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Mechanistic surrogate discovery II: decision memo",
            "",
            "## Decision",
            "",
            "**No headline surrogate is recommended.** All 58 new mechanisms and 41 inherited routes failed at "
            "least one frozen gate; none was eligible for independent review or untouched confirmation.",
            "",
            "This is a model-identification negative result, not evidence that two-phase schedules are useless. The "
            "random swarms identify ordinary ranking reasonably well but do not identify a reliable raw optimum.",
            "",
            "## Evidence",
            "",
            markdown_table(decisive),
            "",
            "- The closest new route, orthogonal gradient-Gram transport, had a stable interior phase parameter but "
            "failed both StarCoder shape gates, predicted the wrong schedule ordering, and put its raw optima far from "
            "the observed best policies.",
            "- On 710 development runs, different incumbents win RMSE, Regret@1, and calibration. On the exposed "
            "adversarial panel, those winners change again and every model has a selection stratum with calibration "
            "slope below 0.5 on both targets.",
            "- All four inherited raw-optimum refit families predict optima below the observed frontier in 25/25 "
            "refits, yet those optima are more than twice the fit-panel 95th-percentile support radius away. Stable "
            "fantasy values are therefore not evidence of a stable decision.",
            "- Exact phase-reversal contrasts are absent from the 39-bucket random swarm, and aggregate and phase "
            "contrast are strongly correlated. The data cannot nonparametrically separate odd order effects from even "
            "phase-variation costs.",
            "",
            "## Use now",
            "",
            "Keep the Pareto baseline for descriptive interpolation and support-aware candidate ranking. Do not claim "
            "a solved two-phase scaling law, and do not treat deployment KL or trust-region constraints as evidence "
            "that a raw surrogate surface is correct.",
            "",
            "## Resolve next",
            "",
            "The next discriminating experiment holds aggregate mixture fixed and applies signed phase contrasts around "
            "independently fitted one-phase anchors. The inactive preregistration specifies 86 unique policies, at most "
            "170 runs, exact no-clipping reconstruction, deterministic family directions, and 15 decisive repeats per "
            "arm under Holm control. It must remain inactive until a genuinely new candidate passes every development gate.",
            "",
            "Any future survivor remains provisional until it passes a newly sealed panel; no result from this drive is "
            "confirmatory.",
        ]
    )


def build_data_dictionary() -> str:
    return "\n".join(
        [
            "# Final-synthesis data dictionary",
            "",
            "## Metric conventions",
            "",
            "- BPB is lower-is-better throughout.",
            "- `bias_predicted_minus_observed` and `selected_optimism` are prediction minus observation; negative values are optimistic.",
            "- `worst_optimism` is observation minus prediction at the most optimistic row; larger positive values are worse.",
            "- `optimism_gt_0p05_count` counts rows for which observation minus prediction exceeds 0.05 BPB.",
            "- `observed_on_predicted_slope` is the ordinary least-squares slope in `observed = intercept + slope * predicted`. Values below one indicate response compression.",
            "- `regret_at_k` is the best observed BPB among the model's predicted top-k policies minus the best observed BPB in that evaluation slice.",
            "- `lower_tail_optimism` and `low_tail_rmse` are computed on the fixed predicted lower tail, not an observed-outcome-selected subset.",
            "- Spearman correlations are undefined for two-row strata and are stored as `NaN`, rather than assigned a misleading rank score.",
            "",
            "## Policy and fit semantics",
            "",
            "- `one_phase` is a phase-tied policy; `two_phase` permits independent phase mixtures under the stated phase fractions.",
            "- `algebraic_restriction_of_two_phase_fit` evaluates a two-phase fit on tied inputs without refitting its parameters.",
            "- `independent_one_phase_refit` fits the restricted functional form directly to one-phase data. It is the fitted one-phase ablation.",
            "- `candidate_target` is the target used to generate an adversarial policy. Rows where evaluation `target` differs are cross-target transfer evidence and are never pooled with target-matched selection metrics.",
            "- The 710-row archive contains 690 unique policy hashes. Coordinate-balanced tables give each unique policy equal weight; run-balanced tables preserve independent repeats.",
            "",
            "## Support and optimization semantics",
            "",
            "- Support distance is computed in the frozen policy representation used by the source diagnostic; smaller means closer to the fit panel.",
            "- `distance_over_fit_p95` divides an optimum's nearest-support distance by the fit panel's 95th-percentile distance.",
            "- `phase_tv` is total-variation distance between phase mixtures and is dimensionless.",
            "- `max_simulated_epoch` is realized token exposure divided by the corresponding bucket's declared token pool.",
            "- Raw optima are unconstrained surrogate optima. Deployment KL or trust-region choices are reported separately and never count as evidence that a raw response surface is correct.",
            "",
            "## Structural missingness",
            "",
            "- `one_phase_restriction_comparison.csv` has `NaN` Spearman only for two-row adversarial strata, where rank correlation is not informative.",
            "- `phase_reversal_observability_summary.csv` combines two different audits: exact StarCoder reversal pairs and Delphi reflected-policy feasibility. Columns belonging only to the other audit are intentionally `NaN`.",
            "- `two_stage_design_identifiability.csv` reports singular values for individual aggregate/contrast blocks and canonical correlations for joint rows; non-applicable fields are intentionally `NaN`.",
            "- `cross_target_component_transfer.csv` has undefined correlations only when a component is constant under the compared policy class.",
            "- No generated CSV contains an infinite numeric value or a duplicated full row; the bundle validator enforces the decision-critical schemas separately.",
            "",
            "## Evidence status",
            "",
            "- `fit` and ordinary grouped-OOF quantities are interpolation diagnostics.",
            "- Historical and exposed-adversarial quantities are development/falsification evidence, not confirmatory evidence.",
            "- The future confirmation design is inactive. No sealed outcome was read in this drive.",
        ]
    )


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(REGISTRY).fillna("")
    ledger = pd.read_csv(LEDGER).fillna("")
    source_inventory = build_source_inventory()
    metrics = pd.read_csv(FROZEN_DIR / "baseline_metrics.csv")
    restriction_metrics = pd.read_csv(OUTPUT_ROOT / "round54_single_phase_refit" / "metrics.csv")
    tail_metrics = pd.read_csv(OUTPUT_ROOT / "round55_low_tail_influence" / "metrics.csv")
    support_metrics = pd.read_csv(OUTPUT_ROOT / "round53_partial_identification" / "support_stratified_metrics.csv")
    design_metrics = pd.read_csv(OUTPUT_ROOT / "round56_two_stage_design_identifiability" / "design_metrics.csv")
    mechanism_summary = pd.read_csv(OUTPUT_ROOT / "round57_registry_mechanism_coverage" / "mechanism_gate_summary.csv")
    coordinate_metrics = pd.read_csv(
        OUTPUT_ROOT / "round59_coordinate_balanced_metrics" / "coordinate_balanced_metrics.csv"
    )
    ledger_coverage = pd.read_csv(OUTPUT_ROOT / "round60_data_use_ledger_integrity" / "route_ledger_coverage.csv")
    variance_decomposition = pd.read_csv(
        OUTPUT_ROOT / "round61_cross_scale_variance_decomposition" / "variance_decomposition.csv"
    )
    component_transfer = pd.read_csv(
        OUTPUT_ROOT / "round61_cross_scale_variance_decomposition" / "component_scale_transfer.csv"
    )
    stratum_robustness = pd.read_csv(
        OUTPUT_ROOT / "round62_adversarial_strata_robustness" / "stratum_robustness_summary.csv"
    )
    prior_stability_dir = OUTPUT_ROOT / "round63_prior_stability_carryforward"
    prior_identifiability = pd.read_csv(prior_stability_dir / "parameter_identifiability.csv")
    prior_hyperparameter_stability = pd.read_csv(prior_stability_dir / "hyperparameter_cross_panel_stability.csv")
    prior_complexity = pd.read_csv(prior_stability_dir / "candidate_active_set_complexity.csv")
    prior_support = pd.read_csv(prior_stability_dir / "raw_optimum_convex_support.csv")
    prior_crossfit = pd.read_csv(prior_stability_dir / "raw_optimum_crossfit_summary.csv")
    policy_class_dir = OUTPUT_ROOT / "round64_policy_class_robustness"
    policy_class_metrics = pd.read_csv(policy_class_dir / "policy_class_metrics.csv")
    policy_class_winners = pd.read_csv(policy_class_dir / "policy_class_winners.csv")
    policy_class_transfer = pd.read_csv(policy_class_dir / "policy_class_rank_transfer.csv")
    exposure_pattern_dir = OUTPUT_ROOT / "round65_worst_exposure_patterns"
    exposure_pattern_summary = pd.read_csv(exposure_pattern_dir / "feature_correlation_summary.csv")
    frontier_phase_dir = OUTPUT_ROOT / "round66_frontier_phase_benefit"
    frontier_phase_benefit = pd.read_csv(frontier_phase_dir / "frontier_phase_delta_summaries.csv")
    confirmation_power_dir = OUTPUT_ROOT / "round67_confirmation_power"
    confirmation_noise = pd.read_csv(confirmation_power_dir / "repeat_noise_estimates.csv")
    confirmation_power = pd.read_csv(confirmation_power_dir / "confirmation_power.csv")
    support_abstention_dir = OUTPUT_ROOT / "round68_support_abstention"
    support_abstention_metrics = pd.read_csv(support_abstention_dir / "support_abstention_metrics.csv")
    maximum_safe_coverage = pd.read_csv(support_abstention_dir / "maximum_safe_coverage.csv")
    repeat_influence_dir = OUTPUT_ROOT / "round69_repeat_noise_influence"
    repeat_noise_influence = pd.read_csv(repeat_influence_dir / "repeat_noise_influence_summary.csv")
    measurement_error_dir = OUTPUT_ROOT / "round70_cross_scale_measurement_error"
    phase_transfer_deattenuation = pd.read_csv(measurement_error_dir / "phase_transfer_deattenuation.csv")
    cross_target_dir = OUTPUT_ROOT / "round71_cross_target_phase_state"
    cross_target_phase_state = pd.read_csv(cross_target_dir / "cross_target_component_transfer.csv")
    phase_attenuation_bootstrap = pd.read_csv(cross_target_dir / "phase_attenuation_bootstrap.csv")
    confirmation_design_dir = OUTPUT_ROOT / "round72_future_confirmation_design"
    confirmation_design_checks = pd.read_csv(confirmation_design_dir / "confirmation_design_checks.csv")
    phase_reversal_dir = OUTPUT_ROOT / "round73_phase_reversal_observability"
    phase_reversal_observability = pd.read_csv(phase_reversal_dir / "phase_reversal_observability_summary.csv")
    multiplicity_dir = OUTPUT_ROOT / "round74_confirmation_multiplicity"
    multiplicity_required_repeats = pd.read_csv(multiplicity_dir / "multiplicity_adjusted_required_repeats.csv")
    metric_reproduction_dir = OUTPUT_ROOT / "round76_final_metric_reproduction"
    metric_reproduction = pd.read_csv(metric_reproduction_dir / "summary.csv")
    row_prediction_dir = OUTPUT_ROOT / "round77_final_row_predictions"
    row_prediction_summary = pd.read_csv(row_prediction_dir / "row_prediction_summary.csv")
    deliverable_traceability_dir = OUTPUT_ROOT / "round78_deliverable_traceability"

    required_fields = [column for column in registry.columns if registry[column].astype(str).str.strip().eq("").any()]
    if required_fields:
        raise ValueError(f"Incomplete registry fields: {required_fields}")
    if registry["id"].duplicated().any() or registry["family"].duplicated().any():
        raise ValueError("Registry IDs and family names must be unique")
    if set(registry["status"]) - set(STATUS_STAGE):
        raise ValueError(f"Unknown registry statuses: {set(registry['status']) - set(STATUS_STAGE)}")

    acceptance = build_acceptance_table(registry, ledger)
    archive, target_matched, cross_target = build_pareto_tables(metrics)
    decisive = decisive_table(archive, target_matched)
    adversarial_strata = pd.read_csv(FROZEN_DIR / "adversarial_strata_metrics.csv")
    adversarial_stratum_winners = adversarial_stratum_winner_table(adversarial_strata)

    acceptance.to_csv(FINAL_DIR / "acceptance_gate_evaluation.csv", index=False)
    archive.to_csv(FINAL_DIR / "heldout_pareto_baseline.csv", index=False)
    target_matched.to_csv(FINAL_DIR / "adversarial_target_matched_metrics.csv", index=False)
    cross_target.to_csv(FINAL_DIR / "adversarial_cross_target_metrics.csv", index=False)
    adversarial_strata.to_csv(FINAL_DIR / "adversarial_proposal_strata_metrics.csv", index=False)
    adversarial_stratum_winners.to_csv(FINAL_DIR / "adversarial_stratum_winners.csv", index=False)
    restriction_metrics.to_csv(FINAL_DIR / "one_phase_restriction_comparison.csv", index=False)
    tail_metrics.to_csv(FINAL_DIR / "low_tail_influence_metrics.csv", index=False)
    support_metrics.to_csv(FINAL_DIR / "support_stratified_metrics.csv", index=False)
    support_degradation = support_degradation_table(support_metrics)
    support_degradation.to_csv(FINAL_DIR / "support_degradation_summary.csv", index=False)
    for source_name, final_name in (
        ("feature_correlations.csv", "worst_exposure_feature_correlations.csv"),
        ("feature_correlation_summary.csv", "worst_exposure_feature_summary.csv"),
        ("top10_feature_enrichment.csv", "worst_exposure_top10_enrichment.csv"),
        ("worst_optimism_rows.csv", "worst_optimism_rows.csv"),
    ):
        pd.read_csv(exposure_pattern_dir / source_name).to_csv(FINAL_DIR / final_name, index=False)
    for source_name, final_name in (
        ("frontier_phase_delta_summaries.csv", "frontier_phase_delta_summaries.csv"),
        ("one_phase_rank_phase_delta_correlations.csv", "one_phase_rank_phase_delta_correlations.csv"),
        ("phase_benefit_sign_transitions.csv", "phase_benefit_sign_transitions.csv"),
    ):
        pd.read_csv(frontier_phase_dir / source_name).to_csv(FINAL_DIR / final_name, index=False)
    for source_name in ("repeat_noise_estimates.csv", "confirmation_power.csv", "required_repeats.csv"):
        pd.read_csv(confirmation_power_dir / source_name).to_csv(FINAL_DIR / source_name, index=False)
    support_abstention_metrics.to_csv(FINAL_DIR / "support_abstention_metrics.csv", index=False)
    maximum_safe_coverage.to_csv(FINAL_DIR / "maximum_safe_coverage.csv", index=False)
    for source_name in ("repeat_noise_leave_one_group_out.csv", "repeat_noise_influence_summary.csv"):
        pd.read_csv(repeat_influence_dir / source_name).to_csv(FINAL_DIR / source_name, index=False)
    for source_name in ("cross_scale_noise_inputs.csv", "phase_transfer_deattenuation.csv"):
        pd.read_csv(measurement_error_dir / source_name).to_csv(FINAL_DIR / source_name, index=False)
    for source_name in ("cross_target_component_transfer.csv", "phase_attenuation_bootstrap.csv"):
        pd.read_csv(cross_target_dir / source_name).to_csv(FINAL_DIR / source_name, index=False)
    confirmation_design_checks.to_csv(FINAL_DIR / "confirmation_design_checks.csv", index=False)
    phase_reversal_observability.to_csv(FINAL_DIR / "phase_reversal_observability_summary.csv", index=False)
    multiplicity_required_repeats.to_csv(FINAL_DIR / "multiplicity_adjusted_required_repeats.csv", index=False)
    pd.read_csv(multiplicity_dir / "multiplicity_adjusted_power.csv").to_csv(
        FINAL_DIR / "multiplicity_adjusted_power.csv", index=False
    )
    metric_reproduction.to_csv(FINAL_DIR / "metric_reproduction_summary.csv", index=False)
    pd.read_csv(metric_reproduction_dir / "metric_reproduction.csv").to_csv(
        FINAL_DIR / "metric_reproduction.csv", index=False
    )
    for source_name in (
        "all_3e18_row_predictions.csv",
        "adversarial_row_predictions.csv",
        "row_prediction_summary.csv",
    ):
        pd.read_csv(row_prediction_dir / source_name).to_csv(FINAL_DIR / source_name, index=False)
    pd.read_csv(deliverable_traceability_dir / "deliverable_traceability.csv").to_csv(
        FINAL_DIR / "deliverable_traceability.csv", index=False
    )
    design_metrics.to_csv(FINAL_DIR / "two_stage_design_identifiability.csv", index=False)
    mechanism_summary.to_csv(FINAL_DIR / "mechanism_coverage.csv", index=False)
    ledger_coverage.to_csv(FINAL_DIR / "route_ledger_coverage.csv", index=False)
    pd.read_csv(OUTPUT_ROOT / "round60_data_use_ledger_integrity" / "superseded_batch_markers.csv").to_csv(
        FINAL_DIR / "superseded_batch_markers.csv", index=False
    )
    variance_decomposition.to_csv(FINAL_DIR / "cross_scale_variance_decomposition.csv", index=False)
    component_transfer.to_csv(FINAL_DIR / "cross_scale_component_transfer.csv", index=False)
    stratum_robustness.to_csv(FINAL_DIR / "adversarial_stratum_robustness.csv", index=False)
    coordinate_metrics.to_csv(FINAL_DIR / "coordinate_balanced_metrics.csv", index=False)
    pd.read_csv(OUTPUT_ROOT / "round59_coordinate_balanced_metrics" / "coordinate_balanced_comparison.csv").to_csv(
        FINAL_DIR / "coordinate_balanced_comparison.csv", index=False
    )
    for name in (
        "heldout_provenance_index.csv",
        "excluded_coordinate_aliases.csv",
        "coordinate_repeat_groups.csv",
        "archive_split_summary.csv",
        "adversarial_provenance.csv",
    ):
        pd.read_csv(OUTPUT_ROOT / "round58_heldout_provenance" / name).to_csv(FINAL_DIR / name, index=False)
    decisive.to_csv(FINAL_DIR / "diagnostic_winners.csv", index=False)
    complexity = (
        metrics[["swarm", "target", "policy", "model", "parameter_count"]]
        .drop_duplicates()
        .sort_values(["swarm", "target", "policy", "model"])
    )
    complexity.to_csv(FINAL_DIR / "baseline_complexity.csv", index=False)
    pd.read_csv(FROZEN_DIR / "paired_bootstrap_comparisons.csv").to_csv(
        FINAL_DIR / "paired_bootstrap_comparisons.csv", index=False
    )
    pd.read_csv(FROZEN_DIR / "calibration_bins.csv").to_csv(FINAL_DIR / "calibration_bins.csv", index=False)
    for source_name, final_name in (
        ("parameter_identifiability.csv", "prior_parameter_identifiability.csv"),
        ("hyperparameter_cross_panel_stability.csv", "prior_hyperparameter_cross_panel_stability.csv"),
        ("candidate_active_set_complexity.csv", "prior_candidate_active_set_complexity.csv"),
        ("raw_optimum_convex_support.csv", "prior_raw_optimum_convex_support.csv"),
        ("raw_optimum_crossfit_summary.csv", "prior_raw_optimum_crossfit_summary.csv"),
        ("heldout_calibration_bootstrap.csv", "prior_heldout_calibration_bootstrap.csv"),
        ("source_manifest.csv", "prior_stability_source_manifest.csv"),
    ):
        pd.read_csv(prior_stability_dir / source_name).to_csv(FINAL_DIR / final_name, index=False)
    policy_class_metrics.to_csv(FINAL_DIR / "policy_class_metrics.csv", index=False)
    policy_class_winners.to_csv(FINAL_DIR / "policy_class_winners.csv", index=False)
    policy_class_transfer.to_csv(FINAL_DIR / "policy_class_rank_transfer.csv", index=False)

    followup = OUTPUT_ROOT / "round55_low_tail_influence" / "numerical_followup" / "convergence_summary.csv"
    convergence = pd.read_csv(followup) if followup.exists() else None
    if convergence is not None:
        convergence.to_csv(FINAL_DIR / "low_tail_optimum_convergence.csv", index=False)

    evidence_rows = [
        ("frozen gate", FROZEN_DIR / "report.md", "Frozen Pareto baseline and immutable acceptance thresholds."),
        ("prior drive", PRIOR_FINAL, "The 41 previously rejected routes and their terminal evidence."),
        (
            "closest new route",
            OUTPUT_ROOT / "round52_orthogonal_gradient_gram_starcoder" / "report.md",
            "Orthogonal gradient-Gram transport near miss.",
        ),
        (
            "StarCoder cosine surface",
            OUTPUT_ROOT / "round52_orthogonal_gradient_gram_starcoder" / "starcoder_cosine_50_50__surface.html",
            "Closest new route on cosine 50/50.",
        ),
        (
            "StarCoder WSD surface",
            OUTPUT_ROOT / "round52_orthogonal_gradient_gram_starcoder" / "starcoder_wsd_80_20__surface.html",
            "Closest new route on WSD 80/20.",
        ),
        (
            "scale transfer",
            OUTPUT_ROOT / "round1_cross_scale_matched_policy" / "matched_policy_scale_transfer.html",
            "Matched 300M/Delphi scale-transfer audit.",
        ),
        (
            "phase identifiability",
            OUTPUT_ROOT / "round18_phase_identification_ceiling" / "report.md",
            "Effective design dimension, learning curves, and cross-scale coefficient transfer.",
        ),
        (
            "phase coefficient stability",
            OUTPUT_ROOT / "round18_phase_identification_ceiling" / "coefficient_stability.csv",
            "Foldwise coefficient cosine, sign agreement, and norm stability.",
        ),
        (
            "partial identification",
            OUTPUT_ROOT / "round53_partial_identification" / "report.md",
            "Finite-design theorem, support, and decision disagreement.",
        ),
        (
            "support calibration",
            OUTPUT_ROOT / "round53_partial_identification" / "support_stratified_calibration.html",
            "Heldout calibration by support quartile.",
        ),
        (
            "model disagreement",
            OUTPUT_ROOT / "round53_partial_identification" / "model_disagreement_vs_error.html",
            "Disagreement as warning rather than surrogate.",
        ),
        (
            "one-phase refit",
            OUTPUT_ROOT / "round54_single_phase_refit" / "report.md",
            "Algebraic tying versus independent one-phase fitting.",
        ),
        (
            "one-phase calibration",
            OUTPUT_ROOT / "round54_single_phase_refit" / "historical_one_phase_calibration.html",
            "Historical one-phase calibration.",
        ),
        (
            "low-tail influence",
            OUTPUT_ROOT / "round55_low_tail_influence" / "report.md",
            "Frozen low-observed-tail deletion sensitivity.",
        ),
        (
            "low-tail optimum",
            OUTPUT_ROOT / "round55_low_tail_influence" / "raw_optimum_sensitivity.html",
            "Raw optimum sensitivity under tail deletion.",
        ),
        (
            "same-budget design identifiability",
            OUTPUT_ROOT / "round56_two_stage_design_identifiability" / "report.md",
            "Coordinate-only comparison of random and signed phase-fiber designs at equal row count.",
        ),
        (
            "same-budget design visualization",
            OUTPUT_ROOT / "round56_two_stage_design_identifiability" / "design_identifiability.html",
            "Singular spectra and aggregate/contrast canonical correlations.",
        ),
        (
            "mechanism coverage",
            OUTPUT_ROOT / "round57_registry_mechanism_coverage" / "report.md",
            "Exact route taxonomy, gate coverage, and near-duplicate screen.",
        ),
        (
            "heldout provenance",
            OUTPUT_ROOT / "round58_heldout_provenance" / "report.md",
            "Run, coordinate, alias, target, and adversarial proposal provenance.",
        ),
        (
            "coordinate-balanced sensitivity",
            OUTPUT_ROOT / "round59_coordinate_balanced_metrics" / "report.md",
            "Equal policy-coordinate versus equal run-row metric weighting.",
        ),
        (
            "data-use ledger integrity",
            OUTPUT_ROOT / "round60_data_use_ledger_integrity" / "report.md",
            "Append-only chronology, historical reconciliation, and adversarial-use boundary audit.",
        ),
        (
            "cross-scale variance decomposition",
            OUTPUT_ROOT / "round61_cross_scale_variance_decomposition" / "report.md",
            "Exact aggregate and paired phase-effect decomposition on 238 matched policies.",
        ),
        (
            "cross-scale component transfer",
            OUTPUT_ROOT / "round61_cross_scale_variance_decomposition" / "component_scale_transfer.html",
            "Matched-policy aggregate, phase-effect, and total transfer visualization.",
        ),
        (
            "adversarial stratum robustness",
            OUTPUT_ROOT / "round62_adversarial_strata_robustness" / "report.md",
            "Worst-case calibration, rank, error, and regret across proposal strata.",
        ),
        (
            "adversarial stratum visualization",
            OUTPUT_ROOT / "round62_adversarial_strata_robustness" / "stratum_robustness.html",
            "Worst-stratum RMSE versus calibration for every baseline.",
        ),
        (
            "prior complexity and optimum stability",
            OUTPUT_ROOT / "round63_prior_stability_carryforward" / "report.md",
            "Provenance-checked inherited effective-complexity, identifiability, and raw-optimum cross-fit evidence.",
        ),
        (
            "policy-class heldout robustness",
            OUTPUT_ROOT / "round64_policy_class_robustness" / "report.md",
            "Frozen model-rank and diagnostic-winner transfer between one-phase and two-phase heldouts.",
        ),
        (
            "policy-class RMSE transfer visualization",
            OUTPUT_ROOT / "round64_policy_class_robustness" / "policy_class_rmse_transfer.html",
            "Historical and target-matched adversarial one-phase versus two-phase RMSE.",
        ),
        (
            "worst-residual exposure patterns",
            OUTPUT_ROOT / "round65_worst_exposure_patterns" / "report.md",
            "Frozen-baseline optimism concentration in support distance, phase divergence, repetition, and aggregate tilt.",
        ),
        (
            "worst-residual exposure visualization",
            OUTPUT_ROOT / "round65_worst_exposure_patterns" / "worst_exposure_pattern_diagnostics.html",
            "Correlations and exact worst-optimism summaries for every frozen baseline.",
        ),
        (
            "frontier phase benefit",
            OUTPUT_ROOT / "round66_frontier_phase_benefit" / "report.md",
            "Fixed one-phase frontier slices and paired two-minus-one-phase effects across scales.",
        ),
        (
            "frontier phase benefit visualization",
            OUTPUT_ROOT / "round66_frontier_phase_benefit" / "frontier_phase_benefit.html",
            "Cross-scale-ranked paired phase effects with fixed bootstrap intervals.",
        ),
        (
            "future confirmation power",
            OUTPUT_ROOT / "round67_confirmation_power" / "report.md",
            "Independent exact-policy repeat noise and power for the frozen 0.005-BPB threshold.",
        ),
        (
            "future confirmation power visualization",
            OUTPUT_ROOT / "round67_confirmation_power" / "confirmation_power.html",
            "One-sided power by target, effect size, and independent repeats per policy arm.",
        ),
        (
            "support abstention",
            OUTPUT_ROOT / "round68_support_abstention" / "report.md",
            "Fixed nearest-support coverage tradeoff, treated as a deployment diagnostic rather than a response model.",
        ),
        (
            "support abstention visualization",
            OUTPUT_ROOT / "round68_support_abstention" / "support_abstention_regret.html",
            "Coverage versus local and global archive Regret@1 under fixed support abstention.",
        ),
        (
            "repeat-noise influence",
            OUTPUT_ROOT / "round69_repeat_noise_influence" / "report.md",
            "Leave-one-policy-group-out robustness of the future-confirmation repeat budget.",
        ),
        (
            "repeat-noise influence visualization",
            OUTPUT_ROOT / "round69_repeat_noise_influence" / "repeat_noise_influence.html",
            "Pooled nuisance variance and six-repeat power after omitting each exact-policy group.",
        ),
        (
            "cross-scale measurement error",
            OUTPUT_ROOT / "round70_cross_scale_measurement_error" / "report.md",
            "Point and upper-confidence errors-in-variables bounds for matched phase-effect transfer.",
        ),
        (
            "cross-target phase state",
            OUTPUT_ROOT / "round71_cross_target_phase_state" / "report.md",
            "Cross-target phase-effect signs, correlations, and target-specific scale attenuation.",
        ),
        (
            "cross-target phase-state visualization",
            OUTPUT_ROOT / "round71_cross_target_phase_state" / "cross_target_phase_effects.html",
            "Matched-policy Uncheatable and Table-9 phase-effect scatter at both scales.",
        ),
        (
            "future confirmation design specification",
            OUTPUT_ROOT / "round72_future_confirmation_design" / "report.md",
            "Machine-validated algebra, deterministic direction rules, and panel-count arithmetic.",
        ),
        (
            "phase-reversal observability",
            OUTPUT_ROOT / "round73_phase_reversal_observability" / "report.md",
            "Exact odd/even phase-effect invariant and current-design observability audit.",
        ),
        (
            "phase-reversal support visualization",
            OUTPUT_ROOT / "round73_phase_reversal_observability" / "delphi_reflection_support.html",
            "Distances from the random 39-bucket design to algebraic reflected contrasts.",
        ),
        (
            "confirmation multiplicity",
            OUTPUT_ROOT / "round74_confirmation_multiplicity" / "report.md",
            "Family-wise superiority testing and conservative repeat-power correction.",
        ),
        (
            "multiplicity-adjusted power visualization",
            OUTPUT_ROOT / "round74_confirmation_multiplicity" / "multiplicity_adjusted_power.html",
            "Power after the frozen two-target Holm correction under point and upper-bound noise.",
        ),
        (
            "independent metric reproduction",
            OUTPUT_ROOT / "round76_final_metric_reproduction" / "report.md",
            "Independent reconstruction of archive and target-matched adversarial decision metrics.",
        ),
        (
            "terminal row-level predictions",
            OUTPUT_ROOT / "round77_final_row_predictions" / "report.md",
            "Every archive prediction, residual, optimism value, and proposal stratum in a terminal export.",
        ),
        (
            "terminal deliverable traceability",
            OUTPUT_ROOT / "round78_deliverable_traceability" / "report.md",
            "Machine-checked map from every requested deliverable and boundary to terminal evidence.",
        ),
    ]
    if followup.exists():
        evidence_rows.append(
            (
                "low-tail numerical followup",
                followup.parent / "report.md",
                "High-budget multi-start convergence audit for failed raw-optimum searches.",
            )
        )
    evidence = pd.DataFrame(evidence_rows, columns=["artifact", "path", "role"])
    evidence["path"] = evidence["path"].map(lambda path: str(Path(path).resolve().relative_to(TWO_PHASE_ROOT.resolve())))
    evidence.to_csv(FINAL_DIR / "evidence_index.csv", index=False)

    write_tradeoff_plots(archive, target_matched, restriction_metrics)
    confirmation = future_confirmation_payload()
    confirmation["generated_at"] = datetime.now(UTC).isoformat()
    (FINAL_DIR / "future_confirmation_preregistration.json").write_text(
        json.dumps(confirmation, indent=2, sort_keys=True) + "\n"
    )
    (FINAL_DIR / "future_confirmation_preregistration.md").write_text(
        build_future_confirmation_report(confirmation) + "\n"
    )
    report = build_report(
        registry,
        ledger,
        archive,
        target_matched,
        cross_target,
        adversarial_stratum_winners,
        decisive,
        restriction_metrics,
        tail_metrics,
        convergence,
        design_metrics,
        mechanism_summary,
        coordinate_metrics,
        ledger_coverage,
        variance_decomposition,
        component_transfer,
        phase_transfer_deattenuation,
        cross_target_phase_state,
        phase_attenuation_bootstrap,
        stratum_robustness,
        prior_identifiability,
        prior_hyperparameter_stability,
        prior_complexity,
        prior_support,
        prior_crossfit,
        policy_class_winners,
        policy_class_transfer,
        support_degradation,
        support_abstention_metrics,
        maximum_safe_coverage,
        exposure_pattern_summary,
        frontier_phase_benefit,
        confirmation_noise,
        confirmation_power,
        repeat_noise_influence,
        phase_reversal_observability,
        multiplicity_required_repeats,
        metric_reproduction,
        row_prediction_summary,
    )
    if not report.startswith("# Mechanistic surrogate discovery II: final synthesis"):
        raise ValueError("Final report builder returned malformed content")
    (FINAL_DIR / "final_report.md").write_text(report + "\n")
    (FINAL_DIR / "executive_summary.md").write_text(build_executive_summary(decisive) + "\n")
    (FINAL_DIR / "data_dictionary.md").write_text(build_data_dictionary() + "\n")
    source_inventory.to_csv(FINAL_DIR / "source_inventory.csv", index=False)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "verdict": "no_candidate_passed",
        "route_count": len(registry),
        "new_route_count": int((~registry["id"].str.startswith("prior_")).sum()),
        "ledger_rows": len(ledger),
        "source_file_count": len(source_inventory),
        "promoted_candidate_count": 0,
        "claude_reviews_run": 0,
        "frozen_gate_digest": "c4f711312423f038ef8610950d1ae6be30ffba588648177fbf5077e6931f93be",
        "round55b_complete": followup.exists(),
        "round56_complete": True,
        "round57_complete": True,
        "round58_complete": True,
        "round59_complete": True,
        "round60_complete": True,
        "round61_complete": True,
        "round62_complete": True,
        "round63_complete": True,
        "round64_complete": True,
        "round65_complete": True,
        "round66_complete": True,
        "round67_complete": True,
        "round68_complete": True,
        "round69_complete": True,
        "round70_complete": True,
        "round71_complete": True,
        "round72_complete": True,
        "round73_complete": True,
        "round74_complete": True,
        "round75_complete": bool(ledger["round_id"].eq("round_75_final_negative_synthesis").any()),
        "round76_complete": bool(ledger["round_id"].eq("round_76_final_metric_reproduction").any()),
        "round77_complete": bool(ledger["round_id"].eq("round_77_final_row_prediction_export").any()),
        "round78_complete": bool(ledger["round_id"].eq("round_78_terminal_deliverable_traceability").any()),
        "round79_complete": bool(ledger["round_id"].eq("round_79_exposed_prediction_reconciliation").any()),
        "sealed_confirmation_outcomes_inspected": False,
    }
    (FINAL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

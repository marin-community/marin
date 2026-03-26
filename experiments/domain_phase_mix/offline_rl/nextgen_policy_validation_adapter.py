# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nextgen validation adapter for offline policy candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from experiments.domain_phase_mix.nextgen.contracts import Candidate, ValidationRecord
from experiments.domain_phase_mix.nextgen.validation import register_validation_execution_adapter
from experiments.domain_phase_mix.offline_rl.evaluate_policy_three_phase_starcoder import (
    EvaluateConfig,
    evaluate_policy,
)


@dataclass(frozen=True)
class OfflineRLPolicyValidationAdapter:
    """Validation adapter that evaluates policy candidates via chained runs."""

    wandb_entity: str = "marin-community"
    wandb_project: str = "marin"
    objective_metric: str = "eval/paloma/dolma_100_programing_languages/bpb"
    n_replicates: int = 1
    run_name_prefix: str = "pinlin_calvin_xu/data_mixture/three_phase_starcoder_nextgen_validation"
    dry_run: bool = False

    def execute(self, *, model_name: str, candidate: Candidate, output_path: str) -> ValidationRecord:
        if candidate.kind != "policy" or candidate.policy_ref is None:
            return ValidationRecord(
                candidate_id=candidate.candidate_id,
                model_name=model_name,
                status="failed",
                details={"reason": "candidate_is_not_policy"},
            )

        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        results_df = evaluate_policy(
            EvaluateConfig(
                policy_artifact_path=candidate.policy_ref.uri,
                output_dir=str(out_dir),
                n_replicates=self.n_replicates,
                wandb_entity=self.wandb_entity,
                wandb_project=self.wandb_project,
                objective_metric=self.objective_metric,
                run_name_prefix=self.run_name_prefix,
                dry_run=self.dry_run,
            )
        )
        mean_objective = float(results_df["final_objective"].mean())
        first_run_id = str(results_df.iloc[0]["phase_2_run_id"]) if len(results_df) > 0 else None

        with (out_dir / "adapter_result.json").open("w") as f:
            json.dump(
                {
                    "candidate_id": candidate.candidate_id,
                    "model_name": model_name,
                    "mean_final_objective": mean_objective,
                    "n_replicates": len(results_df),
                },
                f,
                indent=2,
                sort_keys=True,
            )

        return ValidationRecord(
            candidate_id=candidate.candidate_id,
            model_name=model_name,
            status="completed",
            wandb_run_id=first_run_id,
            metric_value=mean_objective,
            details={
                "n_replicates": len(results_df),
                "results_csv": str(out_dir / "policy_rollout_results.csv"),
            },
        )


def register_offline_rl_validation_adapter(adapter: OfflineRLPolicyValidationAdapter | None) -> None:
    """Register this adapter with nextgen validation slot execution."""
    register_validation_execution_adapter(adapter)

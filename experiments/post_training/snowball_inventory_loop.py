# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a pinned initializer for 25 updates, then score its committed HF export.

The initial baseline is cached by its complete evaluation contract. Training and
export finish before final scoring, so a verifier failure cannot destroy the
training artifact or require repeating optimizer work.
"""

import hashlib
from dataclasses import asdict, replace
from pathlib import Path

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.training.training import LevanterCheckpoint
from rigging.provenance import launch_provenance

from experiments.post_training.snowball_inventory_baseline import (
    CACHE_VERSION,
    BaselineIdentity,
    scoring_source_sha256,
)
from experiments.post_training.snowball_inventory_compare import (
    INITIALIZERS,
    EvaluationSchedule,
    Initializer,
    comparison_step,
)
from experiments.post_training.snowball_inventory_evaluate import (
    EvaluationConfig,
    InventoryEvaluation,
    SavedAnswerRecovery,
    recoverable_evaluation_recipe,
    recoverable_evaluation_step,
    run_recoverable_evaluation,
)
from experiments.post_training.snowball_inventory_panel import DevelopmentPanel
from experiments.post_training.snowball_inventory_train import CHECKPOINT_INTERVAL

RUNTIME_LOCK = Path(__file__).resolve().parents[2] / "vendor" / "skyrl" / "uv.lock"
SAVED_ANSWER_RECOVERY = SavedAnswerRecovery(per_answer_timeout=600, total_timeout=900)


def run_complete_evaluation(config: EvaluationConfig) -> InventoryEvaluation:
    """Keep incomplete diagnostics durable without caching them as successful scores."""
    result = run_recoverable_evaluation(config)
    if not result.complete:
        raise ValueError(f"Evaluation remains incomplete after bounded saved-answer grading: {result.report_uri}")
    return result


def iteration_step(
    initializer: Initializer,
    *,
    version: str,
    max_steps: int,
    model_uri: str,
    model_manifest_identity: str,
    tokenizer_template_sha256: str,
    pool_uri: str,
    pool_version: str,
    pool_sha256: str,
    validation_uri: str,
    validation_version: str,
    validation_sha256: str,
    panel: DevelopmentPanel,
    panel_uri: str,
    panel_version: str,
    panel_sha256: str,
    judge_base_url: str,
    judge_model: str,
) -> ArtifactStep[InventoryEvaluation]:
    """Bind the reusable baseline, export-first training and downstream paired score."""
    provenance = launch_provenance()
    if provenance.dirty or not provenance.tree_hash.strip():
        raise ValueError("Reusable initializer baselines require clean, identified launch provenance")
    source = INITIALIZERS[initializer]
    protocol = recoverable_evaluation_recipe(judge_base_url=judge_base_url, judge_model=judge_model)
    identity = BaselineIdentity(
        model_uri=model_uri,
        model_manifest_identity=model_manifest_identity,
        tokenizer_uri=source.repository,
        tokenizer_revision=source.revision,
        tokenizer_template_sha256=tokenizer_template_sha256,
        panel_sha256=panel_sha256,
        family_counts=panel.family_counts,
        evaluation_recipe=protocol,
        recovery_policy=asdict(SAVED_ANSWER_RECOVERY),
        runtime_tree_hash=provenance.tree_hash,
        runtime_lock_sha256=hashlib.sha256(RUNTIME_LOCK.read_bytes()).hexdigest(),
        scoring_source_sha256=scoring_source_sha256(),
    )
    initial_model = ArtifactStep.adopt(
        f"models/snowball-iteration-initializer/{model_manifest_identity}",
        CACHE_VERSION,
        model_uri,
        kind=LevanterCheckpoint,
        config={"manifest_identity": model_manifest_identity, "source": asdict(source)},
    )
    evaluation_panel = ArtifactStep.adopt(f"documents/snowball-iteration-panel/{panel_sha256}", panel_version, panel_uri)
    baseline = recoverable_evaluation_step(
        name=user_owned_name(f"evals/snowball-initializer-baseline/{identity.digest}"),
        version=CACHE_VERSION,
        model=initial_model,
        relative_path="",
        panel=evaluation_panel,
        panel_sha256=panel_sha256,
        family_counts=panel.family_counts,
        tokenizer_uri=source.repository,
        tokenizer_revision=source.revision,
        expected_model_manifest_identity=model_manifest_identity,
        expected_tokenizer_template_sha256=tokenizer_template_sha256,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
        recovery=SAVED_ANSWER_RECOVERY,
    )
    baseline = replace(baseline, run=run_complete_evaluation)
    training = comparison_step(
        initializer,
        version=version,
        model_uri=model_uri,
        pool_uri=pool_uri,
        pool_version=pool_version,
        pool_sha256=pool_sha256,
        validation_uri=validation_uri,
        validation_version=validation_version,
        validation_sha256=validation_sha256,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
        max_steps=max_steps,
        evaluation_schedule=EvaluationSchedule.AFTER_EXPORT,
        expected_model_manifest_identity=model_manifest_identity,
        expected_tokenizer_template_sha256=tokenizer_template_sha256,
    )
    # Baseline admission checks the exact initializer manifest/template before
    # allocating training GPUs. A cache hit needs no repeat evaluation.
    training = replace(training, deps=(*training.deps, baseline))
    final = recoverable_evaluation_step(
        name=user_owned_name(f"evals/snowball-iteration/{initializer.value}/{panel.value}/step-{max_steps}"),
        version=version,
        model=training,
        panel=evaluation_panel,
        panel_sha256=panel_sha256,
        family_counts=panel.family_counts,
        tokenizer_uri=source.repository,
        tokenizer_revision=source.revision,
        expected_tokenizer_template_sha256=tokenizer_template_sha256,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
        recovery=SAVED_ANSWER_RECOVERY,
    )
    return replace(final, run=run_complete_evaluation)


@click.command(help=__doc__)
@click.option("--initializer", type=click.Choice([item.value for item in Initializer]), required=True)
@click.option(
    "--max-steps", type=click.IntRange(min=CHECKPOINT_INTERVAL), default=CHECKPOINT_INTERVAL, show_default=True
)
@click.option("--model-uri", required=True)
@click.option("--model-manifest-identity", required=True)
@click.option("--tokenizer-template-sha256", required=True)
@click.option("--pool-uri", required=True)
@click.option("--pool-version", required=True)
@click.option("--pool-sha256", required=True)
@click.option("--validation-uri", required=True)
@click.option("--validation-version", required=True)
@click.option("--validation-sha256", required=True)
@click.option("--panel", type=click.Choice([item.value for item in DevelopmentPanel]), default="fast", show_default=True)
@click.option("--panel-uri", required=True)
@click.option("--panel-version", required=True)
@click.option("--panel-sha256", required=True)
@click.option("--judge-base-url", required=True)
@click.option("--judge-model", required=True)
@rl_build_options
def main(
    initializer: str,
    panel: str,
    max_steps: int,
    model_uri: str,
    model_manifest_identity: str,
    tokenizer_template_sha256: str,
    pool_uri: str,
    pool_version: str,
    pool_sha256: str,
    validation_uri: str,
    validation_version: str,
    validation_sha256: str,
    panel_uri: str,
    panel_version: str,
    panel_sha256: str,
    judge_base_url: str,
    judge_model: str,
) -> ArtifactStep[InventoryEvaluation]:
    if max_steps % CHECKPOINT_INTERVAL:
        raise click.BadParameter("max-steps must be a multiple of 25", param_hint="--max-steps")
    return iteration_step(
        Initializer(initializer),
        panel=DevelopmentPanel(panel),
        version=resolve_version("evals/snowball-iteration", None),
        max_steps=max_steps,
        model_uri=model_uri,
        model_manifest_identity=model_manifest_identity,
        tokenizer_template_sha256=tokenizer_template_sha256,
        pool_uri=pool_uri,
        pool_version=pool_version,
        pool_sha256=pool_sha256,
        validation_uri=validation_uri,
        validation_version=validation_version,
        validation_sha256=validation_sha256,
        panel_uri=panel_uri,
        panel_version=panel_version,
        panel_sha256=panel_sha256,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
    )


if __name__ == "__main__":
    main()

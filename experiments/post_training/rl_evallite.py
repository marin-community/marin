# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a published export with the model's SkyRL EvalLite workload."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

from evalrecords.records import EvalRef, ModelRef, RLAttemptRef, RunStatus
from finestore.rl.run_records import LaunchReceipt, read_attempt, read_attempt_metrics, read_evaluations
from marin.execution.artifact import is_mutable_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import SkyRLModelSource, SkyRLRunConfig, run_skyrl, skyrl_execution
from marin.rl.skyrl_records import recorded_launch
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import rl_compose
from experiments.post_training.rl_evaluation_records import SkyRLRecordedEvaluation, SkyRLSession
from experiments.post_training.rl_presets import Preset
from experiments.post_training.skyrl_evaluation import EvaluationPreset, export_evaluation_name, model_rl_attempt

if TYPE_CHECKING:
    from experiments.evaluation.pipeline import EvaluationResult


@dataclass(frozen=True)
class EvalLiteConfig:
    output_path: str
    native: SkyRLRunConfig
    rl_attempt: RLAttemptRef | None


def run_eval_lite(config: EvalLiteConfig, *, version: str) -> EvaluationResult:
    from experiments.evaluation.pipeline import EvaluationResult  # noqa: PLC0415

    skyrl = config.native
    terminal = StoragePath(skyrl.output.terminal_manifest_uri)
    if terminal.exists():
        receipt = LaunchReceipt.model_validate_json(terminal.read_bytes())
        attempt_id = receipt.config["run"]["attempt_id"]
    else:
        run_skyrl(skyrl)
        attempt_id = skyrl.attempt_id
    root = skyrl.output.export_root
    attempt = read_attempt(root, attempt_id)
    rows = read_evaluations(root, attempt_id=attempt_id, evaluation="eval/", step=-1)
    metrics = read_attempt_metrics(root, attempt_id).get(0, {})
    if attempt is None or not rows or sum(len(row.answers) for row in rows) != metrics.get("eval/all/num_attempted"):
        raise ValueError(f"EvalLite has incomplete answer records for native attempt {attempt_id}")
    launch = recorded_launch(attempt)
    if launch.run_id != skyrl.run_id or launch.model_identity != skyrl.model.identity:
        raise ValueError(f"EvalLite native attempt {attempt_id} belongs to different inputs")
    score = metrics.get("eval/all/avg_verifier_score")
    recipe = EvaluationPreset("eval-lite", SkyRLRecordedEvaluation(EvalRef(name="eval-lite", mechanism="skyrl")))
    record = recipe.evaluated(
        SkyRLSession(
            attempt=attempt,
            model=ModelRef(
                name=launch.model_name or "eval-lite", location=launch.model_uri, backend=launch.engine.backend
            ),
            rows=rows,
            rl_attempt=config.rl_attempt,
            score=score if score is not None and math.isfinite(score) else None,
        ),
        version=version,
    )
    if record.status is not RunStatus.SUCCEEDED:
        raise ValueError(f"EvalLite {record.run_id} failed: {record.error}")
    return EvaluationResult(
        path=config.output_path,
        group_id=record.group_id,
        records_prefix=str(StoragePath(record.results_path).parent.parent),
        run_ids=(record.run_id,),
        results_paths=(record.results_path,),
    )


@dataclass(frozen=True)
class EvalLite:
    preset: Preset
    model_name: str

    def evaluated(self, model: SkyRLModelSource, *, version: str) -> ArtifactStep[EvaluationResult]:
        from experiments.evaluation.pipeline import EvaluationResult  # noqa: PLC0415

        preset = self.preset
        if (placement := skyrl_execution(model.step)) is not None:
            layout = replace(
                preset.recipe.layout,
                cluster=placement.cluster,
                priority=placement.priority,
                excluded_nvlink_domains=placement.excluded_nvlink_domains,
            )
            preset = replace(preset, recipe=replace(preset.recipe, layout=layout))
        name = export_evaluation_name(model, "eval-lite")
        spec = rl_compose.run_spec(preset, name=name, init=model, model_name=self.model_name)
        native = rl_compose.run_step(preset, replace(spec, version=version))

        def build(context: StepContext) -> EvalLiteConfig:
            output_path = context.output_path
            if is_mutable_version(version) and not context.is_fingerprint:
                context = replace(context, output_path=str(StoragePath(output_path) / "attempts" / uuid.uuid4().hex))
            return EvalLiteConfig(output_path, native.build_config(context), model_rl_attempt(model, context))

        return ArtifactStep(
            name=name,
            version=version,
            artifact_type=EvaluationResult,
            run=partial(run_eval_lite, version=version),
            build_config=build,
            deps=native.deps,
            runtime_args=native.runtime_args,
        )

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind checkpoint evaluation menus to the shared curated-corpus launcher."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.training.training import LevanterCheckpoint

from experiments.evaluation.pipeline import EvaluationResult
from experiments.evaluation.pipeline import eval_step as launch_step

DEFAULT_ACCELERATOR = AcceleratorChoice(platform=Platform.TPU, tpu_type="v6e-8")


@dataclass(frozen=True)
class EvalGroup:
    name: str
    evals: tuple[str, ...]
    serve: ServeConfig = field(default_factory=ServeConfig)
    resource_hint: ResourceHint = field(default_factory=ResourceHint)
    accelerator: AcceleratorChoice = DEFAULT_ACCELERATOR
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    discover_latest_checkpoint: bool = True
    limit: int | None = None
    max_tokens: int | None = None


class EvaluationReport(Artifact):
    """Durable run references; Evaldash computes the scores from their raw archives."""

    runs: tuple[EvaluationResult, ...]


def eval_step(
    model: ArtifactStep[LevanterCheckpoint], group: EvalGroup, *, version: str | None = None
) -> ArtifactStep[EvaluationResult]:
    def resolve_model(context: StepContext) -> ModelConfig:
        path = context.artifact_path(model)
        if group.discover_latest_checkpoint and not context.is_fingerprint:
            path = discover_hf_checkpoints(path)[-1]
        resource_hint = group.resource_hint
        if group.accelerator.platform is Platform.GPU:
            assert group.accelerator.gpu_type is not None
            resource_hint = replace(
                resource_hint, hbm_gb=None, gpu={group.accelerator.gpu_type: group.accelerator.gpu_count}
            )
        return ModelConfig(
            name=model.name.replace("/", "-"),
            location=path,
            identity=model.name + "@" + model.version + ":" + path,
            tokenizer=group.tokenizer,
            tokenizer_revision=group.tokenizer_revision,
            apply_chat_template=False,
            resource_hint=resource_hint,
            serve=group.serve,
            generation=GenerationConfig(max_gen_toks=group.max_tokens),
        )

    name = f"evaluation/curated/{model.name}/{group.name}"
    accelerator = group.accelerator
    step = launch_step(
        resolve_model(StepContext.for_fingerprint((), (model,))),
        ",".join(group.evals),
        version=resolve_version(name, version),
        deps=(model,),
        resolve_model=resolve_model,
        limit=group.limit,
        accelerator=accelerator.label,
        federated_cluster=accelerator.target_cluster,
    )
    return replace(step, name=name)


def eval_steps(
    model: ArtifactStep[LevanterCheckpoint], groups: Sequence[EvalGroup], *, version: str | None = None
) -> list[ArtifactStep[EvaluationResult]]:
    return [eval_step(model, group, version=version) for group in groups]


def eval_report(
    results: Sequence[ArtifactStep[EvaluationResult]], *, name: str, version: str | None = None
) -> ArtifactStep[EvaluationReport]:
    def build_config(context: StepContext) -> tuple[str, ...]:
        return tuple(context.artifact_path(result) for result in results)

    def run(paths: tuple[str, ...]) -> EvaluationReport:
        return EvaluationReport(runs=tuple(EvaluationResult.raw_load(path) for path in paths))

    name = "evaluation/report/" + name
    return ArtifactStep(
        name=name,
        version=resolve_version(name, version),
        artifact_type=EvaluationReport,
        run=run,
        build_config=build_config,
        deps=tuple(results),
    )

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn a policy's evaluations for one model into launcher specs: one server per model, several evaluations at once."""

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from evalrecords.records import EvaluationSelection
from marin.evaluation.hardware import Platform, default_platform
from marin.evaluation.model_config import ModelConfig, prompt_format
from marin.evaluation.serving_config import (
    checkpoint_digest,
    inference_build,
    native_context_window,
    resolved_serve_config,
)
from rigging.provenance import Provenance, git_output

from experiments.evaluation.curated import pinned_model, selection_dataset
from experiments.evaluation.fleet import MARIN_EVAL_HARDWARE
from experiments.evaluation.launch import EVALUATION_CONTROLLER_CLUSTER, LaunchSpec
from experiments.evaluation.policy.schema import Policy

# Evaluations each server runs at once; requests within each evaluation are concurrent already.
EVALUATIONS_PER_SERVER = 4


@dataclass(frozen=True)
class PolicyModel:
    """A model a policy launch runs, pinned and served at the policy's window or its own if smaller.

    ``declared`` models are the policy's own; others are supplied, such as a training checkpoint.
    ``checkpoint`` digests an object-store checkpoint's files, so rewriting it in place changes its run IDs.
    """

    model: ModelConfig
    declared: bool
    checkpoint: str | None

    @property
    def name(self) -> str:
        return self.model.name

    @property
    def window(self) -> int:
        window = self.model.serve.max_model_len
        assert window is not None, "policy models are served at a set window"
        return window


def policy_model(policy: Policy, model: ModelConfig, *, declared: bool) -> PolicyModel:
    """Pin a model and serve it at the policy's window, or at its own if smaller.

    The model's own window is its resolved serving config's, else the one its checkpoint declares.
    """
    model = pinned_model(model)
    serve = resolved_serve_config(model)
    own = serve.max_model_len or native_context_window(model.location, model.revision)
    if own is None:
        raise ValueError(f"{model.name}: set serve.max_model_len; its checkpoint declares no context window")
    model = replace(model, serve=replace(serve, max_model_len=min(own, policy.serving_window)))
    checkpoint = checkpoint_digest(model.location) if "://" in model.location else None
    return PolicyModel(model=model, declared=declared, checkpoint=checkpoint)


def evaluation_context(model: PolicyModel, evaluation: EvaluationSelection) -> int:
    """The context an evaluation runs with on the model's server.

    A protocol window is the one its items were built for, so a run uses exactly it, and a model served below it
    can't run the evaluation. An agent gets the smaller of its input limit and the serving window; any other
    evaluation runs at the serving window.
    """
    variant = selection_dataset(evaluation, prompt_format(model.model))
    if (protocol_window := variant.protocol.context_window) is not None:
        if protocol_window > model.window:
            raise ValueError(f"served at {model.window}, below the benchmark's {protocol_window}-token window")
        return protocol_window
    if variant.agent is not None and variant.agent.max_input_tokens is not None:
        return min(variant.agent.max_input_tokens, model.window)
    return model.window


@dataclass(frozen=True)
class ModelEvaluations:
    """The evaluations one model runs under a policy, and the ones it leaves out with the reason."""

    model: PolicyModel
    evaluations: tuple[EvaluationSelection, ...]
    excluded: Mapping[str, str]


def model_evaluations(policy: Policy, model: PolicyModel, names: Collection[str] | None = None) -> ModelEvaluations:
    """The policy's evaluations for one model, limited to ``names`` if given.

    A declared model leaves out only the policy's exclusions, and any evaluation it can't run is an error. A
    supplied model also leaves out every evaluation it can't run, such as one whose protocol window doesn't fit.
    """
    if names is not None and (unknown := set(names) - {evaluation.name for evaluation in policy.evaluations}):
        raise ValueError(f"{policy.version} has no evaluations {sorted(unknown)}")
    chosen, excluded = [], {}
    for evaluation in policy.evaluations:
        if names is not None and evaluation.name not in names:
            continue
        if model.declared and (reason := policy.exclusions.get((model.name, evaluation.name))) is not None:
            excluded[evaluation.name] = reason
            continue
        try:
            context = evaluation_context(model, evaluation)
            if (cap := evaluation.run.max_tokens) is not None and cap >= context:
                raise ValueError(f"output limit {cap} does not fit in context {context}")
        except ValueError as error:
            if model.declared:
                raise ValueError(f"{model.name} on {evaluation.name}: {error}; exclude it") from error
            excluded[evaluation.name] = str(error)
            continue
        chosen.append(evaluation)
    return ModelEvaluations(model=model, evaluations=tuple(chosen), excluded=excluded)


def _judge_key(evaluation: EvaluationSelection) -> tuple[str, ...] | None:
    return evaluation.run.judge_api_key or None


def server_groups(
    evaluations: Collection[EvaluationSelection],
) -> tuple[tuple[EvaluationSelection, ...], ...]:
    """Split a model's evaluations across servers only where their judges need different keys.

    A launch hands its judges one key; evaluations without a judge join the first server.
    """
    keyed: dict[tuple[str, ...], list[EvaluationSelection]] = {}
    for evaluation in evaluations:
        if (key := _judge_key(evaluation)) is not None:
            keyed.setdefault(key, []).append(evaluation)
    parts = list(keyed.values()) or [[]]
    parts[0] = [evaluation for evaluation in evaluations if _judge_key(evaluation) is None or evaluation in parts[0]]
    return tuple(tuple(part) for part in parts if part)


def run_key(model: PolicyModel, platform: Platform) -> str:
    """The part of a run's inputs a policy launch adds to the launcher's own: the server's build and accelerator,
    which fix the vLLM release and tensor parallelism the model's identity leaves out, and the checkpoint digest."""
    return json.dumps(
        {
            "inference": inference_build(model.model.serve, platform),
            "accelerator": MARIN_EVAL_HARDWARE.select(model.model, platform, None).label,
            "checkpoint": model.checkpoint,
        },
        sort_keys=True,
    )


def source_checkout(root: Path) -> str:
    """The checkout a launch runs from, as a run-ID part: the commit, and for uncommitted changes the tree they
    make, which identifies their tracked content."""
    commit = git_output(["rev-parse", "HEAD"], root)[:12]
    provenance = Provenance.from_git(root)
    return f"{commit}-dirty{provenance.tree_hash}" if provenance.dirty else commit


def launch_specs(
    policy: Policy,
    plan: ModelEvaluations,
    *,
    limit: int | None,
    priority_band: int,
    run_suffixes: Mapping[str, str],
    federated_cluster: str | None = None,
) -> tuple[LaunchSpec, ...]:
    """One launch per server for one model, labelled with the policy version.

    ``limit`` caps items per task for a smoke run; ``run_suffixes`` gives each evaluation's run its commit and
    attempt; ``federated_cluster`` overrides the fleet's GPU cluster. Unset seeds stay unset so each protocol
    decides them.
    """
    model = plan.model
    platform = default_platform(model.model)
    return tuple(
        LaunchSpec(
            model=model.model,
            evals=group,
            platform=platform,
            accelerator=None,
            limit=limit,
            records_prefix=None,
            submission_cluster=EVALUATION_CONTROLLER_CLUSTER,
            federated_cluster=federated_cluster,
            priority_band=priority_band,
            default_seed=None,
            max_concurrent=EVALUATIONS_PER_SERVER,
            version=policy.version,
            cache_key=run_key(model, platform),
            run_suffixes={evaluation.name: run_suffixes[evaluation.name] for evaluation in group},
            description=f"{policy.version} {'smoke' if limit is not None else 'full'} run of {model.name}",
        )
        for group in server_groups(plan.evaluations)
    )

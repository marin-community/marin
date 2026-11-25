# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn one candidate's resolved policy evaluations into launcher specs, normally one server per candidate."""

import json
from collections.abc import Collection
from dataclasses import replace

from marin.evaluation.hardware import Platform, default_platform
from marin.evaluation.model_config import ModelConfig, ServeBackend
from marin.evaluation.serving_config import resolved_serve_config
from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT, VLLM_GPU_RELEASE

from experiments.evaluation.launch import EVALUATION_CONTROLLER_CLUSTER, LaunchSpec
from experiments.evaluation.policy.resolve import ResolvedPolicyEvaluation, resolve, serving_window
from experiments.evaluation.policy.schema import EvalPolicy, PolicyCandidate

# Evaluations each server runs at once; requests within each evaluation are concurrent already.
EVALUATIONS_PER_SERVER = 4


def _judge_key(item: ResolvedPolicyEvaluation) -> tuple[str, ...] | None:
    return item.selection.run.judge_api_key or None


def server_groups(items: Collection[ResolvedPolicyEvaluation]) -> tuple[tuple[ResolvedPolicyEvaluation, ...], ...]:
    """Split a candidate's evaluations across servers only where their judges need different keys.

    A launch hands its judges one key; evaluations without a judge join the first server.
    """
    keyed: dict[tuple[str, ...], list[ResolvedPolicyEvaluation]] = {}
    for item in items:
        if (key := _judge_key(item)) is not None:
            keyed.setdefault(key, []).append(item)
    parts = list(keyed.values()) or [[]]
    parts[0] = [item for item in items if _judge_key(item) is None or item in parts[0]]
    return tuple(tuple(part) for part in parts if part)


def _inference_build(model: ModelConfig, platform: Platform) -> str:
    """The inference build a server for the model runs, which can change its outputs."""
    if model.serve.backend is not ServeBackend.VLLM:
        return model.serve.backend.value
    if platform is Platform.GPU:
        return VLLM_GPU_RELEASE.release_tag
    return f"{VLLM_FORK_REQUIREMENT}; {TPU_INFERENCE_FORK_REQUIREMENT}"


def run_key(policy: EvalPolicy, release_commit: str | None, model: ModelConfig, platform: Platform) -> str:
    """The part of every run ID a policy launch sets: the version, a release's commit, and the server's
    inference build and tensor parallelism, which the model's serving config leaves out of its identity."""
    return json.dumps(
        {
            "policy": policy.version,
            "release": release_commit,
            "inference": _inference_build(model, platform),
            "tensor_parallel": model.serve.tensor_parallel_size,
        },
        sort_keys=True,
    )


def launch_specs(
    policy: EvalPolicy,
    candidate: PolicyCandidate,
    *,
    evaluations: Collection[str] | None,
    limit: int | None,
    priority_band: int,
    federated_cluster: str | None = None,
    release_commit: str | None = None,
) -> tuple[LaunchSpec, ...]:
    """One launch per server, labelled with the policy version, normally one per candidate.

    Every server runs the candidate's serving config with its checkpoint-derived settings resolved, at the
    candidate's serving window, so a run's identity does not depend on which evaluations share its server.
    ``evaluations`` narrows the launch to those names; ``limit`` caps items per task for a smoke run;
    ``federated_cluster`` overrides the fleet's GPU cluster. Unset seeds stay unset so each protocol decides them.
    Run IDs derive from the policy version, the release's commit for a released policy, and each run's identity.
    """
    resolved = resolve(policy, candidate)
    if evaluations is not None:
        unknown = set(evaluations) - {item.evaluation.name for item in resolved}
        if unknown:
            raise ValueError(f"{candidate.name} does not run {sorted(unknown)} under {policy.version}")
        resolved = tuple(item for item in resolved if item.evaluation.name in evaluations)
    serve = resolved_serve_config(candidate.model)
    # resolve used the catalog's window; the checkpoint itself can allow less.
    window = min(serving_window(policy, candidate), serve.max_model_len or serving_window(policy, candidate))
    if too_long := [item.evaluation.name for item in resolved if item.context > window]:
        raise ValueError(f"{candidate.name}: the checkpoint serves at most {window} tokens, too few for {too_long}")
    model = replace(candidate.model, serve=replace(serve, max_model_len=window))
    platform = default_platform(model)
    return tuple(
        LaunchSpec(
            model=model,
            evals=tuple(item.selection for item in group),
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
            cache_key=run_key(policy, release_commit, model, platform),
            description=f"{policy.version} {'smoke' if limit is not None else 'full'} run of {candidate.name}",
        )
        for group in server_groups(resolved)
    )

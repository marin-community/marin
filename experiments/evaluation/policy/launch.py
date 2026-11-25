# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn one candidate's resolved policy evaluations into launcher specs, normally one server per candidate."""

from collections.abc import Collection
from dataclasses import replace

from marin.evaluation.hardware import default_platform
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.serving_config import native_context_window, resolved_serve_config

from experiments.evaluation.curated import pinned_model
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


def model_candidate(model: ModelConfig) -> PolicyCandidate:
    """Wrap a supplied model, such as a training checkpoint, as a candidate.

    Its revision is pinned, and its largest context is the window it is served at. A model that sets no
    window is served at the one its checkpoint declares, stated in its config so records name it.
    """
    model = pinned_model(model)
    window = resolved_serve_config(model).max_model_len
    if window is None:
        window = native_context_window(model.location, model.revision)
        if window is None:
            raise ValueError(f"{model.name}: set serve.max_model_len; its checkpoint declares no context window")
        model = replace(model, serve=replace(model.serve, max_model_len=window))
    return PolicyCandidate(model=model, max_context=window)


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
    ``evaluations`` narrows the launch to those names, leaving out any the candidate is excluded from;
    ``limit`` caps items per task for a smoke run; ``federated_cluster`` overrides the fleet's GPU cluster.
    Unset seeds stay unset so each protocol decides them. Run IDs derive from the policy version, the release's
    commit for a released policy, and each run's identity.
    """
    if evaluations is not None:
        if not evaluations:
            raise ValueError("no evaluations named")
        unknown = set(evaluations) - {evaluation.name for evaluation in policy.in_distribution}
        if unknown:
            raise ValueError(f"{policy.version} has no evaluations {sorted(unknown)}")
    resolved = resolve(policy, candidate, evaluations)
    serve = resolved_serve_config(candidate.model)
    # resolve used the catalog's window; the checkpoint itself can allow less.
    window = min(serving_window(policy, candidate), serve.max_model_len or serving_window(policy, candidate))
    if too_long := [item.evaluation.name for item in resolved if item.context > window]:
        raise ValueError(f"{candidate.name}: the checkpoint serves at most {window} tokens, too few for {too_long}")
    model = replace(candidate.model, serve=replace(serve, max_model_len=window))
    return tuple(
        LaunchSpec(
            model=model,
            evals=tuple(item.selection for item in group),
            platform=default_platform(model),
            accelerator=None,
            limit=limit,
            records_prefix=None,
            submission_cluster=EVALUATION_CONTROLLER_CLUSTER,
            federated_cluster=federated_cluster,
            priority_band=priority_band,
            default_seed=None,
            max_concurrent=EVALUATIONS_PER_SERVER,
            version=policy.version,
            cache_key=policy.version if release_commit is None else f"{policy.version}@{release_commit}",
            description=f"{policy.version} {'smoke' if limit is not None else 'full'} run of {candidate.name}",
        )
        for group in server_groups(resolved)
    )

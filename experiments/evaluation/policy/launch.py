# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn one candidate's resolved policy evaluations into launcher specs, one per server."""

from collections.abc import Collection
from dataclasses import dataclass, replace

from marin.evaluation.hardware import default_platform
from marin.evaluation.model_config import prompt_format
from marin.evaluation.serving_config import resolved_serve_config

from experiments.evaluation.curated import resolve_selection, runs_agent, selection_dataset
from experiments.evaluation.launch import EVALUATION_CONTROLLER_CLUSTER, LaunchSpec
from experiments.evaluation.policy.resolve import ResolvedPolicyEvaluation, benchmark_window, resolve
from experiments.evaluation.policy.schema import EvalPolicy, PolicyCandidate
from experiments.evaluation.preparation import EvaluationCorpusRecipe


@dataclass(frozen=True, kw_only=True)
class ServerGroup:
    """Evaluations that share one server, whose ``max_model_len`` is ``window``.

    Agent and single-turn evaluations never share a server: agent runs serve the model's resolved
    serving config, which is part of every run's identity. Evaluations whose judges need different keys
    never share one either, since a launch hands its judges one key.
    """

    window: int
    agentic: bool
    evaluations: tuple[ResolvedPolicyEvaluation, ...]


def evaluation_dataset(item: ResolvedPolicyEvaluation) -> EvaluationCorpusRecipe:
    """The catalog dataset, with its protocol, that the evaluation runs as for the candidate."""
    return selection_dataset(item.evaluation.selection, prompt_format(item.candidate.model))


def _server_window(item: ResolvedPolicyEvaluation, served: int | None) -> int:
    """Choose a server window for a resolved evaluation.

    A benchmark that runs at its own window requests ``served``, the candidate's window after the checkpoint
    clamps it. An agent run below its input limit, and a benchmark with no window of its own, request a
    server at the evaluation's context, which the agent's input then fills.
    """
    own = benchmark_window(item.evaluation, item.candidate)
    if own is None or item.context < own.tokens:
        window = item.context
        if own is not None and (served is None or served < window):
            raise ValueError(f"{item.candidate.name} on {item.evaluation.name}: the served window cannot hold {window}")
        return window
    if served is None or served < own.tokens:
        raise ValueError(f"{item.candidate.name} on {item.evaluation.name}: the served window cannot hold {own.tokens}")
    return served


def _judge_key(item: ResolvedPolicyEvaluation) -> tuple[str, ...] | None:
    judge = item.evaluation.judge
    return None if judge is None else judge.api_key


def _by_judge_key(members: list[ResolvedPolicyEvaluation]) -> list[list[ResolvedPolicyEvaluation]]:
    """Split evaluations so each part's judges share one key; evaluations without a judge join the first part."""
    keyed: dict[tuple[str, ...], list[ResolvedPolicyEvaluation]] = {}
    for item in members:
        if (key := _judge_key(item)) is not None:
            keyed.setdefault(key, []).append(item)
    parts = list(keyed.values()) or [[]]
    parts[0] = [item for item in members if _judge_key(item) is None or item in parts[0]]
    return parts


def server_groups(items: Collection[ResolvedPolicyEvaluation], served: int | None) -> tuple[ServerGroup, ...]:
    """Group evaluations by the server window they need, keeping agent and single-turn runs apart.

    ``served`` is the candidate's served window as the launcher resolves it.
    """
    groups: dict[tuple[int, bool], list[ResolvedPolicyEvaluation]] = {}
    for item in items:
        agentic = runs_agent(resolve_selection(item.evaluation.selection, prompt_format(item.candidate.model)))
        groups.setdefault((_server_window(item, served), agentic), []).append(item)
    return tuple(
        ServerGroup(window=window, agentic=agentic, evaluations=tuple(part))
        for (window, agentic), members in groups.items()
        for part in _by_judge_key(members)
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
    """One launch per server group, labelled with the policy version.

    ``evaluations`` narrows the launch to those names; ``limit`` caps items per task for a smoke run;
    ``federated_cluster`` overrides the fleet's GPU cluster.
    Unset seeds stay unset so each protocol decides them. Run IDs derive from the policy version, the
    release's commit for a released policy, and each run's identity.
    """
    resolved = resolve(policy, candidate)
    if evaluations is not None:
        if not evaluations:
            raise ValueError("no evaluations named")
        unknown = set(evaluations) - {item.evaluation.name for item in resolved}
        if unknown:
            raise ValueError(f"{candidate.name} does not run {sorted(unknown)} under {policy.version}")
        resolved = tuple(item for item in resolved if item.evaluation.name in evaluations)
    specs = []
    served = resolved_serve_config(candidate.model).max_model_len
    for group in server_groups(resolved, served):
        model = candidate.model
        if model.serve.max_model_len != group.window:
            model = replace(model, serve=replace(model.serve, max_model_len=group.window))
        specs.append(
            LaunchSpec(
                model=model,
                evals=tuple(item.selection for item in group.evaluations),
                platform=default_platform(model),
                accelerator=None,
                limit=limit,
                records_prefix=None,
                submission_cluster=EVALUATION_CONTROLLER_CLUSTER,
                federated_cluster=federated_cluster,
                priority_band=priority_band,
                default_seed=None,
                version=policy.version,
                cache_key=policy.version if release_commit is None else f"{policy.version}@{release_commit}",
                description=f"{policy.version} {'smoke' if limit is not None else 'full'} run of {candidate.name}",
            )
        )
    return tuple(specs)

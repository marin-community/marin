# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn one candidate's resolved policy evaluations into launcher specs, one per server."""

from collections.abc import Collection
from dataclasses import dataclass, replace

from evalrecords.records import EvaluationSelection
from marin.evaluation.hardware import default_platform
from marin.evaluation.model_config import ModelConfig, prompt_format
from marin.evaluation.serving_config import native_context_window, resolved_serve_config

from experiments.evaluation.curated import pinned_model, resolve_selection, runs_agent, selection_dataset
from experiments.evaluation.launch import EVALUATION_CONTROLLER_CLUSTER, LaunchSpec
from experiments.evaluation.policy.resolve import ResolvedPolicyEvaluation, benchmark_window, resolve
from experiments.evaluation.policy.schema import EvalPolicy, PolicyCandidate
from experiments.evaluation.preparation import EvaluationCorpusRecipe


@dataclass(frozen=True, kw_only=True)
class ServerGroup:
    """Evaluations that share one server, whose ``max_model_len`` is ``window``.

    Agent and single-turn evaluations never share a server: agent runs serve the model's resolved
    serving config, which is part of every run's identity.
    """

    window: int
    agentic: bool
    evaluations: tuple[ResolvedPolicyEvaluation, ...]


def evaluation_dataset(item: ResolvedPolicyEvaluation) -> EvaluationCorpusRecipe:
    """The catalog dataset, with its protocol, that the evaluation runs as for the candidate."""
    return selection_dataset(item.evaluation.selection, prompt_format(item.candidate.model))


def _server_window(item: ResolvedPolicyEvaluation, served: int | None) -> int:
    """Choose a server window for a resolved evaluation.

    Benchmarks that enforce their own window, which ``resolve`` made the evaluation's context, request
    ``served``, the candidate's window after the checkpoint clamps it; others request a server at the
    evaluation's context.
    """
    own = benchmark_window(item.evaluation, item.candidate)
    if own is None:
        return item.context
    if served is None or served < own:
        raise ValueError(f"{item.candidate.name} on {item.evaluation.name}: the served window cannot hold {own}")
    return served


def server_groups(items: Collection[ResolvedPolicyEvaluation], served: int | None) -> tuple[ServerGroup, ...]:
    """Group evaluations by the server window they need, keeping agent and single-turn runs apart.

    ``served`` is the candidate's served window as the launcher resolves it.
    """
    groups: dict[tuple[int, bool], list[ResolvedPolicyEvaluation]] = {}
    for item in items:
        agentic = runs_agent(resolve_selection(item.evaluation.selection, prompt_format(item.candidate.model)))
        groups.setdefault((_server_window(item, served), agentic), []).append(item)
    return tuple(
        ServerGroup(window=window, agentic=agentic, evaluations=tuple(members))
        for (window, agentic), members in groups.items()
    )


def model_candidate(model: ModelConfig) -> PolicyCandidate:
    """A model outside the policy's candidates, such as a training checkpoint, as a candidate.

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


def policy_selection(item: ResolvedPolicyEvaluation) -> EvaluationSelection:
    """The catalog selection with the policy's output limit and seed applied to its run settings."""
    selection = item.evaluation.selection
    run = selection.run.model_copy(update={"max_tokens": item.max_output, "seed": item.seed})
    return selection.model_copy(update={"run": run})


def launch_specs(
    policy: EvalPolicy,
    candidate: PolicyCandidate,
    *,
    evaluations: Collection[str] | None,
    limit: int | None,
    priority_band: int,
    federated_cluster: str | None = None,
) -> tuple[LaunchSpec, ...]:
    """One launch per server group, labelled with the policy version.

    ``evaluations`` narrows the launch to those names; ``limit`` caps items per task for a smoke run;
    ``federated_cluster`` overrides the fleet's GPU cluster.
    Unset seeds stay unset so each protocol decides them. Run IDs derive from the policy version and
    each run's identity.
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
                evals=tuple(policy_selection(item) for item in group.evaluations),
                platform=default_platform(model),
                accelerator=None,
                limit=limit,
                records_prefix=None,
                submission_cluster=EVALUATION_CONTROLLER_CLUSTER,
                federated_cluster=federated_cluster,
                priority_band=priority_band,
                default_seed=None,
                version=policy.version,
                cache_key=policy.version,
                description=f"{policy.version} {'smoke' if limit is not None else 'full'} run of {candidate.name}",
            )
        )
    return tuple(specs)

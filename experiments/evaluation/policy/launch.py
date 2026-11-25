# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn one candidate's resolved policy evaluations into launcher specs, one per server."""

from collections.abc import Collection
from dataclasses import dataclass, replace

from evalrecords.records import EvaluationSelection, PromptFormat
from marin.evaluation.hardware import default_platform
from marin.evaluation.serving_config import resolved_serve_config

from experiments.evaluation.curated import resolve_selection, runs_agent, selection_dataset
from experiments.evaluation.launch import EVALUATION_CONTROLLER_CLUSTER, LaunchSpec
from experiments.evaluation.policy.resolve import ResolvedPolicyEvaluation, resolve
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


def _model_format(item: ResolvedPolicyEvaluation) -> PromptFormat:
    return PromptFormat.CHAT if item.candidate.model.apply_chat_template else PromptFormat.COMPLETION


def evaluation_dataset(item: ResolvedPolicyEvaluation) -> EvaluationCorpusRecipe:
    """The catalog dataset, with its protocol, that the evaluation runs as for the candidate."""
    return selection_dataset(item.evaluation.selection, _model_format(item))


def benchmark_window(item: ResolvedPolicyEvaluation) -> int | None:
    """The input window the benchmark enforces on any server: its agent's input limit, else its protocol's window."""
    config = evaluation_dataset(item)
    if config.agent is not None:
        return config.agent.max_input_tokens
    return config.protocol.context_window


def _server_window(item: ResolvedPolicyEvaluation, served: int | None) -> int:
    """Choose a server window from the declared benchmark window and policy context.

    Benchmarks with a declared window request ``served``, the candidate's window after the checkpoint
    clamps it; others request a server at the policy context.
    """
    own = benchmark_window(item)
    if own is None:
        return item.context
    if own != item.context:
        raise ValueError(
            f"{item.candidate.name} on {item.evaluation.name}: the policy's context {item.context} is not the "
            f"window {own} the benchmark enforces"
        )
    if served is None or served < own:
        raise ValueError(f"{item.candidate.name} on {item.evaluation.name}: the served window cannot hold {own}")
    return served


def server_groups(items: Collection[ResolvedPolicyEvaluation], served: int | None) -> tuple[ServerGroup, ...]:
    """Group evaluations by the server window they need, keeping agent and single-turn runs apart.

    ``served`` is the candidate's served window as the launcher resolves it.
    """
    groups: dict[tuple[int, bool], list[ResolvedPolicyEvaluation]] = {}
    for item in items:
        agentic = runs_agent(resolve_selection(item.evaluation.selection, _model_format(item)))
        groups.setdefault((_server_window(item, served), agentic), []).append(item)
    return tuple(
        ServerGroup(window=window, agentic=agentic, evaluations=tuple(members))
        for (window, agentic), members in groups.items()
    )


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
    Unset seeds stay unset so each protocol decides them.
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
                description=f"{policy.version} {'smoke' if limit is not None else 'full'} run of {candidate.name}",
            )
        )
    return tuple(specs)

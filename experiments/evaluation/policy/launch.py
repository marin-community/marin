# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn one candidate's resolved policy evaluations into launcher specs, one per server."""

from collections.abc import Collection
from dataclasses import dataclass, replace

from evalrecords.records import EvaluationSelection
from marin.evaluation.hardware import default_platform
from taskcompendium.evaluation.spec import OperationKind

from experiments.evaluation.curated import selection_dataset
from experiments.evaluation.launch import EVALUATION_CONTROLLER_CLUSTER, LaunchSpec, prompt_format
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


def evaluation_dataset(item: ResolvedPolicyEvaluation) -> EvaluationCorpusRecipe:
    """The catalog dataset, with its protocol, that the evaluation runs as for the candidate."""
    return selection_dataset(item.evaluation.selection, prompt_format(item.candidate.model))


def benchmark_window(item: ResolvedPolicyEvaluation) -> int | None:
    """The input window the benchmark enforces on any server: its agent's input limit, else its protocol's window."""
    config = evaluation_dataset(item)
    if config.agent is not None:
        return config.agent.max_input_tokens
    return config.protocol.context_window


def _server_window(item: ResolvedPolicyEvaluation) -> int:
    """The window a server must have for the run to see exactly the evaluation's context.

    A benchmark that enforces its own window runs on the candidate's served window, which must hold it;
    any other runs at the server's window, so it needs a server of exactly its context.
    """
    served = item.candidate.model.serve.max_model_len
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


def server_groups(items: Collection[ResolvedPolicyEvaluation]) -> tuple[ServerGroup, ...]:
    """Group evaluations by the server window they need, keeping agent and single-turn runs apart."""
    groups: dict[tuple[int, bool], list[ResolvedPolicyEvaluation]] = {}
    for item in items:
        config = evaluation_dataset(item)
        agentic = config.protocol.operation is OperationKind.AGENT_EPISODE
        groups.setdefault((_server_window(item), agentic), []).append(item)
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
    Unset seeds stay unset so each protocol decides them. Run IDs derive from the policy version and
    each run's identity, so a relaunch finds finished runs at the same paths and reruns failed ones.
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
    for group in server_groups(resolved):
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

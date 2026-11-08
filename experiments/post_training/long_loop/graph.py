# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind stage implementations to Marin's existing artifact runner."""

from collections.abc import Callable
from dataclasses import dataclass

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run

StageBuilder = Callable[[ArtifactStep], ArtifactStep]


@dataclass(frozen=True)
class StageBindings:
    sft: StageBuilder
    sft_handoff: StageBuilder
    rl_preflight: StageBuilder
    rl: StageBuilder
    rl_handoff: StageBuilder
    eval_preflight: StageBuilder
    evaluation: StageBuilder
    report: StageBuilder


def build_loop(preflight: ArtifactStep, bindings: StageBindings) -> ArtifactStep:
    """Chain committed outputs; each builder must consume its immediate predecessor.

    Handoffs verify checkpoint payloads, preserve router biases, and return the
    model artifact consumed by the next stage. Preflights forward that identity.
    Stage completion must include releasing allocations before returning.
    """
    previous = preflight
    for builder in (
        bindings.sft,
        bindings.sft_handoff,
        bindings.rl_preflight,
        bindings.rl,
        bindings.rl_handoff,
        bindings.eval_preflight,
        bindings.evaluation,
        bindings.report,
    ):
        current = builder(previous)
        if not any(dep is previous for dep in current.deps):
            raise ValueError(f"{current.name} does not depend on {previous.name}")
        previous = current
    return previous


def execute_local_graph(output: ArtifactStep) -> list[Artifact]:
    """Run a bound graph without automatic retries of failed submissions.

    Remote bindings are deliberately not installed yet. Their submission-intent
    reconciliation and allocation-release receipts must be qualified first.
    """
    return run(output, max_concurrent=1, force_run_failed=False)

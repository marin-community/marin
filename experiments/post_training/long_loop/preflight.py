# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks on measured endpoint and learner evidence, not config declarations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointEvidence:
    expected_revision: str
    served_revision: str
    expected_tokenizer_sha256: str
    served_tokenizer_sha256: str
    rendered_prompt: str
    thinking_marker: str
    forced_stop_results: dict[int, str]


def check_endpoint(evidence: EndpointEvidence) -> None:
    if evidence.served_revision != evidence.expected_revision:
        raise ValueError("Served checkpoint differs from the predecessor checkpoint")
    if evidence.served_tokenizer_sha256 != evidence.expected_tokenizer_sha256:
        raise ValueError("Tokenizer differs from the pinned SFT tokenizer")
    if not evidence.thinking_marker or evidence.thinking_marker not in evidence.rendered_prompt:
        raise ValueError("Live template did not render the thinking marker")
    if any(evidence.forced_stop_results.get(token) != "stop" for token in (128001, 128009)):
        raise ValueError("Both EOS tokens must stop a live generation")


@dataclass(frozen=True)
class LearnerProjection:
    expected_action_positions: tuple[int, ...]
    loss_mask: tuple[int, ...]
    route_positions: tuple[int, ...]
    logprob_positions: tuple[int, ...]
    expected_behavior_revision: str
    behavior_revisions: tuple[str, ...]
    router_allocation_bytes: int
    router_budget_bytes: int


def check_projection(projection: LearnerProjection) -> None:
    """Check actual projected tensors against independently annotated action spans.

    The annotations must include earlier actions and malformed assistant turns;
    deriving them from the learner mask would make this check tautological.
    """
    expected = set(projection.expected_action_positions)
    actual = {i for i, value in enumerate(projection.loss_mask) if value}
    if not expected or not expected <= actual:
        raise ValueError("Learner dropped intended action tokens")
    if not expected <= set(projection.route_positions) or not expected <= set(projection.logprob_positions):
        raise ValueError("Action route/log-prob alignment is incomplete")
    if len(projection.behavior_revisions) != len(projection.loss_mask):
        raise ValueError("Behavior identities are not token-aligned")
    if any(projection.behavior_revisions[i] != projection.expected_behavior_revision for i in expected):
        raise ValueError("Action behavior-policy identity changed")
    if projection.router_allocation_bytes > projection.router_budget_bytes:
        raise ValueError("Router canvas exceeds the qualified memory budget")

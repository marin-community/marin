# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks on measured checkpoint and tokenizer evidence from serving."""

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

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent request identities and a guard against duplicated RFT samples."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SampleRequest:
    text: str
    prompt_index: int
    sample_index: int
    seed: int


def independent_requests(texts: list[str], prompt_indices: list[int], count: int) -> list[SampleRequest]:
    """Expand each prompt into independently seeded single-output requests.

    Global prompt indices keep seeds stable when chunks change or resume.
    vLLM 0.6.6 copies the same seed to every child of an n>1 request.
    """
    if count < 1 or len(texts) != len(prompt_indices) or len(set(prompt_indices)) != len(prompt_indices):
        raise ValueError("Expected unique prompt indices, matching texts and positive sample count")
    if any(index < 0 for index in prompt_indices):
        raise ValueError("Prompt indices must be nonnegative")
    return [
        SampleRequest(text, index, sample, 290 + count * index + sample)
        for text, index in zip(texts, prompt_indices, strict=True)
        for sample in range(count)
    ]


def require_diverse_probe(completions: list[list[str]], count: int, minimum_mean: float = 5.0) -> dict:
    """Reject a probe whose independent samples repeat the same completion."""
    if not completions or any(len(group) != count for group in completions):
        raise ValueError("Incomplete diversity probe")
    distinct = [len(set(group)) for group in completions]
    mean = sum(distinct) / len(distinct)
    if mean < minimum_mean:
        raise ValueError(f"Sampling diversity failed: mean {mean:.2f}/{count} < {minimum_mean}")
    return {"per_prompt_distinct": distinct, "mean_distinct": mean, "samples_per_prompt": count}

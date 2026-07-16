# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic next-token parity: score any marin-serve backend against the grug goldens.

The ``vllm`` and ``levanter`` backends both load the 67B ``grug_moe`` HF export through
``marin.inference.serving_backend`` and are scored identically: for each golden prompt the backend's
next-token distribution is compared to the frozen top-25 rank-independently -- the greedy token must
match exactly, and the worst single-token probability error on the golden's token set must stay
within a per-backend bound.

The invariant that is framework-independent is the **greedy match**: both backends predict the grug
reference's next token exactly, on every prompt. The probability-error bound is looser for vLLM than
for Snowball, and deliberately so -- the goldens are the levanter reference (VendoredTransformer +
sonic), Snowball is a levanter *reimplementation* of that graph (so only bf16 reduction-order noise
separates it, which also reorders the golden's exact-tied tail tokens), while vLLM is a *different
framework* serving the same weights (its kernels and per-DP-rank reductions diverge more on
higher-entropy prompts). We assert the worst single-token probability error, not the summed (L1)
error: L1 scales with the tail entropy of each prompt and with per-rank noise, so it is not a clean
fixed threshold across a diverse prompt set. L1 is still recorded for observability.

The two backends reach the distribution by different marin-serve entry points -- vLLM through
``serve()`` + HTTP (``max_tokens=1``, ``logprobs``), Snowball through ``load_model()`` + a single
forward, since Snowball has no paged decode yet so its ``serve()`` generation path is separate work.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tests.cluster.vllm.june_67b_a2b import InferenceGolden, read_inference_golden

_RESOURCES = Path(__file__).parent / "resources"


def read_golden_set() -> list[InferenceGolden]:
    """Every committed next-token golden, sorted by filename for a deterministic parametrization.

    The goldens are the grug reference's frozen top-25 next-token distributions -- one per prompt.
    Both backends are scored against this shared set.
    """
    paths = sorted(_RESOURCES.glob("*_logprobs.json"))
    assert paths, f"no goldens under {_RESOURCES}"
    return [read_inference_golden(path) for path in paths]


# Same-framework bound: Snowball is a levanter reimplementation of the levanter golden, so only bf16
# reduction-order noise separates them. Observed worst across the golden set is 0.0012.
LEVANTER_MAX_PROBABILITY_ERROR = 0.008

# Cross-framework bound: vLLM serves the same weights through a different framework, so its per-token
# probabilities diverge more from the levanter golden on higher-entropy prompts, and its DP ranks
# differ from each other. Observed worst across the golden set is 0.010 (on " Paris"); bound left at
# 0.02 for cross-run / cross-rank margin. The greedy token still matches exactly on every prompt.
VLLM_MAX_PROBABILITY_ERROR = 0.02


@dataclass(frozen=True)
class NextTokenParity:
    """One backend's next-token agreement with a golden, for one prompt on one rank/device."""

    prompt: str
    greedy_token_id: int
    golden_greedy_token_id: int
    max_probability_error: float
    top_probability_l1_error: float

    def assert_matches(self, *, max_probability_error: float) -> None:
        assert self.greedy_token_id == self.golden_greedy_token_id, self
        assert self.max_probability_error <= max_probability_error, self


def parity_from_logprob_map(
    golden: InferenceGolden, greedy_token_id: int, actual_logprobs: dict[int, float]
) -> NextTokenParity:
    """Score parity from a ``{token_id: logprob}`` map -- the vLLM HTTP ``top_logprobs`` path.

    ``actual_logprobs`` must cover every golden token (the request asks for enough logprobs that the
    golden's top-25 are always present); a missing token is a hard error, not a silent skip.
    """
    golden_logprobs = {entry.token_id: entry.logprob for entry in golden.top_logprobs}
    missing = golden_logprobs.keys() - actual_logprobs.keys()
    assert not missing, f"golden tokens missing from backend logprobs: {sorted(missing)}"
    probability_errors = [
        abs(math.exp(actual_logprobs[token_id]) - math.exp(golden_logprob))
        for token_id, golden_logprob in golden_logprobs.items()
    ]
    return NextTokenParity(
        prompt=golden.prompt,
        greedy_token_id=greedy_token_id,
        golden_greedy_token_id=golden.top_logprobs[0].token_id,
        max_probability_error=max(probability_errors),
        top_probability_l1_error=sum(probability_errors),
    )


def parity_from_logprob_row(golden: InferenceGolden, logprobs_row: np.ndarray) -> NextTokenParity:
    """Score parity from a full ``[vocab]`` log-softmax row -- the Levanter forward path."""
    golden_ids = np.asarray([entry.token_id for entry in golden.top_logprobs])
    golden_logprobs = np.asarray([entry.logprob for entry in golden.top_logprobs])
    probability_errors = np.abs(np.exp(logprobs_row[golden_ids]) - np.exp(golden_logprobs))
    return NextTokenParity(
        prompt=golden.prompt,
        greedy_token_id=int(logprobs_row.argmax()),
        golden_greedy_token_id=int(golden.top_logprobs[0].token_id),
        max_probability_error=float(probability_errors.max()),
        top_probability_l1_error=float(probability_errors.sum()),
    )

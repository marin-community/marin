# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate K completions from a proposal model, rerank by a scoring model's log-likelihood."""

import logging
import time
from dataclasses import dataclass, field

import openai

from experiments.rerank_decode.scorer import Scorer

logger = logging.getLogger(__name__)


@dataclass
class RerankStats:
    num_chunks: int = 0
    remaining_tokens: int = 0
    stop_reason: str = ""  # "eos" or "max_tokens"
    total_proposal_time: float = 0.0
    total_scoring_time: float = 0.0


def rerank(
    proposal_client: openai.Client,
    proposal_model: str,
    scorer: Scorer,
    prompt: str,
    num_samples: int,
    max_tokens: int,
    chunk_size: int,
    temperature: float,
    eos_token: str = "",
    verbose: bool = False,
) -> str | tuple[str, RerankStats]:
    """Generate num_samples completions from the proposal model, return the one
    with the highest score under the scorer.

    If the best completion was truncated, continues generating from it until
    it finishes naturally.

    Args:
        eos_token: The scoring model's EOS token string. Appended to
            completions that finish with stop so the scorer includes
            P(stop) in the score.

    Returns:
        A tuple of (generated_text, stats).
    """
    current_prompt = prompt
    generated = ""
    remaining_tokens = max_tokens
    stats = RerankStats(remaining_tokens=max_tokens)

    while remaining_tokens > 0:
        effective_chunk_size = min(chunk_size, remaining_tokens)

        t0 = time.monotonic()
        gen_resp = proposal_client.completions.create(
            model=proposal_model,
            prompt=current_prompt,
            n=num_samples,
            max_tokens=effective_chunk_size,
            temperature=temperature,
        )
        t1 = time.monotonic()

        completions = gen_resp.choices
        completion_texts = [
            c.text + eos_token if c.finish_reason == "stop" else c.text
            for c in completions
        ]

        t2 = time.monotonic()
        scores = scorer.score(current_prompt, completion_texts)
        t3 = time.monotonic()

        stats.num_chunks += 1
        stats.total_proposal_time += t1 - t0
        stats.total_scoring_time += t3 - t2

        best_idx = max(range(len(scores)), key=lambda i: scores[i])

        best_text = completions[best_idx].text
        generated += best_text

        if completions[best_idx].finish_reason == "stop":
            stats.stop_reason = "eos"
            stats.remaining_tokens = remaining_tokens
            break

        scorer.accept(current_prompt, best_text)
        remaining_tokens -= effective_chunk_size
        current_prompt = current_prompt + best_text

    if not stats.stop_reason:
        stats.stop_reason = "max_tokens"
        stats.remaining_tokens = remaining_tokens

    if verbose:
        return generated, stats
    return generated

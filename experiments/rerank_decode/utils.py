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

import openai

from experiments.rerank_decode.scorer import Scorer

logger = logging.getLogger(__name__)


def rerank(
    proposal_client: openai.Client,
    proposal_model: str,
    scorer: Scorer,
    prompt: str,
    num_samples: int,
    max_tokens: int,
    chunk_size: int,
    temperature: float,
) -> str:
    """Generate num_samples completions from the proposal model, return the one
    with the highest score under the scorer.

    If the best completion was truncated, continues generating from it until
    it finishes naturally."""
    current_prompt = prompt
    generated = ""
    remaining_tokens = max_tokens

    while remaining_tokens > 0:
        effective_chunk_size = min(chunk_size, remaining_tokens)

        gen_resp = proposal_client.completions.create(
            model=proposal_model,
            prompt=current_prompt,
            n=num_samples,
            max_tokens=effective_chunk_size,
            temperature=temperature,
        )

        completions = sorted(gen_resp.choices, key=lambda c: c.index)
        completion_texts = [c.text for c in completions]

        scores = scorer.score(current_prompt, completion_texts)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])

        generated += completions[best_idx].text

        if completions[best_idx].finish_reason == "stop":
            break

        remaining_tokens -= effective_chunk_size
        current_prompt = current_prompt + completions[best_idx].text

    return generated

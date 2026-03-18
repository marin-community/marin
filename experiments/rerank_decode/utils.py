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

logger = logging.getLogger(__name__)


def rerank(
    proposal_client: openai.Client,
    scoring_client: openai.Client,
    proposal_model: str,
    scoring_model: str,
    prompt: str,
    num_samples: int,
    max_tokens: int,
    chunk_size: int,
    temperature: float,
) -> str:
    """Generate num_samples completions from the proposal model, return the one
    with the highest log-likelihood under the scoring model.

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
        candidate_texts = [current_prompt + c.text for c in completions]

        score_resp = scoring_client.completions.create(
            model=scoring_model,
            prompt=candidate_texts,
            max_tokens=1,  # TODO: this is a hack
            echo=True,
            logprobs=1,
        )

        scored_choices = sorted(score_resp.choices, key=lambda c: c.index)

        best_idx = -1
        best_score = float("-inf")
        for i, choice in enumerate(scored_choices):
            token_logprobs = choice.logprobs.token_logprobs
            score = sum(lp for lp in token_logprobs[:-1] if lp is not None)  # see above hack
            if score > best_score:
                best_score = score
                best_idx = i

        generated += completions[best_idx].text

        if completions[best_idx].finish_reason == "stop":
            break

        remaining_tokens -= effective_chunk_size
        current_prompt = current_prompt + completions[best_idx].text

    return generated

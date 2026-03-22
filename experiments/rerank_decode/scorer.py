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

"""Scoring models for rerank decode.

A Scorer takes a prompt and a list of candidate completions and returns
a score for each candidate. Higher scores are better.
"""

import abc
import logging

import openai

logger = logging.getLogger(__name__)


class Scorer(abc.ABC):
    """Scores candidate completions given a shared prompt."""

    @abc.abstractmethod
    def score(self, prompt: str, completions: list[str]) -> list[float]:
        """Return a score for each completion. Higher is better.

        Args:
            prompt: The shared prefix / prompt text.
            completions: N candidate continuation strings.

        Returns:
            A list of N floats, one score per completion.
        """
        ...


class VLLMLogprobScorer(Scorer):
    """Scores completions by log-likelihood under a vLLM server.

    Uses the completions API with echo=True to get prompt logprobs
    for the full (prompt + completion) sequence, then sums logprobs
    over just the completion tokens.

    Note: this disables vLLM's automatic prefix caching (APC) because
    echo=True requests prompt logprobs. See DESIGN_NOTES.md for
    alternative approaches.
    """

    def __init__(self, client: openai.Client, model: str):
        self.client = client
        self.model = model

    def score(self, prompt: str, completions: list[str]) -> list[float]:
        candidate_texts = [prompt + c for c in completions]

        resp = self.client.completions.create(
            model=self.model,
            prompt=candidate_texts,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )

        scores = []
        for choice in resp.choices:
            token_logprobs = choice.logprobs.token_logprobs
            
            score = sum(lp for lp in token_logprobs if lp is not None)
            scores.append(score)

        return scores


class GuidedDecodingScorer(Scorer):
    """Scores completions by log-likelihood using guided decoding.

    Forces the scoring model to regenerate each completion exactly via
    guided_choice, then sums the output logprobs. Because this uses
    output logprobs (not prompt logprobs / echo), vLLM's automatic
    prefix caching (APC) is enabled — the shared prompt KV cache is
    computed once and reused across candidates.
    """

    def __init__(self, client: openai.Client, model: str):
        self.client = client
        self.model = model

    def score(self, prompt: str, completions: list[str]) -> list[float]:
        scores = []
        for completion in completions:
            resp = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=4096,
                logprobs=1,
                temperature=1.0,
                extra_body={"guided_choice": [completion]},
            )
            choice = resp.choices[0]
            token_logprobs = choice.logprobs.token_logprobs
            score = sum(lp for lp in token_logprobs if lp is not None)
            scores.append(score)
        return scores

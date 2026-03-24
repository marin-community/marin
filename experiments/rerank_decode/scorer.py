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
from dataclasses import dataclass

import openai
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

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

    def accept(self, prompt: str, completion: str) -> None:
        """Notify the scorer that a completion was selected.

        Stateful scorers can use this to update internal state (e.g., extend
        a KV cache). The default implementation is a no-op.

        Args:
            prompt: The current prompt.
            completion: The selected completion text.
        """


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


class KVCacheScorer(Scorer):
    """Scores completions using a HuggingFace model with incremental KV caching.

    Maintains a KV cache for the prompt prefix across calls. On each call to
    score(), a batched teacher-forced forward pass scores all suffix tokens.
    The accept() method extends the KV cache with the selected completion.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.model.eval()
        self.device = device
        self._past_key_values: DynamicCache | None = None
        self._prefix_tokens: list[int] = []

    def _prefill(self, prompt: str) -> None:
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                use_cache=True,
            )

        self._past_key_values = out.past_key_values
        self._prefix_tokens = tokens

    def _make_scoring_cache(self, bsz: int):
        """Create a trimmed, batch-expanded cache view for scoring.

        Removes the last token's KV entry (so it can be re-input to get its
        logits) and expands the batch dimension. No tensor copies.
        """
        cache = DynamicCache()
        for layer in self._past_key_values.layers:
            new_layer = DynamicLayer()
            trimmed_keys = layer.keys[:, :, :-1, :]
            trimmed_values = layer.values[:, :, :-1, :]
            new_layer.keys = trimmed_keys.expand(bsz, *trimmed_keys.shape[1:])
            new_layer.values = trimmed_values.expand(bsz, *trimmed_values.shape[1:])
            new_layer.is_initialized = True
            cache.layers.append(new_layer)
        return cache

    @torch.inference_mode()
    def score(self, prompt: str, completions: list[str]) -> list[float]:
        if self._past_key_values is None:
            self._prefill(prompt)

        bsz = len(completions)

        enc = self.tokenizer(completions, add_special_tokens=False, padding=True, return_tensors="pt").to(self.device)
        suffix_ids = enc["input_ids"]       # [bsz, max_len]
        suffix_mask = enc["attention_mask"]  # [bsz, max_len]
        max_len = suffix_ids.shape[1]

        # Prepend last prefix token so position 0 logits score the first suffix token
        last_prefix_tok = torch.full((bsz, 1), self._prefix_tokens[-1], dtype=torch.long, device=self.device)
        input_ids = torch.cat([last_prefix_tok, suffix_ids], dim=1)  # [bsz, max_len + 1]
        attention_mask = torch.cat(
            [
                torch.ones((bsz, len(self._prefix_tokens)), dtype=torch.long, device=self.device),
                suffix_mask,
            ],
            dim=1,
        )

        scoring_cache = self._make_scoring_cache(bsz)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=scoring_cache,
            use_cache=False,
        )

        scoring_logits = out.logits[:, :max_len, :]
        neg_logprobs = F.cross_entropy(
            scoring_logits.reshape(-1, scoring_logits.size(-1)),
            suffix_ids.reshape(-1),
            reduction='none',
        ).view(bsz, max_len)
        scores = -(neg_logprobs * suffix_mask).sum(dim=1)

        return scores.tolist()

    def accept(self, prompt: str, completion: str) -> None:
        if self._past_key_values is None:
            self._prefill(prompt)
            return

        accepted_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        if not accepted_ids:
            return

        input_ids = torch.tensor([accepted_ids], dtype=torch.long, device=self.device)
        prefix_len = len(self._prefix_tokens)
        attention_mask = torch.ones(
            (1, prefix_len + len(accepted_ids)),
            dtype=torch.long,
            device=self.device,
        )

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=self._past_key_values,
                use_cache=True,
            )

        self._past_key_values = out.past_key_values
        self._prefix_tokens.extend(accepted_ids)

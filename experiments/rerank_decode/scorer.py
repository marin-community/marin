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

    def reset(self) -> None:
        """Reset internal state for a new prompt.

        Stateful scorers should clear cached state (e.g., KV cache).
        The default implementation is a no-op.
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

    def __init__(self, model_name: str, device: str = "cuda", score_batch_size: int | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.model.eval()
        self.device = device
        self.score_batch_size = score_batch_size
        self._past_key_values: DynamicCache | None = None
        self._prefix_tokens: list[int] = []

    def reset(self) -> None:
        self._past_key_values = None
        self._prefix_tokens = []

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

    def _score_batch(self, completions: list[str]) -> list[float]:
        bsz = len(completions)

        enc = self.tokenizer(completions, add_special_tokens=False, padding=True, return_tensors="pt").to(self.device)
        suffix_ids = enc["input_ids"]  # [bsz, max_len]
        suffix_mask = enc["attention_mask"]  # [bsz, max_len]
        max_len = suffix_ids.shape[1]

        # Prepend last prefix token so position 0 logits score the first suffix token.
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
            reduction="none",
        ).view(bsz, max_len)
        scores = -(neg_logprobs * suffix_mask).sum(dim=1)

        return scores.tolist()

    @torch.inference_mode()
    def score(self, prompt: str, completions: list[str]) -> list[float]:
        if self._past_key_values is None:
            self._prefill(prompt)

        assert prompt == self.tokenizer.decode(self._prefix_tokens, skip_special_tokens=False), (
            f"Prompt does not match cached prefix. "
            f"Expected: {self.tokenizer.decode(self._prefix_tokens, skip_special_tokens=False)!r}, "
            f"Got: {prompt!r}"
        )

        if not completions:
            return []

        batch_size = self.score_batch_size or len(completions)
        scores: list[float] = []
        for start in range(0, len(completions), batch_size):
            scores.extend(self._score_batch(completions[start : start + batch_size]))
        return scores

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


class LevanterScorer(Scorer):
    """Scorer backed by Levanter's paged-KV ``ScoringEngine``.

    Prefills the prompt once into an anchor slot. Each ``score`` call clones
    the anchor N times (sharing the prompt's KV pages) and runs a single
    teacher-forced forward over all candidate suffixes. ``accept`` extends
    the anchor's KV with the selected completion's tokens so subsequent
    ``score`` calls benefit from the extended prefix cache.

    Requires Levanter's TrainerConfig-based TPU initialization (device mesh,
    axis mappings). Mirrors the setup used in ``experiments/rerank_decode/tpu/``.
    """

    def __init__(
        self,
        model_name: str,
        *,
        model_config,
        max_batch_size: int,
        max_prompt_len: int,
        max_completion_len: int,
        page_size: int = 32,
        max_pages: int | None = None,
    ):
        """Initialize a Levanter-backed scorer.

        Args:
            model_name: HuggingFace repo id or path for the scoring model.
            model_config: An instance of an ``LmConfig`` subclass matching the model
                family (e.g. ``LlamaConfig()`` for Llama, ``Qwen3Config()`` for Qwen).
                Used to build the HF checkpoint converter and select ``model_type``.
        """
        import jax.numpy as jnp
        import jmp

        import haliax as hax
        import levanter
        from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
        from levanter.inference.scoring import ScoringEngine, ScoringEngineConfig
        from levanter.tracker import NoopConfig
        from levanter.trainer import TrainerConfig
        from levanter.utils.tree_utils import inference_mode

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        trainer_config = TrainerConfig(
            tracker=NoopConfig(),
            mp=jmp.get_policy("c=bf16"),
            per_device_eval_parallelism=1,
        )
        levanter.initialize(trainer_config)
        self._trainer_config = trainer_config
        self._param_axis_mapping = trainer_config.parameter_axis_mapping
        self._compute_axis_mapping = trainer_config.compute_axis_mapping
        self._hax = hax

        with trainer_config.use_device_mesh(), hax.axis_mapping(self._param_axis_mapping):
            converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
            converter = converter.replaced(reference_checkpoint=RepoRef.from_string(model_name))
            model = converter.load_pretrained(
                model_config.model_type,
                ref=RepoRef.from_string(model_name),
                dtype=jnp.bfloat16,
                axis_mapping=self._param_axis_mapping,
            )
            self._model = inference_mode(model, True)

            engine_config = ScoringEngineConfig(
                max_seq_len=max_prompt_len + max_completion_len,
                max_batch_size=max_batch_size,
                max_completion_len=max_completion_len,
                page_size=page_size,
                max_pages=max_pages,
                compute_dtype=jnp.bfloat16,
            )
            self._engine = ScoringEngine.from_model_with_config(
                self._model, self.tokenizer, engine_config, axis_resources=self._compute_axis_mapping
            )
        self._cached_prompt_text: str | None = None
        self._cached_prompt_ids: list[int] | None = None

    def reset(self) -> None:
        with self._trainer_config.use_device_mesh(), self._hax.axis_mapping(self._compute_axis_mapping):
            self._engine.reset()
        self._cached_prompt_text = None
        self._cached_prompt_ids = None

    def _prompt_ids_for(self, prompt: str) -> list[int]:
        if self._cached_prompt_text == prompt and self._cached_prompt_ids is not None:
            return list(self._cached_prompt_ids)

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        self._cached_prompt_text = prompt
        self._cached_prompt_ids = list(prompt_ids)
        return prompt_ids

    def score(self, prompt: str, completions: list[str]) -> list[float]:
        prompt_ids = self._prompt_ids_for(prompt)
        completion_ids = [self.tokenizer.encode(c, add_special_tokens=False) for c in completions]
        with self._trainer_config.use_device_mesh(), self._hax.axis_mapping(self._compute_axis_mapping):
            return self._engine.score(prompt_ids, completion_ids)

    def accept(self, prompt: str, completion: str) -> None:
        prompt_ids = self._prompt_ids_for(prompt)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        with self._trainer_config.use_device_mesh(), self._hax.axis_mapping(self._compute_axis_mapping):
            self._engine.accept(prompt_ids, completion_ids)
        self._cached_prompt_text = prompt + completion
        self._cached_prompt_ids = prompt_ids + completion_ids

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

"""Mock inference context for testing training throughput without TPU/GPU inference."""

import logging
import random
import time

import numpy as np
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob

from marin.rl.environments.inference_ctx.base import BaseInferenceContext

logger = logging.getLogger(__name__)


class MockInferenceContext(BaseInferenceContext):
    """Mock inference context that generates random strings for throughput testing.

    This context doesn't require any TPU/GPU and just generates random token sequences
    to test the training pipeline throughput. Useful for benchmarking and debugging
    without needing to spin up expensive inference hardware.
    """

    def __init__(
        self,
        tokenizer,
        min_response_tokens: int = 10,
        max_response_tokens: int = 512,
        vocab_size: int | None = None,
        simulate_latency: bool = False,
        latency_per_token_ms: float = 1.0,
    ):
        """Initialize the mock inference context.

        Args:
            tokenizer: Tokenizer to use for prompt tokenization
            min_response_tokens: Minimum number of tokens to generate per response
            max_response_tokens: Maximum number of tokens to generate per response
            vocab_size: Vocabulary size for generating random token IDs (default: use tokenizer vocab_size)
            simulate_latency: Whether to simulate inference latency
            latency_per_token_ms: Simulated latency per token in milliseconds
        """
        self.tokenizer = tokenizer
        self.min_response_tokens = min_response_tokens
        self.max_response_tokens = max_response_tokens
        self.vocab_size = vocab_size or tokenizer.vocab_size
        self.simulate_latency = simulate_latency
        self.latency_per_token_ms = latency_per_token_ms

        # Metrics for throughput tracking
        self.total_tokens_generated = 0
        self.total_inference_time = 0.0
        self.total_requests = 0
        
        # Cache for token IDs by choice index (for mock round-trip)
        self._token_id_cache = {}

        logger.info(
            f"Initialized MockInferenceContext with vocab_size={self.vocab_size}, "
            f"response_tokens=[{min_response_tokens}, {max_response_tokens}]"
        )

    def get_metrics(self) -> dict:
        """Get inference metrics."""
        metrics = {
            "total_tokens_generated": self.total_tokens_generated,
            "total_inference_time_sec": self.total_inference_time,
            "total_requests": self.total_requests,
        }
        if self.total_inference_time > 0:
            metrics["avg_tokens_per_second"] = self.total_tokens_generated / self.total_inference_time
        return metrics

    def _generate_random_choice(self, index: int) -> Choice:
        """Generate a random ChatCompletion Choice.

        Args:
            index: The index of this choice

        Returns:
            OpenAI Choice object with random tokens and logprobs
        """
        # Random response length
        response_length = random.randint(self.min_response_tokens, self.max_response_tokens)

        # Generate random token IDs (avoiding special tokens at extremes)
        token_ids = [random.randint(100, self.vocab_size - 100) for _ in range(response_length)]

        # Decode tokens to text
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        # Create logprobs for each token
        logprobs_content = []
        for token_id in token_ids:
            # Decode token for display (may be gibberish for random IDs)
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            logprob_value = random.uniform(-10.0, -0.1)
            
            logprobs_content.append(
                ChatCompletionTokenLogprob(
                    token=token_str,
                    logprob=logprob_value,
                    bytes=list(token_str.encode('utf-8')) if token_str else None,
                    top_logprobs=[],
                )
            )

        # Create the Choice object with ChoiceLogprobs
        choice = Choice(
            finish_reason="length",
            index=index,
            logprobs=ChoiceLogprobs(content=logprobs_content) if logprobs_content else None,
            message=ChatCompletionMessage(
                content=text,
                role="assistant",
                function_call=None,
                tool_calls=None,
            ),
        )
        
        # Cache token IDs for this choice using its object id
        self._token_id_cache[id(choice)] = token_ids
        
        return choice

    def response_tokens_from_choice(self, choice: Choice) -> np.ndarray:
        """Extract token IDs from mock choice using cached values."""
        # Try to get from cache first
        choice_id = id(choice)
        if choice_id in self._token_id_cache:
            return np.array(self._token_id_cache[choice_id], dtype=np.int32)
        
        # Fallback: try base class method (may not work for random IDs)
        return super().response_tokens_from_choice(choice)
    
    # logprobs_from_choice is inherited from BaseInferenceContext

    def batch_completions(
        self,
        prompts: list[str],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> list[ChatCompletion]:
        """Generate mock batch completions.

        Args:
            prompts: List of prompts
            temperature: Sampling temperature (ignored)
            n: Number of completions per prompt
            max_tokens: Maximum tokens to generate (used to cap response length if provided)
            stop: Stop sequences (ignored)

        Returns:
            List of OpenAI ChatCompletion objects
        """
        start_time = time.time()

        # Use local effective max_tokens without mutating instance state
        effective_max_tokens = min(max_tokens, self.max_response_tokens) if max_tokens else self.max_response_tokens
        
        # Temporarily override for random generation
        original_max = self.max_response_tokens
        self.max_response_tokens = effective_max_tokens

        completions = []
        total_tokens = 0

        for prompt_idx, prompt in enumerate(prompts):
            # Generate n choices for this prompt
            choices = [self._generate_random_choice(i) for i in range(n)]
            
            # Count tokens for this specific completion
            completion_tokens = 0
            for choice in choices:
                if choice.logprobs and choice.logprobs.content:
                    completion_tokens += len(choice.logprobs.content)
            
            total_tokens += completion_tokens
            prompt_tokens = len(self.tokenize_prompt(prompt))

            # Create ChatCompletion object
            completion = ChatCompletion(
                id=f"chatcmpl-mock-{prompt_idx}",
                choices=choices,
                created=int(time.time()),
                model="mock-model",
                object="chat.completion",
                usage=CompletionUsage(
                    completion_tokens=completion_tokens,
                    prompt_tokens=prompt_tokens,
                    total_tokens=completion_tokens + prompt_tokens,
                ),
            )
            completions.append(completion)
        
        # Restore original max
        self.max_response_tokens = original_max

        # Simulate latency if requested
        if self.simulate_latency:
            simulated_time = total_tokens * (self.latency_per_token_ms / 1000.0)
            time.sleep(max(0, simulated_time - (time.time() - start_time)))

        inference_time = time.time() - start_time

        # Update metrics
        self.total_tokens_generated += total_tokens
        self.total_inference_time += inference_time
        self.total_requests += 1

        if inference_time > 0:
            throughput = total_tokens / inference_time
            logger.info(
                f"Mock batch inference: {len(prompts)} prompts x {n} completions, "
                f"{total_tokens} tokens in {inference_time:.2f}s, "
                f"throughput: {throughput:.1f} tokens/sec"
            )

        return completions

#!/usr/bin/env python3
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

"""Export prompts from any MarinEnv as OpenAI-compatible chat completion requests.

This tool extracts all prompts that would be submitted by an environment during
training and exports them as a JSON file of OpenAI ChatCompletionRequest objects.
This is useful for debugging inference server issues by replaying the exact
requests that would be made during training.

Usage:
    uv run src/marin/post_training/scripts/export_env_prompts.py \
        --env math \
        --output math_prompts.json \
        --n-examples 100 \
        --mode train
"""

import json
import logging

import click
import jax
import jax.random as jrandom
from levanter.inference.openai import ChatCompletionRequest, ChatMessage
from transformers import AutoTokenizer

from marin.post_training.environments.load_environments import load_environment_from_spec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--env", "env_spec", required=True, help="Environment spec (e.g., 'math' or 'math:difficulty=hard')")
@click.option("--tokenizer", "tokenizer_name", default="meta-llama/Llama-3.2-1B-Instruct", help="Tokenizer name or path")
@click.option("--output", required=True, help="Output JSON file path")
@click.option("--n-examples", type=int, default=100, help="Number of examples to export")
@click.option("--mode", type=click.Choice(["train", "eval"]), default="train", help="Dataset mode")
@click.option("--n-generations", type=int, default=1, help="Number of generations per prompt")
@click.option("--temperature", type=float, default=1.0, help="Sampling temperature")
@click.option("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
@click.option("--model-name", default="test-model", help="Model name for requests")
@click.option("--stop-tokens", type=int, multiple=True, help="Stop token IDs")
@click.option("--seed", type=int, default=42, help="Random seed")
def export_environment_prompts(
    env_spec: str,
    tokenizer_name: str,
    output: str,
    n_examples: int,
    mode: str,
    n_generations: int,
    temperature: float,
    max_tokens: int,
    model_name: str,
    stop_tokens: tuple[int, ...],
    seed: int = 42,
):
    """Export prompts from an environment as OpenAI chat completion requests."""
    logger.info(f"Exporting prompts from {env_spec} environment")

    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    logger.info(f"Loading environment: {env_spec}")
    env = load_environment_from_spec(env_spec, tokenizer)

    # Get examples from the environment
    if mode == "train":
        available_examples = env.train_examples
    else:
        available_examples = env.eval_examples

    # Sample examples
    rng = jrandom.PRNGKey(seed)
    with jax.default_device(jax.devices("cpu")[0]):
        n_to_sample = min(n_examples, len(available_examples))
        indices = jrandom.choice(rng, len(available_examples), shape=(n_to_sample,), replace=False)
        examples = [available_examples[int(idx)] for idx in indices]

    logger.info(f"Sampled {len(examples)} examples from {len(available_examples)} available")

    # Convert stop tokens to strings if provided
    stop_strings = None
    if stop_tokens:
        stop_strings = [tokenizer.decode([token]) for token in stop_tokens]

    # Create ChatCompletionRequest objects
    requests = []
    for i, example in enumerate(examples):
        prompt = example["prompt"]

        # Create chat messages
        messages = [ChatMessage(role="user", content=prompt)]

        # Create request
        request = ChatCompletionRequest(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n_generations,
            logprobs=True,  # Always enable for debugging
            stop=stop_strings,
            seed=seed + i,  # Unique seed per request
        )

        # Add metadata for debugging
        request_dict = request.model_dump(exclude_none=True)
        request_dict["_metadata"] = {
            "environment": env_spec,
            "example_index": i,
            "ground_truth_answer": example.get("answer", ""),
            "mode": mode,
        }

        requests.append(request_dict)

    # Save to JSON
    with open(output, "w") as f:
        json.dump(requests, f, indent=2)

    logger.info(f"Exported {len(requests)} chat completion requests to {output}")

    # Print summary
    total_generations = sum(req.get("n", 1) for req in requests)
    logger.info(f"Total generations that will be made: {total_generations}")


if __name__ == "__main__":
    export_environment_prompts()

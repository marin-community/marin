#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""View rollout batch pickle files in a human-readable format.

This script loads a rollout batch pickle file and displays its contents
in a nicely formatted table showing prompts, responses, tokens, logprobs,
and rewards.

Usage:
    uv run src/marin/rl/scripts/view_rollout.py <pickle_file> [--tokenizer MODEL_NAME]
"""

import argparse
import pickle
from typing import Any

from transformers import AutoTokenizer

from marin.rl.types import RolloutBatch


def clean_text(text: str) -> str:
    """Clean text for display."""
    return text.replace("\n", "\\n").replace("\t", "\\t")


def print_rollout(tokenizer: Any, group_idx: int, rollout_idx: int, rollout: Any) -> None:
    """Print a single rollout across multiple rows with aligned logprobs and rewards."""
    # Decode texts
    prompt_text = tokenizer.decode(rollout.prompt_tokens, skip_special_tokens=False)
    prompt_text = clean_text(prompt_text)

    # Decode response tokens individually for alignment
    response_tokens_text = []
    for token_id in rollout.response_tokens:
        token_text = tokenizer.decode([int(token_id)])
        token_text = clean_text(token_text)
        response_tokens_text.append(token_text)

    # Calculate column widths based on token text lengths
    col_widths = [max(len(t), 8) for t in response_tokens_text]

    # Print header
    print(
        f"\n[Group {group_idx} | "
        f"Rollout {rollout_idx}] {rollout.env_name} | "
        f"{rollout.env_example_id} | "
        f"Reward: {rollout.episode_reward:.3f}"
    )

    # Print prompt
    print(f"Prompt:   {prompt_text}")

    # Print response tokens
    print("Response: ", end="")
    for token_text, width in zip(response_tokens_text, col_widths, strict=False):
        print(f"{token_text:<{width}} ", end="")
    print()

    # Print logprobs
    print("Logprobs: ", end="")
    for logprob, width in zip(rollout.response_logprobs, col_widths, strict=False):
        print(f"{float(logprob):>{width}.3f} ", end="")
    print()

    # Print rewards
    print("Rewards:  ", end="")
    for reward, width in zip(rollout.token_rewards, col_widths, strict=False):
        print(f"{float(reward):>{width}.3f} ", end="")
    print()


def print_rollout_batch(tokenizer: Any, batch: RolloutBatch) -> None:
    """Print the entire rollout batch."""
    print("=" * 120)
    print(
        f"ROLLOUT BATCH - Worker: {batch.metadata.worker_id} "
        f"| Step: {batch.metadata.weight_step} | Groups: {len(batch.groups)}"
    )
    print("=" * 120)

    for group_idx, group in enumerate(batch.groups):
        for rollout_idx, rollout in enumerate(group.rollouts):
            print_rollout(tokenizer, group_idx, rollout_idx, rollout)

    print("\n" + "=" * 120)


def load_rollout_batch(pickle_file: str) -> RolloutBatch:
    """Load rollout batch from pickle file."""
    with open(pickle_file, "rb") as f:
        batch = pickle.load(f)
    return batch


def main():
    parser = argparse.ArgumentParser(description="View rollout batch pickle files in human-readable format")
    parser.add_argument("pickle_file", help="Path to the rollout batch pickle file")
    parser.add_argument(
        "--tokenizer",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Tokenizer model name or path (default: meta-llama/Llama-3.2-1B-Instruct)",
    )

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load rollout batch
    print(f"Loading rollout batch: {args.pickle_file}")
    batch = load_rollout_batch(args.pickle_file)

    # Display the batch
    print_rollout_batch(tokenizer, batch)


if __name__ == "__main__":
    main()

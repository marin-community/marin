# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Custom processing functions for vLLM environment results."""

import logging
from dataclasses import dataclass
from typing import Any

from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)


@dataclass
class ProcessedVLLMOutputs:
    """Processed outputs from vLLM environment."""

    prompt_ids: list[list[int]]
    """Tokenized prompts, one per example."""

    completion_ids: list[list[int]]
    """Tokenized completions, one per generation."""

    completion_logprobs: list[list[float]]
    """Log probabilities for completion tokens, one per generation."""

    rewards: list[float]
    """Reward for each generation."""


def parse_chat_completion_tokens_from_bytes(chat_completion: ChatCompletion, tokenizer: Any) -> list[int]:
    """
    Parse token IDs from chat completion.

    vLLM returns tokens as their string representations (from convert_ids_to_tokens()).
    We need to convert them back via the vocabulary to get the correct token IDs.

    Args:
        chat_completion: ChatCompletion object from vLLM
        tokenizer: Tokenizer (MarinTokenizer) to use for converting token strings to IDs

    Returns:
        List of token IDs
    """
    assert len(chat_completion.choices) == 1, f"Expected 1 choice, got {len(chat_completion.choices)}: {chat_completion}"
    assert chat_completion.choices[0].logprobs is not None, f"Logprobs should not be None: {chat_completion}"
    assert (
        chat_completion.choices[0].logprobs.content is not None
    ), f"Logprob content should not be None: {chat_completion}"

    vocab = tokenizer.get_vocab()
    tokens = []
    logprob_content = chat_completion.choices[0].logprobs.content

    for token_logprob in logprob_content:
        token_str = token_logprob.token

        # Case 1: Token is in format "token_id:<int>" (when return_tokens_as_token_ids=True works)
        if token_str.startswith("token_id:"):
            try:
                token_id = int(token_str.split(":", 1)[1])
                tokens.append(token_id)
                continue
            except (ValueError, IndexError):
                pass

        # Case 2: Token is a string representation (the standard case with vLLM)
        # Look up in vocab for correct BPE round-trip
        token_id = vocab.get(token_str)
        if token_id is None:
            logger.warning(f"Token '{token_str}' not found in vocabulary, may indicate tokenizer mismatch")
            tokens.append(0)
        else:
            tokens.append(token_id)

    return tokens


def parse_chat_completion_logprobs(chat_completion: ChatCompletion) -> list[float]:
    """
    Parse log probabilities from chat completion.

    Args:
        chat_completion: ChatCompletion object from vLLM

    Returns:
        List of log probabilities
    """
    assert len(chat_completion.choices) == 1, "Response should always have one choice"
    assert chat_completion.choices[0].logprobs is not None, "Logprobs should not be None"
    assert chat_completion.choices[0].logprobs.content is not None, "Logprob content should not be None"

    logprobs = [logprob.logprob for logprob in chat_completion.choices[0].logprobs.content]
    return logprobs


def process_vllm_chat_results(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    states: list[dict[str, Any]],
    rewards: list[float],
    tokenizer: Any,
) -> ProcessedVLLMOutputs:
    """
    Process vLLM results for chat format conversations.

    Args:
        prompts: List of chat message prompts
        completions: List of chat message completions
        states: List of state dicts containing responses
        rewards: List of rewards
        tokenizer: Tokenizer to use for encoding

    Returns:
        ProcessedVLLMOutputs with tokenized data and logprobs
    """
    all_prompt_ids = []
    all_completion_ids = []
    all_completion_logprobs = []
    all_rewards = []

    for idx, (prompt, completion, state, reward) in enumerate(zip(prompts, completions, states, rewards, strict=False)):
        try:
            # Tokenize the prompt using chat template
            prompt_ids = tokenizer.apply_chat_template(
                conversation=prompt,
                add_generation_prompt=True,
            )

            # Extract responses from state
            responses = state.get("responses", [])

            # Process completion messages and extract tokens/logprobs from responses
            completion_ids = []
            completion_logprobs = []

            response_idx = 0
            for msg_idx, message in enumerate(completion):
                # This is a model-generated response
                if response_idx < len(responses):
                    response = responses[response_idx]

                    # Parse tokens and logprobs from the response
                    tokens = parse_chat_completion_tokens_from_bytes(response, tokenizer)
                    logprobs = parse_chat_completion_logprobs(response)

                    completion_ids.extend(tokens)
                    completion_logprobs.extend(logprobs)
                    logger.debug(f"Example {idx}, message {msg_idx}: Parsed {len(tokens)} tokens")

                    response_idx += 1
                else:
                    # No response available, tokenize the message content
                    logger.warning(
                        f"Example {idx}, message {msg_idx}: No response available, using fallback tokenization"
                    )
                    content = message.get("content", "")
                    tokens = tokenizer.encode(content, add_special_tokens=False)
                    completion_ids.extend(tokens)
                    completion_logprobs.extend([0.0] * len(tokens))

            all_prompt_ids.append(prompt_ids)
            all_completion_ids.append(completion_ids)
            all_completion_logprobs.append(completion_logprobs)
            all_rewards.append(reward)

        except Exception as e:
            logger.error(f"Example {idx}: Failed to process: {e}", exc_info=True)
            # Skip this example
            continue

    logger.info(f"Processed {len(all_prompt_ids)} examples successfully out of {len(prompts)} total")

    return ProcessedVLLMOutputs(
        prompt_ids=all_prompt_ids,
        completion_ids=all_completion_ids,
        completion_logprobs=all_completion_logprobs,
        rewards=all_rewards,
    )

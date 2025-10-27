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

"""Custom processing functions for vLLM environment results."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

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


def parse_chat_completion_tokens_from_bytes(chat_completion: Any, tokenizer: Any) -> list[int]:
    """
    Parse token IDs from chat completion.
    
    vLLM returns tokens as their string representations (from convert_ids_to_tokens()).
    We need to convert them back using convert_tokens_to_ids() to get the correct token IDs.
    
    Args:
        chat_completion: ChatCompletion object from vLLM
        tokenizer: Tokenizer to use for converting token strings to IDs
        
    Returns:
        List of token IDs
    """
    if len(chat_completion.choices) != 1:
        raise ValueError("Response should always have one choice")
    
    if chat_completion.choices[0].logprobs is None:
        raise ValueError("Logprobs should not be None")
    
    if chat_completion.choices[0].logprobs.content is None:
        raise ValueError("Logprob content should not be None")
    
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
        # Use convert_tokens_to_ids for correct BPE round-trip
        # The server uses convert_ids_to_tokens which preserves BPE format (e.g., Ġ for spaces)
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            # Check if we got the unknown token ID
            if token_id == tokenizer.unk_token_id:
                logger.warning(f"Token '{token_str}' converted to unk_token_id, may indicate tokenizer mismatch")
            tokens.append(token_id)
        except Exception as e:
            logger.warning(f"Failed to convert token '{token_str}' to ID: {e}")
            # Use unk_token_id as fallback
            tokens.append(tokenizer.unk_token_id or 0)
    
    return tokens


def parse_chat_completion_logprobs(chat_completion: Any) -> list[float]:
    """
    Parse log probabilities from chat completion.
    
    Args:
        chat_completion: ChatCompletion object from vLLM
        
    Returns:
        List of log probabilities
    """
    if len(chat_completion.choices) != 1:
        raise ValueError("Response should always have one choice")
    
    if chat_completion.choices[0].logprobs is None:
        raise ValueError("Logprobs should not be None")
    
    if chat_completion.choices[0].logprobs.content is None:
        raise ValueError("Logprob content should not be None")
    
    logprobs = [
        logprob.logprob for logprob in chat_completion.choices[0].logprobs.content
    ]
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
    
    for idx, (prompt, completion, state, reward) in enumerate(zip(prompts, completions, states, rewards)):
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
                if message["role"] == "assistant":
                    # This is a model-generated response
                    if response_idx < len(responses):
                        response = responses[response_idx]
                        
                        # Parse tokens and logprobs from the response
                        try:
                            tokens = parse_chat_completion_tokens_from_bytes(response, tokenizer)
                            logprobs = parse_chat_completion_logprobs(response)
                            
                            if len(tokens) != len(logprobs):
                                logger.warning(
                                    f"Example {idx}, message {msg_idx}: Token count ({len(tokens)}) "
                                    f"doesn't match logprob count ({len(logprobs)})"
                                )
                                # Truncate to shorter length
                                min_len = min(len(tokens), len(logprobs))
                                tokens = tokens[:min_len]
                                logprobs = logprobs[:min_len]
                            
                            completion_ids.extend(tokens)
                            completion_logprobs.extend(logprobs)
                            logger.debug(
                                f"Example {idx}, message {msg_idx}: Parsed {len(tokens)} tokens"
                            )
                        except Exception as e:
                            # If parsing fails, fall back to tokenizing the message content
                            logger.warning(
                                f"Example {idx}, message {msg_idx}: Failed to parse response tokens: {e}"
                            )
                            content = message.get("content", "")
                            fallback_tokens = tokenizer.encode(content, add_special_tokens=False)
                            completion_ids.extend(fallback_tokens)
                            # Use zero logprobs as fallback
                            completion_logprobs.extend([0.0] * len(fallback_tokens))
                        
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
                else:
                    # Non-assistant messages (user/tool) - these shouldn't typically appear in completion
                    # but if they do, we can handle them
                    logger.debug(
                        f"Example {idx}, message {msg_idx}: Skipping non-assistant message with role {message['role']}"
                    )
            
            all_prompt_ids.append(prompt_ids)
            all_completion_ids.append(completion_ids)
            all_completion_logprobs.append(completion_logprobs)
            all_rewards.append(reward)
        
        except Exception as e:
            logger.error(f"Example {idx}: Failed to process: {e}", exc_info=True)
            # Skip this example
            continue
    
    logger.info(
        f"Processed {len(all_prompt_ids)} examples successfully out of {len(prompts)} total"
    )
    
    return ProcessedVLLMOutputs(
        prompt_ids=all_prompt_ids,
        completion_ids=all_completion_ids,
        completion_logprobs=all_completion_logprobs,
        rewards=all_rewards,
    )


def extract_completion_text_from_state(state: dict[str, Any]) -> str:
    """
    Extract the completion text from a state dict.
    
    Args:
        state: State dict containing responses
        
    Returns:
        Concatenated completion text
    """
    responses = state.get("responses", [])
    texts = []
    
    for response in responses:
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                texts.append(content)
    
    return "".join(texts)


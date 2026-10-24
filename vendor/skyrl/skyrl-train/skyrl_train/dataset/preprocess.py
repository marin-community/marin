from typing import List, Tuple, Optional, Sequence
import numpy as np
import torch
from transformers import AutoTokenizer
from jaxtyping import Float, Integer


def _verify_inputs(
    prompts: List[List[int]],
    responses: List[List[int]],
    rewards: Optional[List[torch.Tensor]],
    loss_masks: List[List[int]],
):
    assert len(prompts) == len(responses) and len(prompts) > 0, (
        "prompts and responses must have the same length and length must be greater than 0, got {} and {}".format(
            len(prompts), len(responses)
        )
    )

    if rewards is not None:
        assert len(rewards) == len(prompts), "rewards must have the same length as prompts, got {} and {}".format(
            len(rewards), len(prompts)
        )
    assert len(loss_masks) == len(prompts), "loss_masks must have the same length as prompt, got {} and {}".format(
        len(loss_masks), len(prompts)
    )

    # Element-type validation. torch.tensor(sequences) raises the cryptic
    # `ValueError: too many dimensions 'str'` if any prompt/response token-id
    # list contains a non-int (e.g. a stringified token leaking from a
    # malformed rollout trajectory). Surface exactly which sample + field +
    # offending element is corrupt instead, so the bad trajectory is
    # actionable rather than a bare ValueError at the tensor build. Valid
    # int-only inputs (the normal path, incl. a3) pass through unchanged.
    def _first_bad_token(seq):
        for tok in seq:
            if not isinstance(tok, (int, bool)):
                return tok
        return None

    for field_name, seqs in (("prompt", prompts), ("response", responses)):
        for idx, seq in enumerate(seqs):
            bad = _first_bad_token(seq)
            if bad is not None:
                raise ValueError(
                    "{field} token-id list at sample index {idx} contains a non-int element "
                    "{bad!r} (type {tname}); expected a flat list of token ids. This corrupts "
                    "torch.tensor() collation (the bare 'too many dimensions \\'str\\'' error). "
                    "The offending trajectory's {field}_ids must be tokenized ints.".format(
                        field=field_name, idx=idx, bad=bad, tname=type(bad).__name__
                    )
                )


def collate_response_token_channel(
    rows: Optional[Sequence[Sequence[float]]],
    response_template: torch.Tensor,
    *,
    dtype: torch.dtype,
    expected_lengths: Sequence[int],
) -> Optional[torch.Tensor]:
    """Validate and right-pad a scalar channel aligned one-to-one with response tokens."""
    if rows is None:
        return None
    if len(rows) != len(expected_lengths):
        raise ValueError("response token channel must have one row per response")
    if any(len(row) != expected for row, expected in zip(rows, expected_lengths)):
        raise ValueError("response token channel must align one-to-one with response tokens")
    result = torch.zeros_like(response_template, dtype=dtype)
    for index, row in enumerate(rows):
        values = torch.as_tensor(row, dtype=dtype)
        result[index, : len(values)] = values
    return result


def convert_prompts_responses_to_batch_tensors(
    tokenizer: AutoTokenizer,
    prompts: List[List[int]],
    responses: List[List[int]],
    rewards: List[List[float]],
    loss_masks: List[List[int]],
    logprobs: Optional[List[np.ndarray]] = None,
    token_level_shaping: Optional[List[List[float]]] = None,
    response_span_tags: Optional[List[List[int]]] = None,
    *,
    max_prompt_len: int | None = None,
    max_response_len: int | None = None,
) -> Tuple[
    Float[torch.Tensor, "batch seq_len"],
    Float[torch.Tensor, "batch seq_len"],
    Float[torch.Tensor, "batch response_len"],
    Float[torch.Tensor, "batch response_len"],
    Float[torch.Tensor, "batch response_len"],
    Optional[Float[torch.Tensor, "batch response_len"]],
    Optional[Float[torch.Tensor, "batch response_len"]],
    Optional[Integer[torch.Tensor, "batch response_len"]],
]:
    """
    Convert prompts and responses to batch tensors for training.

    This function concatenates all prompts and responses to the following format:

    | [PAD] [PAD] token token token | token token [PAD] [PAD] |
    | token token token token token | token token [PAD] [PAD] |
    | [PAD] [PAD] [PAD] token token | token token token [PAD] |
    |<---------- prompt ----------->|<-------- answer ------->|

    Assumes that the responses already contain an eos token at index -1.

    Args:
        tokenizer: Model tokenizer
        prompts: List of tokenized prompts
        responses: List of tokenized responses
        rewards: List of rewards for each response
        loss_masks: List of loss masks for each response
        logprobs: List of rollout log probs for each response
        max_prompt_len: Optional whole-batch prompt width, at least the longest selected prompt
        max_response_len: Optional whole-batch response width, at least the longest selected response

    Returns:
        sequences: Full trajectories (padded and concatenated prompts and responses). Size: (batch, seq_len).
        attention_mask: Attention mask for the model. Size: (batch, seq_len)
        action_mask: Response mask for the model. Size: (batch, response_len)
        rewards: Rewards for each output. Size: (batch, response_len)
        loss_masks: Loss masks for each output. Size: (batch, response_len)
    """
    _verify_inputs(prompts, responses, rewards, loss_masks)

    max_input_len, max_output_len = 0, 0
    prompt_token_lens, response_token_lens = [], []
    for prompt, response in zip(prompts, responses):
        prompt_token_len = len(prompt)
        response_token_len = len(response)
        prompt_token_lens.append(prompt_token_len)
        response_token_lens.append(response_token_len)

        max_input_len = max(max_input_len, prompt_token_len)
        max_output_len = max(max_output_len, response_token_len)

    if max_prompt_len is not None:
        if max_prompt_len < max_input_len:
            raise ValueError("global prompt width must fit every selected prompt")
        max_input_len = max_prompt_len
    if max_response_len is not None:
        if max_response_len < max_output_len:
            raise ValueError("global response width must fit every selected response")
        max_output_len = max_response_len

    # Copy each row's tokens into preallocated tensors. Building the padded batch from nested Python lists would
    # walk every padding element under the GIL, and padding dominates a batch with a long response window.
    batch_size = len(prompts)
    sequences = torch.full((batch_size, max_input_len + max_output_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_input_len + max_output_len), dtype=torch.int64)
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        # Left-pad the prompt and right-pad the response.
        prompt_start = max_input_len - prompt_token_lens[i]
        response_end = max_input_len + response_token_lens[i]
        sequences[i, prompt_start:max_input_len] = torch.as_tensor(prompt, dtype=torch.long)
        sequences[i, max_input_len:response_end] = torch.as_tensor(response, dtype=torch.long)
        attention_mask[i, prompt_start:response_end] = 1
    action_mask = attention_mask[:, max_input_len:].clone()

    # initialize ret loss masks to be the same as action mask
    ret_loss_masks = torch.zeros_like(action_mask, dtype=torch.float)
    for i, loss_mask in enumerate(loss_masks):
        ret_loss_masks[i, : len(loss_mask)] = torch.tensor(loss_mask)

    # do the same for custom rewards
    ret_rewards = torch.zeros_like(action_mask, dtype=torch.float)
    for i, custom_reward in enumerate(rewards):
        if isinstance(custom_reward, list):
            custom_reward = torch.tensor(custom_reward)
        ret_rewards[i, : len(custom_reward)] = custom_reward

    logprobs_tensor = None
    if logprobs:
        logprobs_tensor = torch.zeros_like(action_mask, dtype=torch.float)
        for i, sample_logprobs in enumerate(logprobs):
            logprobs_tensor[i, : len(sample_logprobs)] = torch.as_tensor(sample_logprobs, dtype=torch.float)

    # Loop-behavior reward shaping (Stage B / F5 + F4): right-pad the per-token
    # shaping channel and span tags on the response axis exactly like rewards /
    # loss_mask. Both are gated upstream (only passed when
    # enable_token_reward_channel is on), so when off they stay None and the
    # returned tuple's last two slots are None — the caller attaches the batch keys
    # only when non-None, keeping the flag-off batch byte-identical.
    token_level_shaping_tensor = collate_response_token_channel(
        token_level_shaping,
        action_mask,
        dtype=torch.float,
        expected_lengths=response_token_lens,
    )
    response_span_tags_tensor = collate_response_token_channel(
        response_span_tags,
        action_mask,
        dtype=torch.long,
        expected_lengths=response_token_lens,
    )

    return (
        sequences,
        attention_mask,
        action_mask,
        ret_rewards,
        ret_loss_masks,
        logprobs_tensor,
        token_level_shaping_tensor,
        response_span_tags_tensor,
    )

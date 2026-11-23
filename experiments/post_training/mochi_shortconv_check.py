# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise four-proposal ShortConv rejection states before the serving gate.

Adapted from marin-community/vllm c83c9865 tests/models/test_grugmoe.py;
extends the sequential oracle through all five verifier rows.
"""

import itertools
from dataclasses import replace
from types import SimpleNamespace

import torch
from vllm.config import VllmConfig, get_current_vllm_config, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.models.grugmoe import GrugMoeShortConv
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata


def check_rejection(accepted, dtype, query_lengths):
    config = get_current_vllm_config()
    plain = GrugMoeShortConv(8, 4, dtype, None, "hero.sconv.plain").cuda()
    previous_speculative_config = config.speculative_config
    config.speculative_config = SimpleNamespace(num_speculative_tokens=4)
    try:
        speculative = GrugMoeShortConv(8, 4, dtype, None, "hero.sconv.speculative").cuda()
    finally:
        config.speculative_config = previous_speculative_config

    weight = torch.arange(32, device="cuda", dtype=dtype).reshape(4, 8) / 32
    history = torch.arange(48, device="cuda", dtype=dtype).reshape(2, 3, 8) / 16
    for module in (plain, speculative):
        module.weight.data.copy_(weight)
        module.kv_cache = (torch.zeros(3, *module.get_state_shape()[0], device="cuda", dtype=dtype),)
        state = module.kv_cache[0]
        if is_conv_state_dim_first():
            state = state.transpose(-1, -2)
        state[1:, :3].copy_(history)

    metadata = ShortConvAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=2,
        num_decode_tokens=2,
        num_reqs=2,
        has_initial_states_p=None,
        query_start_loc_p=None,
        num_computed_tokens_p=None,
        state_indices_tensor_p=None,
        state_indices_tensor_d=torch.tensor([1, 2], device="cuda", dtype=torch.int32),
        query_start_loc_d=None,
        num_accepted_tokens=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        seq_lens=torch.tensor([3, 3], device="cuda", dtype=torch.int32),
    )
    proposals = torch.arange(80, device="cuda", dtype=dtype).reshape(2, 5, 8) / 8
    spec_metadata = replace(
        metadata,
        num_decode_tokens=sum(query_lengths),
        query_start_loc_d=torch.tensor([0, query_lengths[0], sum(query_lengths)], device="cuda", dtype=torch.int32),
        state_indices_tensor_d=torch.tensor([[1, 0, 0, 0, 0], [2, 0, 0, 0, 0]], device="cuda", dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, device="cuda", dtype=torch.int32),
    )
    verification_inputs = torch.cat([proposals[request, :length] for request, length in enumerate(query_lengths)])
    with set_forward_context({speculative.prefix: spec_metadata}, config, num_tokens=sum(query_lengths)):
        verified = speculative(verification_inputs).split(query_lengths)
    accepted_counts = [min(accepted, length) for length in query_lengths]
    # A bonus token is included in the accepted count. Only this prefix is
    # committed; all later proposals must disappear from the next state.
    for request, count in enumerate(accepted_counts):
        request_metadata = replace(
            metadata,
            num_decodes=1,
            num_decode_tokens=1,
            num_reqs=1,
            state_indices_tensor_d=metadata.state_indices_tensor_d[request : request + 1],
            seq_lens=metadata.seq_lens[request : request + 1],
        )
        for index in range(count):
            with set_forward_context({plain.prefix: request_metadata}, config, num_tokens=1):
                expected = plain(proposals[request, index : index + 1])
            torch.testing.assert_close(verified[request][index], expected[0])

    next_tokens = torch.full((2, 5, 8), -0.5, device="cuda", dtype=dtype)
    spec_metadata = replace(
        spec_metadata,
        num_decode_tokens=10,
        query_start_loc_d=torch.tensor([0, 5, 10], device="cuda", dtype=torch.int32),
        num_accepted_tokens=torch.tensor(accepted_counts, device="cuda", dtype=torch.int32),
    )
    with set_forward_context({speculative.prefix: spec_metadata}, config, num_tokens=10):
        after_rejection = speculative(next_tokens.flatten(0, 1)).reshape(2, 5, 8)
    for index in range(5):
        with set_forward_context({plain.prefix: metadata}, config, num_tokens=2):
            expected = plain(next_tokens[:, index].contiguous())
        torch.testing.assert_close(after_rejection[:, index], expected)


def run() -> dict:
    cases = 0
    for accepted, dtype, lengths in itertools.product(
        range(1, 6), (torch.float32, torch.bfloat16), ((5, 5), (5, 2), (1, 5))
    ):
        with set_current_vllm_config(VllmConfig()):
            check_rejection(accepted, dtype, lengths)
        cases += 1
    return {"passed": True, "cases": cases, "proposals": 4, "oracle": "sequential non-speculative ShortConv"}

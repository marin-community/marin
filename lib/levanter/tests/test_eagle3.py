# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from typing import Any, NamedTuple

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import save_file

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams, FinishReason
from levanter.inference.eagle3 import propose_eagle3, reconcile_eagle3
from levanter.inference.speculative import verify_snowball_proposals
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from haliax import Axis
from levanter.models.eagle3 import Eagle3Config, Eagle3Draft
from levanter.testing.helpers import skip_if_no_torch


class _Checkpoint(NamedTuple):
    config: dict[str, Any]
    state: dict[str, np.ndarray]
    target_embedding: np.ndarray


def _checkpoint():
    # Keys and normalization flags mirror the pinned embedding-free HF checkpoint.
    config = {
        "architectures": ["Eagle3DraftModel"],
        "speculators_model_type": "eagle3",
        "norm_before_fc": True,
        "norm_before_residual": True,
        "norm_output": True,
        "fc_norm": False,
        "tie_word_embeddings": False,
        "target_hidden_size": None,
        "draft_vocab_size": 8,
        "eagle_aux_hidden_state_layer_ids": [0, 1, 2],
        "transformer_layer_config": {
            "model_type": "llama",
            "num_hidden_layers": 1,
            "hidden_act": "silu",
            "attention_bias": False,
            "mlp_bias": False,
            "layer_types": ["sliding_attention"],
            "use_sliding_window": True,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "vocab_size": 32,
            "sliding_window": 3,
            "max_position_embeddings": 16,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        },
    }
    rng = np.random.default_rng(72)
    targets = np.array([1, 3, 6, 8, 12, 19, 23, 31], dtype=np.int64)
    state = {"d2t": targets - np.arange(8), "t2d": np.isin(np.arange(32), targets)}
    shapes = {
        "fc.weight": (16, 48),
        "input_norm.weight": (48,),
        "layers.0.hidden_norm.weight": (16,),
        "layers.0.input_layernorm.weight": (16,),
        "layers.0.post_attention_layernorm.weight": (16,),
        "norm.weight": (16,),
        "layers.0.self_attn.q_proj.weight": (16, 32),
        "layers.0.self_attn.k_proj.weight": (8, 32),
        "layers.0.self_attn.v_proj.weight": (8, 32),
        "layers.0.self_attn.o_proj.weight": (16, 16),
        "layers.0.mlp.gate_proj.weight": (24, 16),
        "layers.0.mlp.up_proj.weight": (24, 16),
        "layers.0.mlp.down_proj.weight": (16, 24),
        "lm_head.weight": (8, 16),
    }
    for name, shape in shapes.items():
        state[name] = (rng.normal(size=shape) * 0.2 + (1 if len(shape) == 1 else 0)).astype(np.float32)
    embedding = rng.normal(size=(32, 16)).astype(np.float32)
    return _Checkpoint(config, state, embedding)


def _torch_forward(state, embedding, token_ids, hidden, *, project):
    import torch  # noqa: PLC0415  # optional test dependency
    import torch.nn.functional as functional  # noqa: PLC0415

    weights = {k: torch.from_numpy(np.asarray(v)) for k, v in state.items()}
    hidden = torch.from_numpy(np.asarray(hidden).copy())

    def norm(x, name):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * weights[name + ".weight"]

    def linear(x, name):
        return functional.linear(x, weights[name + ".weight"])

    if project:
        hidden = linear(norm(hidden, "input_norm"), "fc")
    residual = norm(hidden, "layers.0.hidden_norm")
    embeds = torch.from_numpy(embedding)[torch.from_numpy(np.asarray(token_ids).copy())]
    x = torch.cat([norm(embeds, "layers.0.input_layernorm"), residual], dim=-1)
    q = linear(x, "layers.0.self_attn.q_proj").reshape(*x.shape[:2], 4, 4)
    k = linear(x, "layers.0.self_attn.k_proj").reshape(*x.shape[:2], 2, 4)
    v = linear(x, "layers.0.self_attn.v_proj").reshape(*x.shape[:2], 2, 4)
    length = x.shape[1]
    angles = torch.arange(length)[:, None] * 10000.0 ** (-torch.arange(0, 4, 2) / 4)
    cosine, sine = angles.cos()[None, :, None], angles.sin()[None, :, None]

    def rotate(value):
        first, second = value.chunk(2, dim=-1)
        return torch.cat([first * cosine - second * sine, second * cosine + first * sine], dim=-1)

    q = rotate(q).transpose(1, 2)
    k = rotate(k).repeat_interleave(2, dim=2).transpose(1, 2)
    v = v.repeat_interleave(2, dim=2).transpose(1, 2)
    positions = torch.arange(length)
    mask = (positions[:, None] >= positions) & (positions[:, None] - positions < 3)
    scores = (q @ k.transpose(-1, -2)) / 2
    probabilities = scores.masked_fill(~mask, -torch.inf).softmax(-1)
    attended = (probabilities @ v).transpose(1, 2).reshape(*x.shape[:2], 16)
    x = residual + linear(attended, "layers.0.self_attn.o_proj")
    mlp_in = norm(x, "layers.0.post_attention_layernorm")
    x = x + linear(
        functional.silu(linear(mlp_in, "layers.0.mlp.gate_proj")) * linear(mlp_in, "layers.0.mlp.up_proj"),
        "layers.0.mlp.down_proj",
    )
    hidden = norm(x, "norm")
    return linear(hidden, "lm_head").numpy(), hidden.numpy()


class _PackedInputs(NamedTuple):
    tokens: hax.NamedArray
    batch_info: PageBatchInfo
    positions: hax.NamedArray


def _decode_inputs(token_ids: list[int], position: int):
    capacity = 8 * jax.device_count()
    tokens = np.zeros(capacity, dtype=np.int32)
    count = len(token_ids)
    tokens[:count] = token_ids
    positions = np.zeros(capacity, dtype=np.int32)
    positions[:count] = np.arange(position, position + count)
    destinations = np.full(capacity, -1, dtype=np.int32)
    destinations[:count] = positions[:count]
    info = PageBatchInfo(
        slot_ids=hax.named(jnp.array([0], jnp.int32), "seq"),
        page_indices=hax.named(jnp.arange(8, dtype=jnp.int32)[None], ("seq", "page")),
        seq_lens=hax.named(jnp.array([position + count], jnp.int32), "seq"),
        cu_q_lens=hax.named(jnp.array([0, count], jnp.int32), "seq"),
        num_seqs=jnp.array(1, jnp.int32),
        new_token_dests=hax.named(jnp.asarray(destinations), "position"),
        page_size=2,
    )
    return _PackedInputs(
        hax.named(jnp.asarray(tokens), "position"), info, hax.named(jnp.asarray(positions), "position")
    )


@skip_if_no_torch
@jax.default_matmul_precision("highest")
def test_eagle3_full_forward_matches_pinned_torch_recipe(tmp_path):
    hf_config, state, embedding = _checkpoint()
    count = jax.device_count()
    tokens = np.broadcast_to(np.array([2, 4, 8, 11, 16], dtype=np.int32), (count, 5))
    auxiliary = np.random.default_rng(9).normal(size=(count, 5, 48)).astype(np.float32)
    expected_logits, expected_hidden = _torch_forward(state, embedding, tokens, auxiliary, project=True)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        (tmp_path / "config.json").write_text(json.dumps(hf_config))
        save_file(state, tmp_path / "model.safetensors")
        model = Eagle3Draft.from_checkpoint(tmp_path, target_embedding=jnp.asarray(embedding))
        output = eqx.filter_jit(lambda m: m(jnp.asarray(tokens), jnp.asarray(auxiliary)))(model)
        np.testing.assert_allclose(output.logits, expected_logits, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(output.hidden_states, expected_hidden, rtol=1e-4, atol=1e-4)
        mapped = np.flatnonzero(state["t2d"])[expected_logits.argmax(-1)]
        np.testing.assert_array_equal(model.greedy_tokens(output), mapped)


@skip_if_no_torch
@jax.default_matmul_precision("highest")
def test_eagle3_learned_recurrent_proposals_match_full_torch_with_separate_cache():
    hf_config, state, embedding = _checkpoint()
    config = Eagle3Config.from_hf_config(hf_config)
    config = dataclasses.replace(config, inference_attention_implementation="reference")
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Eagle3Draft.from_state_dict(config, state, target_embedding=jnp.asarray(embedding))
        cache = model.initial_cache(PageTableSpec(4, 2), dtype=jnp.float32)
        auxiliary = jnp.asarray(np.random.default_rng(51).normal(size=(1, 48)).astype(np.float32))
        hidden = np.asarray(model.project_target_states(auxiliary))[0]
        token = 2
        past_hidden, past_tokens, expected_proposals = [], [], []
        decode = eqx.filter_jit(lambda m, t, h, c, b, p: m.decode(t, h, c, b, p))
        for position in range(6):
            past_tokens.append(token)
            past_hidden.append(hidden)
            expected_logits, expected_hidden = _torch_forward(
                state,
                embedding,
                np.array(past_tokens)[None],
                np.array(past_hidden)[None],
                project=False,
            )
            tokens, info, positions = _decode_inputs([token], position)
            hidden_buffer = np.zeros((tokens.size, 16), dtype=np.float32)
            hidden_buffer[0] = hidden
            result = decode(model, tokens, jnp.asarray(hidden_buffer), cache, info, positions)
            np.testing.assert_allclose(
                np.asarray(result.output.logits)[0], expected_logits[0, -1], rtol=1e-4, atol=1e-4
            )
            np.testing.assert_allclose(
                np.asarray(result.output.hidden_states)[0], expected_hidden[0, -1], rtol=1e-4, atol=1e-4
            )
            token = int(model.greedy_tokens(result.output)[0])
            assert token == np.flatnonzero(state["t2d"])[expected_logits[0, -1].argmax()]
            expected_proposals.append(token)
            hidden = np.asarray(result.output.hidden_states)[0]
            cache = result.cache
        tokens, info, positions = _decode_inputs([2], 0)
        target_auxiliary = np.zeros((tokens.size, 48), dtype=np.float32)
        target_auxiliary[0] = np.asarray(auxiliary)[0]
        proposals = eqx.filter_jit(
            lambda m, c: propose_eagle3(
                m,
                tokens,
                jnp.asarray(target_auxiliary),
                c,
                info,
                positions,
                num_draft_tokens=6,
            )
        )(model, model.initial_cache(PageTableSpec(4, 2), dtype=jnp.float32))
        np.testing.assert_array_equal(proposals.token_ids, [expected_proposals])
        np.testing.assert_allclose(
            proposals.tentative_cache.kv_pages.array, cache.kv_pages.array, rtol=1e-4, atol=1e-4
        )


def _target_config():
    return SnowballConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=24,
        shared_expert_intermediate_dim=24,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        max_seq_len=16,
        sliding_window=2,
        initializer_std=0.2,
        attention_implementation="reference",
        inference_attention_implementation="reference",
    )


@pytest.mark.parametrize("proposal_case", ["learned", "accept_all", "accept_prefix", "cancelled"])
@jax.default_matmul_precision("highest")
def test_learned_eagle_proposals_verify_against_real_snowball_target(proposal_case):
    hf_config, state, _ = _checkpoint()
    config = dataclasses.replace(
        Eagle3Config.from_hf_config(hf_config), inference_attention_implementation="reference"
    )
    target_config = _target_config()
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        target = SnowballLMHeadModel.init(Axis("vocab", 32), target_config, key=jax.random.key(13))
        draft = Eagle3Draft.from_state_dict(config, state, target_embedding=target.transformer.token_embed)
        prompt = _decode_inputs([2, 7, 5], 0)
        prefilled = eqx.filter_jit(
            lambda m, c: m.decode_with_auxiliary_states(
                prompt.tokens,
                c,
                prompt.batch_info,
                prompt.positions,
                layers=config.auxiliary_layers,
            )
        )(target, target.initial_cache(PageTableSpec(8, 2), dtype=jnp.float32))
        pending = int(np.asarray(prefilled.logits.array)[2].argmax())
        target_states = np.asarray(prefilled.auxiliary_states.array)
        draft_prefix = _decode_inputs([7, 5], 0)
        projected = draft.project_target_states(jnp.asarray(target_states))
        draft_prefilled = eqx.filter_jit(
            lambda m, c: m.decode(
                draft_prefix.tokens,
                projected,
                c,
                draft_prefix.batch_info,
                draft_prefix.positions,
            )
        )(draft, draft.initial_cache(PageTableSpec(8, 2), dtype=jnp.float32))
        seed = _decode_inputs([pending], 2)
        auxiliary = np.zeros_like(target_states)
        auxiliary[0] = target_states[2]
        proposed = eqx.filter_jit(
            lambda m, c: propose_eagle3(
                m,
                seed.tokens,
                jnp.asarray(auxiliary),
                c,
                seed.batch_info,
                seed.positions,
                num_draft_tokens=3,
            )
        )(draft, draft_prefilled.cache)

        # Sequential target generation is independent of the learned proposal path.
        decode_target = eqx.filter_jit(lambda m, c, a: m.decode(a.tokens, c, a.batch_info, a.positions))
        cache = prefilled.cache
        oracle_ids, oracle_scores, oracle_logits = [], [], []
        token = pending
        for step in range(9):
            logits, cache = decode_target(target, cache, _decode_inputs([token], 3 + step))
            row = np.asarray(logits.array, dtype=np.float64)[0]
            token = int(row.argmax())
            oracle_ids.append(token)
            oracle_scores.append(row[token] - np.logaddexp.reduce(row))
            oracle_logits.append(row)
        proposal_ids = np.asarray(proposed.token_ids)[0].tolist()
        if proposal_case in ("accept_all", "accept_prefix"):
            # Keep the learned tentative cache: it must be rebuilt from the target
            # even when a better proposal sequence is supplied to verification.
            proposal_ids = oracle_ids[:3].copy()
            if proposal_case == "accept_prefix":
                proposal_ids[-1] = (proposal_ids[-1] + 1) % 32
        block = _decode_inputs([pending, *proposal_ids], 3)
        verified = eqx.filter_jit(
            lambda m, c: verify_snowball_proposals(
                m,
                block.tokens,
                c,
                block.batch_info,
                block.positions,
                jnp.zeros(1),
                jnp.array([5]),
                jnp.array([proposal_case == "cancelled"]),
                max_draft_tokens=3,
                auxiliary_layers=config.auxiliary_layers,
                key=jax.random.key(19),
                logprobs_mode="raw_logprobs",
            )
        )(target, prefilled.cache)
        length = int(np.asarray(verified.tokens.lengths)[0])
        if proposal_case == "cancelled":
            reconciled = eqx.filter_jit(
                lambda m, c: reconcile_eagle3(
                    m,
                    seed.tokens,
                    jnp.asarray(auxiliary),
                    c,
                    seed.batch_info,
                    verified,
                    token_capacity=seed.tokens.size,
                )
            )(draft, proposed.tentative_cache)
            np.testing.assert_array_equal(reconciled.committed_seq_lens.array, [2])
            np.testing.assert_array_equal(reconciled.pending_tokens, [-1])
            np.testing.assert_array_equal(reconciled.target_auxiliary, 0)
            retried = eqx.filter_jit(
                lambda m, c: propose_eagle3(
                    m,
                    seed.tokens,
                    jnp.asarray(auxiliary),
                    c,
                    seed.batch_info,
                    seed.positions,
                    num_draft_tokens=3,
                )
            )(draft, reconciled.cache)
            np.testing.assert_array_equal(retried.token_ids, proposed.token_ids)
            np.testing.assert_allclose(
                retried.tentative_cache.kv_pages.array, proposed.tentative_cache.kv_pages.array, rtol=1e-4, atol=1e-4
            )
            return
        np.testing.assert_array_equal(np.asarray(verified.tokens.token_ids)[0, :length], oracle_ids[:length])
        np.testing.assert_allclose(np.asarray(verified.tokens.logprobs)[0, :length], oracle_scores[:length], atol=1e-5)
        next_input = _decode_inputs([oracle_ids[length - 1]], 3 + length)
        logits, _ = decode_target(target, verified.cache, next_input)
        np.testing.assert_allclose(np.asarray(logits.array)[0], oracle_logits[length], rtol=1e-4, atol=1e-4)

        reconciled = eqx.filter_jit(
            lambda m, c: reconcile_eagle3(
                m,
                seed.tokens,
                jnp.asarray(auxiliary),
                c,
                seed.batch_info,
                verified,
                token_capacity=seed.tokens.size,
            )
        )(draft, proposed.tentative_cache)
        np.testing.assert_array_equal(reconciled.committed_seq_lens.array, [2 + length])
        assert int(np.asarray(reconciled.pending_tokens)[0]) == oracle_ids[length - 1]
        next_seed = _decode_inputs([oracle_ids[length - 1]], 2 + length)
        next_auxiliary = np.zeros_like(auxiliary)
        next_auxiliary[0] = np.asarray(reconciled.target_auxiliary)[0]
        next_proposals = eqx.filter_jit(
            lambda m, c: propose_eagle3(
                m,
                next_seed.tokens,
                jnp.asarray(next_auxiliary),
                c,
                next_seed.batch_info,
                next_seed.positions,
                num_draft_tokens=3,
            )
        )(draft, reconciled.cache)

        # Rebuild the draft prefix from original target rows, without speculative KV.
        replay_tokens = [7, 5, pending, *oracle_ids[: length - 1]]
        replay_auxiliary = np.zeros_like(auxiliary)
        replay_auxiliary[:3] = target_states[:3]
        replay_auxiliary[3 : 2 + length] = np.asarray(verified.auxiliary_states)[0, : length - 1]
        replay = _decode_inputs(replay_tokens, 0)
        fresh = eqx.filter_jit(
            lambda m, c: m.decode(
                replay.tokens,
                m.project_target_states(jnp.asarray(replay_auxiliary)),
                c,
                replay.batch_info,
                replay.positions,
            )
        )(draft, draft.initial_cache(PageTableSpec(8, 2), dtype=jnp.float32))
        fresh_proposals = eqx.filter_jit(
            lambda m, c: propose_eagle3(
                m,
                next_seed.tokens,
                jnp.asarray(next_auxiliary),
                c,
                next_seed.batch_info,
                next_seed.positions,
                num_draft_tokens=3,
            )
        )(draft, fresh.cache)
        np.testing.assert_array_equal(next_proposals.token_ids, fresh_proposals.token_ids)
        np.testing.assert_allclose(
            next_proposals.tentative_cache.kv_pages.array,
            fresh_proposals.tentative_cache.kv_pages.array,
            rtol=1e-4,
            atol=1e-4,
        )
        second_block = _decode_inputs(
            [oracle_ids[length - 1], *np.asarray(next_proposals.token_ids)[0].tolist()], 3 + length
        )
        second = eqx.filter_jit(
            lambda m, c: verify_snowball_proposals(
                m,
                second_block.tokens,
                c,
                second_block.batch_info,
                second_block.positions,
                jnp.zeros(1),
                jnp.array([5]),
                jnp.zeros(1, dtype=bool),
                max_draft_tokens=3,
                auxiliary_layers=config.auxiliary_layers,
                key=jax.random.key(20),
                logprobs_mode="raw_logprobs",
            )
        )(target, verified.cache)
        second_length = int(np.asarray(second.tokens.lengths)[0])
        np.testing.assert_array_equal(
            np.asarray(second.tokens.token_ids)[0, :second_length], oracle_ids[length : length + second_length]
        )
        np.testing.assert_allclose(
            np.asarray(second.tokens.logprobs)[0, :second_length],
            oracle_scores[length : length + second_length],
            atol=1e-5,
        )


@pytest.mark.parametrize("termination", ["length", "stop", "abort", "accept_all", "accept_stop", "short_prompt"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_resident_eagle_generation_matches_ordinary_target_and_retry(termination, dtype):
    hf_config, weights, _ = _checkpoint()
    draft_config = dataclasses.replace(
        Eagle3Config.from_hf_config(hf_config), inference_attention_implementation="reference"
    )
    config = InferenceEngineConfig(
        max_seq_len=16,
        max_seqs=1,
        max_seqs_in_prefill=1,
        max_pages=10,
        page_size=2,
        max_prefill_size=16,
        max_tokens_per_round=4,
        max_queued_tokens=16,
        max_rounds=1,
        compute_dtype=dtype,
    )
    request = Request([2, 7, 5], 17, dataclasses.replace(SeqDecodingParams.default(), max_num_tokens=jnp.array(14)), 1)
    if termination == "short_prompt":
        request = dataclasses.replace(request, prompt_tokens=[2])
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        target = SnowballLMHeadModel.init(Axis("vocab", 32), _target_config(), key=jax.random.key(13))
        target = jax.tree.map(lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, target)
        if termination in ("accept_all", "accept_stop"):
            # Uniform target and draft heads make every greedy proposal agree,
            # exposing bonus-token and within-block stop boundaries.
            target = eqx.tree_at(
                lambda m: m.transformer.output_proj, target, jnp.zeros_like(target.transformer.output_proj)
            )
            weights["lm_head.weight"] = np.zeros_like(weights["lm_head.weight"])
            weights["d2t"][0] = 0
            weights["t2d"][0] = True
            weights["t2d"][1] = False
        draft = Eagle3Draft.from_state_dict(draft_config, weights, target_embedding=target.transformer.token_embed)
        ordinary = InferenceEngine.from_model_with_config(target, None, config)
        expected = ordinary.generate([request])
        engine = InferenceEngine.from_model_with_config(
            target, None, dataclasses.replace(config, num_eagle3_tokens=3), draft=draft
        )
        if termination in ("stop", "accept_stop"):
            request = dataclasses.replace(
                request,
                decode_params=dataclasses.replace(
                    request.decode_params,
                    stop_tokens=hax.named(jnp.array([expected.tokens[0][2:5]]), ("stop_seq", "position")),
                ),
            )
            expected = ordinary.generate([request])
        observed = []

        def capture(_request_id, results):
            observed.append(tuple(results[0].token_list))

        result = engine.generate(
            [request],
            output_callback=capture,
            should_abort=(lambda _: bool(observed and len(observed[-1]) >= 3)) if termination == "abort" else None,
        )
        count = len(result.tokens[0])
        assert result.tokens[0] == expected.tokens[0][:count]
        np.testing.assert_allclose(result.logprobs[0], expected.logprobs[0][:count], atol=1e-4, rtol=1e-4)
        assert all(list(partial) == expected.tokens[0][: len(partial)] for partial in observed)
        if termination == "abort":
            assert result.finish_reasons == [FinishReason.ABORT]
            assert 0 < count < len(expected.tokens[0])
            resumed = engine.generate(
                [dataclasses.replace(request, prompt_tokens=request.prompt_tokens + result.tokens[0])]
            )
            assert result.tokens[0] + resumed.tokens[0] == expected.tokens[0]
            np.testing.assert_allclose(
                result.logprobs[0] + resumed.logprobs[0], expected.logprobs[0], atol=1e-4, rtol=1e-4
            )
        else:
            assert result.tokens == expected.tokens
            assert result.finish_reasons == expected.finish_reasons
        if termination == "accept_all":
            assert max(np.diff([len(partial) for partial in observed])) == 4
        if termination == "accept_stop":
            assert result.tokens == [[0, 0, 0]]
            assert result.finish_reasons == [FinishReason.STOP]

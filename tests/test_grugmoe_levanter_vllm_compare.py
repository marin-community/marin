# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from experiments.grug.moe import real_checkpoint_levanter_vllm_compare as compare
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.real_checkpoint_levanter_vllm_compare import (
    _executable_mlp_from_legacy_split,
    choose_prompt_tokenization,
    compare_levanter_to_vllm,
    derive_vllm_continuation_token_ids,
)


class FakeTokenizer:
    def __init__(self):
        self._encodings = {
            ("prompt", True): [0, 10, 11],
            ("prompt", False): [10, 11],
            ("prompthello", True): [0, 10, 11, 20],
            ("hello", False): [20],
        }

    def encode(self, text: str, *, add_special_tokens: bool = False):
        return list(self._encodings[(text, add_special_tokens)])

    def decode(self, ids, *, skip_special_tokens: bool = False):
        del skip_special_tokens
        return "".join({0: "<bos>", 10: "pro", 11: "mpt", 20: "hello", 21: "bye"}[int(i)] for i in ids)


def test_choose_prompt_tokenization_matches_vllm_prompt_count():
    tokenizer = FakeTokenizer()

    prompt_ids, report = choose_prompt_tokenization(tokenizer, "prompt", expected_prompt_token_count=3)

    assert prompt_ids == [0, 10, 11]
    assert report["selected_add_special_tokens"] is True
    assert report["matches_expected_count"] is True


def test_derive_vllm_continuation_ids_from_prompt_plus_completion_suffix():
    tokenizer = FakeTokenizer()

    token_ids, source, note = derive_vllm_continuation_token_ids(
        tokenizer,
        prompt="prompt",
        prompt_ids=[0, 10, 11],
        prompt_add_special_tokens=True,
        completion="hello",
        raw_completion_token_ids=None,
    )

    assert token_ids == [20]
    assert source == "derived_prompt_plus_completion_suffix"
    assert note == {}


def test_compare_reports_first_token_divergence():
    tokenizer = FakeTokenizer()
    levanter_result = {
        "generated_token_ids": [20, 21],
        "generated_token_texts": ["hello", "bye"],
        "decoded_text": "hellobye",
        "selected_token_logprobs": [-0.1, -0.2],
    }
    vllm_reference = {"prompt": "prompt", "completion": "hellohello", "raw_logprobs": None}

    comparison = compare_levanter_to_vllm(
        tokenizer=tokenizer,
        prompt="prompt",
        tokenization={"matches_expected_count": True},
        levanter_result=levanter_result,
        vllm_reference=vllm_reference,
        vllm_continuation_token_ids=[20, 20],
        vllm_token_ids_source="derived_prompt_plus_completion_suffix",
    )

    assert comparison["passed"] is False
    assert comparison["divergence"]["generated_token_index"] == 1
    assert comparison["divergence"]["levanter_token_id"] == 21
    assert comparison["divergence"]["vllm_token_id"] == 20
    assert comparison["divergence"]["levanter_selected_token_logprob"] == -0.2
    assert comparison["selected_token_logprobs_comparable"] is False


def test_compare_passes_when_prompt_text_count_text_and_tokens_match():
    tokenizer = FakeTokenizer()
    levanter_result = {
        "generated_token_ids": [20],
        "generated_token_texts": ["hello"],
        "decoded_text": "hello",
        "selected_token_logprobs": [-0.1],
    }
    vllm_reference = {"prompt": "prompt", "completion": "hello", "raw_logprobs": None}

    comparison = compare_levanter_to_vllm(
        tokenizer=tokenizer,
        prompt="prompt",
        tokenization={"matches_expected_count": True},
        levanter_result=levanter_result,
        vllm_reference=vllm_reference,
        vllm_continuation_token_ids=[20],
        vllm_token_ids_source="derived_prompt_plus_completion_suffix",
    )

    assert comparison["passed"] is True
    assert comparison["divergence"] is None


def test_executable_mlp_from_legacy_split_concatenates_gate_and_up_weights():
    cfg = GrugModelConfig(
        vocab_size=32,
        hidden_dim=2,
        intermediate_dim=3,
        shared_expert_intermediate_dim=0,
        num_experts=2,
        num_experts_per_token=1,
        num_layers=1,
        num_heads=1,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=4,
    )
    split = SimpleNamespace(
        router=jnp.zeros((2, 2)),
        router_bias=jnp.zeros((2,)),
        w_gate=jnp.ones((2, 2, 3)),
        w_up=jnp.full((2, 2, 3), 2.0),
        w_down=jnp.full((2, 3, 2), 3.0),
        cfg=cfg,
    )

    mlp = _executable_mlp_from_legacy_split(split)

    np.testing.assert_array_equal(np.asarray(mlp.expert_mlp.w_gate_up[..., :3]), np.ones((2, 2, 3)))
    np.testing.assert_array_equal(np.asarray(mlp.expert_mlp.w_gate_up[..., 3:]), np.full((2, 2, 3), 2.0))
    np.testing.assert_array_equal(np.asarray(mlp.expert_mlp.w_down), np.full((2, 3, 2), 3.0))


def test_main_submits_compare_job_with_explicit_reference(monkeypatch):
    calls = []

    def fake_submit_compare(config, *, tpu_type, region, ram, disk, job_name):
        calls.append((config, tpu_type, region, ram, disk, job_name))

    monkeypatch.setattr(compare, "submit_compare", fake_submit_compare)
    reference_path = "gs://marin-eu-west4/tmp/ttl=14d/grugmoe-real-checkpoint-vllm-smoke/test/result.json"

    result = compare.main(
        [
            "--vllm-result-path",
            reference_path,
            "--output-dir",
            "gs://marin-eu-west4/tmp/ttl=14d/grugmoe-real-checkpoint-levanter-vllm-compare/test",
            "--cache-dir",
            "gs://marin-eu-west4/compilation-cache/grugmoe-real-checkpoint-levanter-vllm-compare/test",
            "--job-name",
            "compare-test",
        ]
    )

    assert result == 0
    assert len(calls) == 1
    config, tpu_type, region, ram, disk, job_name = calls[0]
    assert config.vllm_result_path == reference_path
    assert tpu_type == compare.DEFAULT_TPU_TYPE
    assert region == compare.DEFAULT_REGION
    assert ram == compare.DEFAULT_RAM
    assert disk == compare.DEFAULT_DISK
    assert job_name == "compare-test"

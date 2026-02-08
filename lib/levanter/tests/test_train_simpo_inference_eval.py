# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from levanter.main.train_simpo import (
    InferenceEvalConfig,
    _build_host_local_engine_config,
    _normalize_eval_at_steps,
    _resolve_inference_prompts,
    _shard_prompts_for_host,
    _should_run_inference_eval_step,
)


def test_normalize_eval_at_steps_dedupes_and_validates() -> None:
    assert _normalize_eval_at_steps([2, 4, 2]) == {2, 4}
    assert _normalize_eval_at_steps(None) is None
    with pytest.raises(ValueError, match="must be positive"):
        _normalize_eval_at_steps([0, 2])


def test_should_run_inference_eval_step_exact_schedule() -> None:
    eval_steps = {2, 5}
    assert _should_run_inference_eval_step(step=1, eval_every=10, eval_at_steps=eval_steps) is False
    assert _should_run_inference_eval_step(step=2, eval_every=10, eval_at_steps=eval_steps) is True
    assert _should_run_inference_eval_step(step=5, eval_every=10, eval_at_steps=eval_steps) is True
    assert _should_run_inference_eval_step(step=6, eval_every=10, eval_at_steps=eval_steps) is False


def test_should_run_inference_eval_step_periodic() -> None:
    assert _should_run_inference_eval_step(step=1, eval_every=2, eval_at_steps=None) is False
    assert _should_run_inference_eval_step(step=2, eval_every=2, eval_at_steps=None) is True
    assert _should_run_inference_eval_step(step=4, eval_every=2, eval_at_steps=None) is True


def test_resolve_inference_prompts_synthetic() -> None:
    config = InferenceEvalConfig(
        synthetic_prompt_count=3,
        synthetic_prompt_template="Prompt {index}",
    )
    assert _resolve_inference_prompts(config) == ["Prompt 0", "Prompt 1", "Prompt 2"]


def test_resolve_inference_prompts_from_file(tmp_path) -> None:
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("alpha\n\nbeta\n", encoding="utf-8")

    config = InferenceEvalConfig(prompts_path=str(prompts_file))
    assert _resolve_inference_prompts(config) == ["alpha", "beta"]


def test_shard_prompts_for_host_covers_all_prompts() -> None:
    prompts = [f"prompt-{i}" for i in range(10)]
    shards = [_shard_prompts_for_host(prompts, process_index=i, process_count=3) for i in range(3)]

    rebuilt = []
    for shard_prompts, start, end in shards:
        assert shard_prompts == prompts[start:end]
        rebuilt.extend(shard_prompts)
    assert rebuilt == prompts


def test_build_host_local_engine_config_downsizes_shard() -> None:
    config = InferenceEvalConfig(
        max_seq_len=4096,
        max_seqs=128,
        max_seqs_in_prefill=128,
        max_tokens_per_round=128,
        max_queued_tokens=128,
        max_prefill_size=4096,
        max_pages=1088,
    )
    prompt_tokens = [[1, 2, 3] for _ in range(32)]
    engine_config = _build_host_local_engine_config(
        inference_config=config,
        max_seq_len=4096,
        prompt_tokens=prompt_tokens,
        devices=None,
        shard_locally=True,
    )

    assert engine_config.max_seqs == 32
    assert engine_config.max_seqs_in_prefill == 32
    assert engine_config.max_tokens_per_round == 128
    assert engine_config.max_queued_tokens >= 128


def test_build_host_local_engine_config_rejects_undersized_global_max_seqs() -> None:
    config = InferenceEvalConfig(
        max_seq_len=4096,
        max_seqs=4,
        max_seqs_in_prefill=4,
    )
    prompt_tokens = [[1, 2, 3] for _ in range(8)]
    with pytest.raises(ValueError, match="max_seqs"):
        _build_host_local_engine_config(
            inference_config=config,
            max_seq_len=4096,
            prompt_tokens=prompt_tokens,
            devices=None,
            shard_locally=False,
        )

# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference import snowball_rollout_replay as candidate
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.models.snowball import SnowballLMHeadModel

_ROOT = Path(__file__).parents[4]
_BENCHMARK = _ROOT / "lib/levanter/benchmarks/snowball_rollout_performance.py"
_API_PATH = "lib/levanter/src/levanter/inference/snowball_rollout_replay.py"


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(_BENCHMARK.parent))
    spec = importlib.util.spec_from_file_location("snowball_rollout_performance_benchmark", _BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "model_seed,sample_seed,generated_tokens,cache_dtype",
    [(41, 154, 1, jnp.float32), (38127, 44017, 5, jnp.bfloat16)],
)
def test_device_generation_matches_public_decode_all_evidence(
    benchmark, model_seed, sample_seed, generated_tokens, cache_dtype
):
    config = dataclasses.replace(
        benchmark.qualified_config(implementation="native", backend="cpu"),
        max_seq_len=8,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        sliding_window=4,
    )
    fixture = benchmark.make_fixture(sequence_length=8, page_size=4)
    sampling_key = jax.random.key(sample_seed)
    total_length = 3 + generated_tokens
    decode = eqx.filter_jit(lambda m, c, ids, info, positions: m.decode_with_route_captures(ids, c, info, positions))
    teacher_forward = eqx.filter_jit(lambda m, ids, routes: m(ids, route_captures=routes))
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)), jax.default_matmul_precision("highest"):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 23), config, key=jax.random.key(model_seed))
        prompts = hax.named(jnp.asarray(np.asarray(fixture.tokens.array)[:, :3]), ("batch", "position"))
        results, teachers = [], []
        for page_map in fixture.page_maps:
            page_map = page_map[:, : (total_length + 3) // 4]
            current = candidate.generate_with_route_captures(
                model,
                prompts,
                sampling_key=sampling_key,
                generated_tokens=generated_tokens,
                page_map=page_map,
                page_size=4,
                cache_dtype=cache_dtype,
            )
            assert current.serving_logits.shape == (4, total_length, 23)
            assert len(current.serving_captures) == generated_tokens + 1
            np.testing.assert_array_equal(current.predictor_positions, np.arange(2, total_length - 1))
            np.testing.assert_array_equal(np.asarray(current.tokens.array)[:, :3], np.asarray(prompts.array))
            # Public decode receives independently built paging metadata and its
            # own sampled inputs; it never calls the generation implementation.
            cache = model.initial_cache(PageTableSpec(int(page_map.max()) + 1, 4, 4), dtype=cache_dtype)
            inputs = np.asarray(prompts.array)
            for call in range(generated_tokens + 1):
                start, width = (0, 3) if call == 0 else (call + 2, 1)
                positions = np.tile(np.arange(start, start + width, dtype=np.int32), 4)
                slots = np.repeat(np.arange(4), width)
                info = PageBatchInfo(
                    slot_ids=hax.named(jnp.arange(4, dtype=jnp.int32), "seq"),
                    page_indices=hax.named(jnp.asarray(page_map), ("seq", "page")),
                    seq_lens=hax.named(jnp.full(4, start + width, dtype=jnp.int32), "seq"),
                    cu_q_lens=hax.named(jnp.arange(5, dtype=jnp.int32) * width, "seq"),
                    num_seqs=jnp.asarray(4, dtype=jnp.int32),
                    new_token_dests=hax.named(
                        jnp.asarray(page_map[slots, positions // 4] * 4 + positions % 4), "position"
                    ),
                    page_size=4,
                )
                logits, cache, capture = decode(
                    model,
                    cache,
                    hax.named(jnp.asarray(inputs.reshape(-1)), "position"),
                    info,
                    hax.named(jnp.asarray(positions), "position"),
                )
                expected = np.asarray(logits.array).reshape(4, width, 23)
                np.testing.assert_array_equal(
                    current.serving_logits[:, start : start + width].view(np.uint32), expected.view(np.uint32)
                )
                comparison = benchmark.bit_comparison(current.serving_captures[call], jax.device_get(capture))
                assert comparison["bitwise"], comparison
                if call < generated_tokens:
                    inputs = np.asarray(
                        jax.random.categorical(jax.random.fold_in(sampling_key, call), jnp.asarray(expected[:, -1]))
                    ).astype(np.int32)
                    np.testing.assert_array_equal(current.sampled_token_ids[:, call], inputs)
                    np.testing.assert_array_equal(np.asarray(current.tokens.array)[:, 3 + call], inputs)
            # The final sampled input is decoded for its routes and retained
            # next logits even though those logits produce no sampled token.
            np.testing.assert_array_equal(
                current.unsampled_next_logits.view(np.uint32), expected[:, -1].view(np.uint32)
            )
            np.testing.assert_array_equal(current.serving_captures[-1].token_ids, current.sampled_token_ids[:, -1])
            np.testing.assert_array_equal(current.serving_captures[-1].position_ids, np.full(4, total_length - 1))
            np.testing.assert_array_equal(current.teacher_capture.token_ids, np.asarray(current.tokens.array))
            assert current.teacher_capture.segment_ids is None
            # Check teacher row binding slot by slot, independently of the
            # implementation's reshape/concatenate reconstruction.
            for batch in range(4):
                for position in range(total_length):
                    call, row = (0, batch * 3 + position) if position < 3 else (position - 2, batch)
                    for layer in range(config.num_layers):
                        serving = current.serving_captures[call].route_captures[layer]
                        teacher = current.teacher_capture.route_captures[layer]
                        for field in ("selected_experts", "token_valid", "qb_alpha"):
                            np.testing.assert_array_equal(
                                np.asarray(getattr(teacher, field))[batch * total_length + position],
                                np.asarray(getattr(serving, field))[row],
                            )
            current_teacher = candidate.replay_generated_routes(model, current)
            public_teacher = np.asarray(teacher_forward(model, current.tokens, current.teacher_capture).array)
            np.testing.assert_array_equal(current_teacher.view(np.uint32), public_teacher.view(np.uint32))
            results.append(current)
            teachers.append(current_teacher)
        comparison = benchmark.compare_generated(*results, *teachers, physical_layout=False)
        assert comparison["bitwise"], comparison


@pytest.mark.parametrize("supply_substitute_hash", [False, True])
def test_performance_cli_rejects_substitute_baseline(tmp_path, supply_substitute_hash):
    output = tmp_path / "ab.json"
    substitute = (_ROOT / _API_PATH).read_bytes()
    reference_api = tmp_path / "substitute.py"
    reference_api.write_bytes(substitute)
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_NUM_CPU_DEVICES": "4",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONOPTIMIZE": "0",
        "PYTHONPATH": os.pathsep.join((str(_ROOT / "lib/levanter/src"), str(_ROOT / "lib/haliax/src"))),
    }
    # Preserve the lexical venv interpreter, whose resolved target is bare Python.
    command = [
        os.path.abspath(sys.executable),
        "-B",
        str(_BENCHMARK),
        "--backend",
        "cpu",
        "--reference-api",
        str(reference_api),
        "--output",
        str(output),
    ]
    if supply_substitute_hash:
        command.extend(["--reference-sha256", hashlib.sha256(substitute).hexdigest()])
    completed = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0, completed.stderr
    failure = json.loads(output.read_text())
    assert failure["status"] == "failed"
    assert failure["error_type"] == "ValueError"
    assert failure["runs"] == []
    assert failure["native_qualification"]["status"] == "unexecuted"

# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
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
from levanter.inference.page_table import PageTableSpec
from levanter.inference.snowball_rollout_replay import generate_with_route_captures, replay_generated_routes
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

_BENCHMARK = Path(__file__).parents[2] / "benchmarks" / "snowball_rollout_replay.py"
_BASE = "1b3d5e5cb02c9306657f8aae17b76979c7b4bbec"


def _config(architecture):
    return SnowballConfig(
        vocab_size=23,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_layers=2 if architecture == "mixed" else 1,
        num_experts=4,
        num_experts_per_token=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        attention_implementation="reference",
        inference_attention_implementation="reference",
        moe_implementation="ring",
    )


@pytest.mark.parametrize("architecture", ["nope", "mixed"])
@pytest.mark.timeout(180)
def test_generated_routes_preserve_public_decode_sampling_alignment_and_batch_major_order(architecture):
    config = _config(architecture)
    sampling_key = jax.random.key(154)
    page_map = np.asarray([[6, 1], [7, 0], [4, 3], [5, 2]], dtype=np.int32)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)), jax.default_matmul_precision("highest"):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 23), config, key=jax.random.key(41))
        prompts = hax.named(
            jnp.asarray([[1, 2, 7], [4, 6, 3], [8, 0, 5], [9, 2, 1]], jnp.int32), ("batch", "position")
        )
        generated = generate_with_route_captures(
            model,
            prompts,
            sampling_key=sampling_key,
            generated_tokens=5,
            page_map=page_map,
            page_size=4,
            cache_dtype=jnp.float32,
        )
        teacher = replay_generated_routes(model, generated)
        public_teacher = eqx.filter_jit(lambda m, ids, routes: m(ids, route_captures=routes))(
            model, generated.tokens, generated.teacher_capture
        )
        # Independent public non-capturing decode is the oracle for original
        # serving logits, not a fresh teacher-side route capture.
        cache = model.initial_cache(PageTableSpec(8, 4, 4), dtype=jnp.float32)
        decode = eqx.filter_jit(lambda m, c, ids, info, positions: m.decode(ids, c, info, positions))
        for call, capture in enumerate(generated.serving_captures):
            logits, cache = decode(
                model,
                cache,
                hax.named(jnp.asarray(capture.token_ids), "position"),
                capture.batch_info,
                hax.named(jnp.asarray(capture.position_ids), "position"),
            )
            width = 3 if call == 0 else 1
            start = 0 if call == 0 else call + 2
            expected = np.asarray(logits.array).reshape(4, width, 23)
            np.testing.assert_array_equal(
                generated.serving_logits[:, start : start + width].view(np.uint32), expected.view(np.uint32)
            )
        # Position 2 predicts the first sampled token; position 6 predicts the
        # fifth. Position 7 is processed for routes but predicts no sampled ID.
        np.testing.assert_array_equal(generated.predictor_positions, [2, 3, 4, 5, 6])
        np.testing.assert_array_equal(np.asarray(generated.tokens.array)[:, :3], np.asarray(prompts.array))
        np.testing.assert_array_equal(np.asarray(generated.tokens.array)[:, 3:], generated.sampled_token_ids)
        for step, position in enumerate([2, 3, 4, 5, 6]):
            sampled = jax.random.categorical(
                jax.random.fold_in(sampling_key, step), jnp.asarray(generated.serving_logits[:, position]), axis=-1
            )
            np.testing.assert_array_equal(generated.sampled_token_ids[:, step], np.asarray(sampled))
        np.testing.assert_array_equal(generated.unsampled_next_logits, generated.serving_logits[:, 7])
        # Check EACH slot/position/layer/top-k row against its original call,
        # without using the implementation's reshape/concatenate algorithm.
        for batch in range(4):
            for position in range(8):
                call, serving_row = (0, batch * 3 + position) if position < 3 else (position - 2, batch)
                original = generated.serving_captures[call]
                assert original.token_ids[serving_row] == np.asarray(generated.tokens.array)[batch, position]
                assert original.position_ids[serving_row] == position
                assert int(original.batch_info.slot_ids.array[batch]) == batch
                destination = page_map[batch, position // 4] * 4 + position % 4
                assert original.batch_info.new_token_dests.array[serving_row] == destination
                for layer in range(config.num_layers):
                    actual = generated.teacher_capture.route_captures[layer]
                    source = original.route_captures[layer]
                    for field in ("selected_experts", "token_valid", "qb_alpha"):
                        np.testing.assert_array_equal(
                            np.asarray(getattr(actual, field))[batch * 8 + position],
                            np.asarray(getattr(source, field))[serving_row],
                        )
        np.testing.assert_array_equal(teacher.view(np.uint32), np.asarray(public_teacher.array).view(np.uint32))
        # Portable arithmetic is checked at the existing reference tolerance;
        # this assertion is not an exact-bit or native qualification claim.
        np.testing.assert_allclose(teacher, generated.serving_logits, rtol=1e-4, atol=1e-4)
        # A consequential probe: altering only the retained serving expert IDs
        # changes public training logits. The payload cannot be ignored or
        # replaced by teacher-side routing and still satisfy this assertion.
        routes = generated.teacher_capture.route_captures
        for layer, route in enumerate(routes):
            changed = route._replace(selected_experts=(route.selected_experts + 1) % config.num_experts)
            changed_routes = (*routes[:layer], changed, *routes[layer + 1 :])
            changed_capture = generated.teacher_capture._replace(route_captures=changed_routes)
            changed_generated = dataclasses.replace(generated, teacher_capture=changed_capture)
            changed_teacher = replay_generated_routes(model, changed_generated)
            assert np.max(np.abs(changed_teacher - teacher)) > 0.0

        if architecture == "mixed":
            # Same initialized parameter bytes, same actual generated inputs and
            # routes: removing the short layer's window must change outputs only
            # after position 3. This crosses a real window boundary, not just a
            # traced short-layer branch whose window exceeds the sequence.
            full_window = SnowballLMHeadModel.init(
                hax.Axis("vocab", 23), dataclasses.replace(config, sliding_window=8), key=jax.random.key(41)
            )
            for original, alternate in zip(
                jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)),
                jax.tree_util.tree_leaves(eqx.filter(full_window, eqx.is_array)),
                strict=True,
            ):
                np.testing.assert_array_equal(np.asarray(original), np.asarray(alternate))
            without_window = replay_generated_routes(full_window, generated)
            np.testing.assert_array_equal(teacher[:, :4].view(np.uint32), without_window[:, :4].view(np.uint32))
            assert np.max(np.abs(teacher[:, 4:] - without_window[:, 4:])) > 0.0


@pytest.mark.parametrize("architecture", ["nope", "mixed"])
@pytest.mark.timeout(180)
def test_rollout_replay_cli_runs_on_four_explicit_cpu_devices_and_retains_failure(tmp_path, architecture):
    output = tmp_path / "four-device-generated-replay.json"
    # Keep the lexical venv interpreter path. Resolving its symlink can select
    # bare system Python, which does not contain the installed JAX dependencies.
    interpreter = os.path.abspath(sys.executable)
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_NUM_CPU_DEVICES": "4",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONOPTIMIZE": "0",
        "PYTHONPATH": os.pathsep.join(
            (
                str(_BENCHMARK.parents[1] / "src"),
                str(_BENCHMARK.parents[2] / "haliax" / "src"),
            )
        ),
    }
    # Omitting the selection retains the existing NoPE control command.
    architecture_args = ["--architecture", architecture] if architecture == "mixed" else []
    completed = subprocess.run(
        [
            interpreter,
            "-B",
            str(_BENCHMARK),
            "--backend",
            "cpu",
            *architecture_args,
            "--source-sha",
            _BASE,
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "success"
    assert report["metadata"]["architecture"] == architecture
    schedule = report["metadata"]["layer_schedule"]
    if architecture == "mixed":
        assert [(layer["attention"], layer["window"], layer["rotary_dim"]) for layer in schedule] == [
            ("half_rope_windowed_causal", 4, 4),
            ("nope_full_causal", None, 0),
        ]
    else:
        assert [(layer["attention"], layer["window"], layer["rotary_dim"]) for layer in schedule] == [
            ("nope_full_causal", None, 0)
        ]
    assert report["metadata"]["local_device_count"] == 4
    assert report["metadata"]["mesh"]["data"] == 4
    assert all("Explicit" in axis for axis in report["metadata"]["mesh_axis_types"])
    assert report["native_qualification"]["status"] == "unexecuted"
    assert report["parameter_sha256_before"] == report["parameter_sha256_after"]
    assert len(report["runs"]) == 2
    for row in report["runs"]:
        assert row["predictions"]["shape"] == [4, 5, 23]
        assert row["all_output_rows"]["shape"] == [4, 8, 23]
        assert row["extra_unsampled_next_row"]["shape"] == [4, 23]
        assert row["capture_provenance"]["serving_calls"] == 6
        assert row["capture_provenance"]["captured_token_rows_per_layer"] == 32
        assert len(row["capture_provenance"]["per_layer_route_sha256"]) == len(schedule)
        assert row["parameter_sha256"] == report["parameter_sha256_before"]
        assert row["parameter_sha256_after_replay"] == report["parameter_sha256_before"]
        assert row["predictions"]["finite"]
        assert row["predictions"]["sampled_token_log_probs_fp32"]["finite"]
        assert row["predictions"]["max_abs_error"] < 1e-4
        mismatches = row["predictions"]["bit_mismatch_count"]
        assert row["predictions"]["bitwise_fp32"] == (mismatches == 0)
    assert report["prediction_parity"] == (
        "exact" if all(row["predictions"]["bitwise_fp32"] for row in report["runs"]) else "different"
    )
    assert report["cross_page_maps"]["sampled_ids_equal"]
    assert report["cross_page_maps"]["routes_equal"]
    assert report["cross_page_maps"]["serving_logits"]["bitwise_fp32"]
    assert report["cross_page_maps"]["replay_logits"]["bitwise_fp32"]

    failed_output = tmp_path / "unavailable-native.json"
    failed = subprocess.run(
        [interpreter, "-B", str(_BENCHMARK), "--backend", "tpu", "--output", str(failed_output)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert failed.returncode != 0
    failure = json.loads(failed_output.read_text())
    assert failure["status"] == "failed"
    assert failure["native_qualification"]["status"] == "unexecuted"
    assert failure["runs"] == []
    assert failure["error_type"] == "RuntimeError"

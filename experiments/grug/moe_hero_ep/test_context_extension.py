# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math

import draccus
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.attention import RotaryConfig
from levanter.layers.rotary import YarnRotaryEmbeddingsConfig

from experiments.grug.moe_hero_ep.context_extension import ContextExtensionConfig, extended_model, prepare_extension
from experiments.grug.moe_hero_ep.model import GrugModelConfig, _apply_rotary_embedding_fused
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.yarn import yarn_frequencies


def test_yarn_boundary_beyond_frequency_vector_preserves_reference_interpolation():
    # The original YaRN reference clamps against dim - 1, not dim // 2 - 1:
    # https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    # These parameters give a correction interval [0, 7] over four frequencies.
    # With factor 4 the reference ramp is [0, 1/7, 2/7, 3/7], so the final
    # frequency remains partly interpolated rather than reaching 0.125 / 4.
    config = YarnRotaryEmbeddingsConfig(
        theta=16,
        factor=4,
        original_max_position_embeddings=64,
        beta_fast=64 / (2 * math.pi),
        beta_slow=64 / (2 * math.pi * 16**2),
    )
    frequencies, _ = yarn_frequencies(8, config)
    np.testing.assert_allclose(frequencies, [1, 25 / 56, 11 / 56, 19 / 224], rtol=1e-7, atol=0)


def test_yarn_factor_one_preserves_interleaved_rotation_and_global_rope_free_layers():
    q = jax.random.normal(jax.random.key(1), (1, 16, 2, 128))
    k = jax.random.normal(jax.random.key(2), q.shape)
    arguments = dict(seq_len=16, head_dim=128, rotary_dim=64, rope=RotaryConfig(), disable_rope=False)
    original = _apply_rotary_embedding_fused(q, k, **arguments)
    yarn = YarnRotaryEmbeddingsConfig(factor=1, original_max_position_embeddings=16384)
    identity = _apply_rotary_embedding_fused(q, k, yarn=yarn, **arguments)
    for actual, expected in zip(identity, original, strict=True):
        np.testing.assert_array_equal(actual, expected)
    arguments["disable_rope"] = jnp.asarray(True)
    scaled = _apply_rotary_embedding_fused(
        q, k, yarn=YarnRotaryEmbeddingsConfig(factor=4, original_max_position_embeddings=16384), **arguments
    )
    for actual, expected in zip(scaled, (q, k), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_yarn_65k_positions_match_independent_complex_rotation_and_preserve_unrotated_half():
    # The independent reference expresses adjacent pairs as complex numbers.
    width = 128
    values = np.linspace(-1, 1, width, dtype=np.float32)
    q = jnp.broadcast_to(jnp.asarray(values), (1, 65536, 1, width))
    config = YarnRotaryEmbeddingsConfig(factor=4, original_max_position_embeddings=16384)
    rotated, _ = _apply_rotary_embedding_fused(
        q, q, seq_len=65536, head_dim=width, rotary_dim=64, rope=RotaryConfig(), disable_rope=False, yarn=config
    )
    positions = np.asarray([0, 16383, 32767, 65535])
    dimensions = np.arange(32)
    # For theta=10000, original length=16384 and rotary width=64,
    # the YaRN correction boundaries are floor(15.29)=15 and ceil(27.33)=28.
    low = math.floor(64 * math.log(16384 / (32 * 2 * math.pi)) / (2 * math.log(10000)))
    high = math.ceil(64 * math.log(16384 / (2 * math.pi)) / (2 * math.log(10000)))
    ramp = np.clip((dimensions - low) / (high - low), 0, 1)
    frequencies = (10000 ** (-dimensions / 32)) * (1 - ramp + ramp / 4)
    pairs = values[:64:2] + 1j * values[1:64:2]
    expected_pairs = pairs[None, :] * np.exp(1j * positions[:, None] * frequencies) * (1 + 0.1 * math.log(4))
    expected = np.empty((len(positions), 64))
    expected[:, ::2], expected[:, 1::2] = expected_pairs.real, expected_pairs.imag
    actual = np.asarray(rotated)[0, positions, 0]
    # At 65K, float32 phase multiplication has up to a few milliradians of error.
    np.testing.assert_allclose(actual[:, :64], expected, rtol=0, atol=0.004)
    np.testing.assert_array_equal(actual[:, 64:], np.broadcast_to(values[64:], (len(positions), 64)))
    assert np.max(np.abs(actual[-1, :64] - values[:64])) > 0.5


def test_extension_manifest_and_hf_roundtrip_preserve_native_rotary_behavior(tmp_path):
    checkpoint = tmp_path / "step-10"
    checkpoint.mkdir()
    source = GrugModelConfig(vocab_size=128256, max_seq_len=16384, rope_fused=True)
    prepared = tmp_path / "candidate-C"
    prepared.mkdir()
    provenance = {"candidate": "C", "model": draccus.encode(source), "model_digest": digest(draccus.encode(source))}
    prepared_metadata = {"step": 0, "is_temporary": False, "averaging_digest": digest(provenance)}
    (prepared / "metadata.json").write_text(json.dumps(prepared_metadata))
    (prepared / "averaging.json").write_text(json.dumps(provenance))
    metadata = {
        "step": 10,
        "is_temporary": False,
        "merge_base": "BF16-rounded candidate C",
        "hero_sft": {
            "candidate": "D",
            "base": {"checkpoint": str(prepared), "metadata_digest": digest(prepared_metadata)},
        },
    }
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    request = ContextExtensionConfig(str(checkpoint), digest(metadata), source, "a" * 40, mscale=1.5)
    destination = tmp_path / "context-candidate.json"
    manifest = prepare_extension(request, destination)
    restored = GrugModelConfig.from_hf_config(extended_model(request).to_hf_config(source.vocab_size))
    q = jax.random.normal(jax.random.key(3), (1, 16, 1, 128))
    arguments = dict(seq_len=16, head_dim=128, rotary_dim=64, disable_rope=False)
    native, _ = _apply_rotary_embedding_fused(q, q, rope=source.rope, yarn=extended_model(request).yarn, **arguments)
    roundtrip, _ = _apply_rotary_embedding_fused(q, q, rope=restored.rope, yarn=restored.yarn, **arguments)
    np.testing.assert_array_equal(native, roundtrip)
    saved = json.loads(destination.read_text())
    assert saved["source_checkpoint"] == str(checkpoint)
    assert saved["source_metadata_digest"] == digest(metadata)
    assert manifest["validation_status"] == "pending"
    assert sorted(path.name for path in checkpoint.iterdir()) == ["metadata.json"]
    # The importer must not silently change a serving artifact's declared phase/amplitude.
    hf_config = extended_model(request).to_hf_config(source.vocab_size)
    hf_config.rope_parameters["attention_factor"] = 1.0
    with pytest.raises(ValueError, match="attention_factor"):
        GrugModelConfig.from_hf_config(hf_config)

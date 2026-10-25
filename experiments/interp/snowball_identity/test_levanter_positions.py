# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the Levanter positions writer on a tiny random Levanter Snowball."""

import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
import torch
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

from experiments.interp.snowball_identity.levanter_positions import TOP_K, positions_records, write_positions
from experiments.interp.snowball_identity.logit_check import stored_sequences
from experiments.interp.snowball_identity.sequences import Sequence

CONFIG = SnowballConfig(
    vocab_size=64,
    hidden_dim=32,
    intermediate_dim=64,
    shared_expert_intermediate_dim=48,
    num_experts=8,
    num_experts_per_token=2,
    num_layers=5,
    num_heads=4,
    num_kv_heads=2,
    head_dim=12,
    max_seq_len=32,
    sliding_window=4,
    qk_mult=1.37,
    initializer_std=0.02,
    attention_implementation="reference",
)
SEQUENCES = [
    Sequence("check|a", "check", (3, 17, 5, 60, 2, 9, 41, 8, 11, 30)),
    Sequence("long|x", "long", tuple(range(1, 21))),
    Sequence("contrastive|r|no_system|0", "contrastive", (7, 7, 12, 50, 1)),
]


@pytest.fixture(scope="module")
def mesh_and_model():
    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh):
        model = SnowballLMHeadModel.init(Axis("vocab", CONFIG.vocab_size), CONFIG, key=jax.random.key(3))
    return mesh, model


def test_records_match_the_model_forward(mesh_and_model):
    mesh, model = mesh_and_model
    batch = jax.device_count()
    with jax.set_mesh(mesh):
        records = positions_records(model, SEQUENCES, batch=1, pad_multiple=1)
        for sequence in SEQUENCES:
            tokens = jnp.broadcast_to(jnp.asarray(sequence.tokens, dtype=jnp.int32), (batch, len(sequence.tokens)))
            ids = hax.named(tokens, (Axis("batch", batch), model.Pos.resize(len(sequence.tokens))))
            expected = np.asarray(jax.nn.log_softmax(model(ids).array[0].astype(jnp.float32), axis=-1))
            record = records[sequence.id]
            assert record["tokens"] == list(sequence.tokens)
            assert torch.isnan(record["actual"][0]) and (record["top_ids"][0] == -1).all()
            assert record["top_ids"].shape == (len(sequence.tokens), TOP_K)
            # Entry i is the distribution over token i given tokens 0..i-1.
            for i in range(1, len(sequence.tokens)):
                np.testing.assert_allclose(record["actual"][i], expected[i - 1, sequence.tokens[i]], atol=1e-4)
                assert int(record["top_ids"][i, 0]) == int(expected[i - 1].argmax())


def test_write_positions_uses_the_compute_view(mesh_and_model, tmp_path):
    mesh, model = mesh_and_model
    with jax.set_mesh(mesh):
        write_positions(
            model, jmp.get_policy("p=f32,c=bfloat16"), SEQUENCES, str(tmp_path / "positions.pt"), pad_multiple=4
        )
        exact = positions_records(model, SEQUENCES, pad_multiple=4)
    stored = torch.load(tmp_path / "positions.pt")
    assert [s.tokens for s in stored_sequences(str(tmp_path / "positions.pt"))] == [s.tokens for s in SEQUENCES]
    # The BF16 compute view scores close to, but not exactly like, the FP32 model.
    diff = (stored["check|a"]["actual"][1:] - exact["check|a"]["actual"][1:]).abs()
    assert 0 < float(diff.max()) < 0.5


def test_batching_and_right_padding_keep_real_positions(mesh_and_model):
    mesh, model = mesh_and_model
    with jax.set_mesh(mesh):
        single = positions_records(model, SEQUENCES, batch=1, pad_multiple=1)
        # Two rows per step (the last step gets a filler copy) and padding to a multiple of 8.
        batched = positions_records(model, SEQUENCES, batch=2, pad_multiple=8, pad_id=63)
    assert list(batched) == [s.id for s in SEQUENCES]
    for sequence in SEQUENCES:
        a, b = single[sequence.id], batched[sequence.id]
        assert a["tokens"] == b["tokens"] and len(b["actual"]) == len(sequence.tokens)
        torch.testing.assert_close(a["actual"], b["actual"], atol=1e-5, rtol=0, equal_nan=True)
        torch.testing.assert_close(a["top_lp"], b["top_lp"], atol=1e-5, rtol=0, equal_nan=True)
        assert torch.equal(a["top_ids"][:, :5], b["top_ids"][:, :5])

# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding

from levanter.models.llama import LlamaEmbedding
from levanter.models.llama_native import NativeLlamaConfig, NativeLlamaModel, NativeLlamaSharding


@pytest.mark.multidevice
def test_native_bf16_embedding_gradient_matches_named_for_repeated_tokens():
    if jax.device_count() < 4:
        pytest.skip("Requires two data shards and two vocabulary shards")

    devices = np.asarray(jax.devices()[:4]).reshape(2, 2)
    explicit = Mesh(devices, ("data", "model"), axis_types=(AxisType.Explicit,) * 2)
    auto = Mesh(devices, ("data", "model"), axis_types=(AxisType.Auto,) * 2)
    layout = NativeLlamaSharding(batch_axes=("data",), fsdp_axis="data", model_axis="model")
    vocab, embed = hax.Axis("vocab", 128), hax.Axis("embed", 32)
    batch, position = hax.Axis("batch", 4), hax.Axis("position", 16)
    token_ids = (np.arange(batch.size * position.size, dtype=np.int32) % 11).reshape(batch.size, position.size)
    rng = np.random.default_rng(146)
    cotangent = rng.standard_normal((batch.size, position.size, embed.size), dtype=np.float32).astype(jnp.bfloat16)

    with jax.set_mesh(explicit):
        config = NativeLlamaConfig(
            vocab_size=vocab.size,
            hidden_dim=embed.size,
            intermediate_dim=8,
            num_layers=1,
            num_heads=4,
            num_kv_heads=4,
            head_dim=2,
            max_seq_len=position.size,
            tie_word_embeddings=True,
        )
        model = jax.jit(lambda key: NativeLlamaModel.init(config, key=key, sharding=layout))(jax.random.PRNGKey(145))
        weights = model.token_embeddings
        tokens = jax.device_put(token_ids, NamedSharding(explicit, layout.tokens))
        incoming = jax.device_put(cotangent, NamedSharding(explicit, layout.residual))

        def native_vjp(weight, tokens, incoming):
            def forward(master):
                updated = eqx.tree_at(lambda m: m.token_embeddings, model, master.astype(jnp.bfloat16))
                return updated.embed_tokens(tokens, sharding=layout)

            output, pullback = jax.vjp(forward, weight)
            return output, pullback(incoming)[0]

        native, native_gradient = jax.jit(native_vjp, out_shardings=(layout.residual, layout.column_storage))(
            weights, tokens, incoming
        )

    with jax.set_mesh(auto):

        def named_vjp(weight, tokens, incoming):
            def forward(master):
                module = LlamaEmbedding(hnn.Embedding(master.astype(jnp.bfloat16), vocab, embed))
                return module.embed(tokens)

            output, pullback = jax.vjp(forward, weight)
            return output.rename({"embed": "activation_embed"}), pullback(incoming)[0]

        named, named_gradient = hax.named_jit(
            named_vjp,
            axis_resources={"vocab": "model", "batch": "data"},
            out_axis_resources={"vocab": "model", "batch": "data", "embed": "data"},
        )(
            hax.named(jax.device_put(weights, NamedSharding(auto, layout.column_storage)), (vocab, embed)),
            hax.named(jax.device_put(token_ids, NamedSharding(auto, layout.tokens)), (batch, position)),
            hax.named(jax.device_put(cotangent, NamedSharding(auto, layout.residual)), (batch, position, embed)),
        )

    np.testing.assert_array_equal(native, named.array)
    np.testing.assert_array_equal(native_gradient, named_gradient.array)

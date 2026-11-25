# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import pytest

from levanter.layers.kv_cache import KvPageCache


class _FixedTokenModel(eqx.Module):
    vocab_size: int = eqx.field(static=True)
    token_id: int = eqx.field(static=True)

    @property
    def Vocab(self) -> hax.Axis:
        return hax.Axis("vocab", self.vocab_size)

    def initial_cache(self, spec, *, dtype):
        return KvPageCache.init(spec, hax.Axis("kv_head", jax.device_count()), hax.Axis("embed", 1), dtype=dtype)

    def decode(self, input_ids, kv_cache, batch_info, pos_ids):
        logits = hax.nn.one_hot(self.token_id, self.Vocab, dtype=jnp.float32)
        return logits.broadcast_axis(input_ids.resolve_axis("position")), kv_cache


@pytest.fixture(scope="module")
def fixed_token_model():
    return _FixedTokenModel

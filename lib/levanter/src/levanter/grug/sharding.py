# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import P
from jax.sharding import reshard

# convenience shorthand for batch sharding.
# if this were Haliax, we'd say {"batch": ("data",)}
Pbatch = P(("data",))
Pembed_vocab = P("model", Pbatch[0])
Plm_head = P(Pbatch[0], "model")
Plogits = P(Pbatch[0], None, "model")


def embed_vocab_pspec(batch_axis) -> P:
    return P("model", batch_axis)


def lm_head_pspec(batch_axis) -> P:
    return P(batch_axis, "model")


def logits_pspec(batch_axis) -> P:
    return P(batch_axis, None, "model")


def unshard(x: jax.Array) -> jax.Array:
    return reshard(x, P(None))

# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import P
from jax.sharding import reshard

# convenience shorthand for batch sharding.
# if this were Haliax, we'd say {"batch": ("replica_dcn", "replica", "data")}
Pbatch = P(
    ("replica_dcn", "replica", "data"),
)
Pvocab = P(None, None)


def unshard(x: jax.Array) -> jax.Array:
    return reshard(x, P((None,)))

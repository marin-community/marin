# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import P
from jax.sharding import reshard

# convenience shorthand for batch sharding.
# if this were Haliax, we'd say {"batch": ("replica_dcn", "replica", "data")}
#
# Note: Levanter's default mesh includes ("replica_dcn", "replica", "data") as the batch/data-parallel axes
# (even when some are size-1). Grug uses explicit shardings, so this should match the mesh to avoid
# ShardingTypeErrors when combining batch-shaped arrays (e.g., multiplying loss by loss weights).
Pbatch = P(("replica_dcn", "replica", "data"))
Pvocab = P(None, None)


def unshard(x: jax.Array) -> jax.Array:
    return reshard(x, P((None,)))

# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""One CPU participant in the fixed-CE process-coordination regression."""

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from levanter.kernels.pallas.fused_cross_entropy_loss.api import (
    FixedCEPlan,
    FixedCESignature,
    fused_cross_entropy_loss_and_logsumexp_penalty,
    prepare_fixed_ce_plan,
)
from levanter.utils.jax_utils import multihost_allgather_sync


def main() -> None:
    rank, address, destination = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    jax.distributed.initialize(
        coordinator_address=address,
        num_processes=4,
        process_id=rank,
        local_device_ids=[0],
        initialization_timeout=30,
    )
    try:
        profile = {"shape": [4, 4, 16], "dtype": "bfloat16", "implementation": "xla"}
        plan = FixedCEPlan(
            (FixedCESignature((4, 4), (4,), (4, 16), "bfloat16", "int32", "bfloat16"),),
            block_size=8,
            logits_dtype="bfloat16",
            execution_profile_sha256=hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
        )

        def prepare(step, *, local_error=None):
            return prepare_fixed_ce_plan(
                replace(plan),
                execution_profile=profile,
                command={"step": step},
                process_ids=(0, 1, 2, 3),
                timeout_seconds=10,
                local_error=local_error,
            )

        def loss(x, labels, w, fixed_plan):
            return fused_cross_entropy_loss_and_logsumexp_penalty(
                x,
                labels,
                w,
                implementation="xla",
                reduction=None,
                block_size=8,
                logits_dtype=jnp.bfloat16,
                fixed_plan=fixed_plan,
            ).sum()

        mesh = Mesh(np.asarray(jax.devices()), ("data",))
        with jax.set_mesh(mesh):
            replicated = NamedSharding(mesh, PartitionSpec())
            x = jax.device_put(np.arange(16, dtype=np.float32).reshape(4, 4) / 16, replicated).astype(jnp.bfloat16)
            labels = jax.device_put(np.arange(4, dtype=np.int32), replicated)
            w = jax.device_put(np.arange(64, dtype=np.float32).reshape(4, 16) / 64, replicated).astype(jnp.bfloat16)
            evaluate = jax.jit(jax.value_and_grad(loss, argnums=(0, 2)), static_argnums=3)
            before = jax.block_until_ready(evaluate(x, labels, w, prepare(0)))
            if rank != 3:
                jax.clear_caches()
            # Rank 3 executes its cached graph while peers must trace again.
            after = jax.block_until_ready(evaluate(x, labels, w, prepare(1)))
            for expected, actual in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            rejected = False
            try:
                prepare(2, local_error="invalid execution profile" if rank == 3 else None)
            except ValueError:
                rejected = True
            assert rejected, "Every participant must reject a peer's invalid profile before entering JIT"
            (destination / f"rank{rank}.json").write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "cached_execution_exact": True,
                        "peer_profile_error_rejected": rejected,
                        "loss": float(after[0]),
                    }
                )
            )
        multihost_allgather_sync({"finished": True}, timeout=10)
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Align the routed-expert axis between two checkpoints via their routers.

Mochi checkpoints are saved under run-specific learned expert placements, so the
same logical expert can sit at different indices in two checkpoints. The router
weight [hidden, num_experts] identifies each expert by its column, which moves
only marginally during training: matching columns by cosine similarity recovers
the permutation exactly (verified: best-match similarity 1.000 vs 0.083 for the
identity order between the hero base and the exp3-v5 final).
"""

from __future__ import annotations

import numpy as np

ROUTER_TENSOR = "stacked_blocks/stacked/mlp/router"


def router_expert_permutation(side_a, side_b, layer: int = 0, min_similarity: float = 0.95) -> dict:
    """Return {"perm": index array (base expert -> side-b expert), "mean_similarity": float}.

    ``side_a``/``side_b`` are reader-side objects exposing ``read(SliceRequest)``.
    """
    from measure import SliceRequest  # noqa: PLC0415 - measure <-> expert_perm cycle

    req = SliceRequest(ROUTER_TENSOR, "router", "router", layer, (layer,), ())
    col_a = np.asarray(side_a.read(req))[0]
    col_b = np.asarray(side_b.read(req))[0]
    if col_a.shape != col_b.shape:
        raise ValueError(f"router shapes differ: {col_a.shape} vs {col_b.shape}")
    na = col_a / (np.linalg.norm(col_a, axis=0, keepdims=True) + 1e-12)
    nb = col_b / (np.linalg.norm(col_b, axis=0, keepdims=True) + 1e-12)
    sim = na.T @ nb
    perm = sim.argmax(axis=1)
    mean_similarity = float(sim.max(axis=1).mean())
    identity = bool((perm == np.arange(len(perm))).all())
    if mean_similarity < min_similarity:
        raise ValueError(f"router column matching failed (mean similarity {mean_similarity:.4f})")
    return {"perm": perm, "mean_similarity": mean_similarity, "identity": identity}

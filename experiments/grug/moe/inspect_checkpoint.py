# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report which leaves of a levanter/orbax checkpoint are non-finite (NaN/Inf).

Used to root-cause a NaN: distinguish corrupted model PARAMS (optimizer poisoned them) from
finite params with a NaN only in eval (a forward/precision issue). Run IN-REGION next to the
checkpoint bucket to avoid cross-region egress.

    CKPT=gs://.../checkpoints/step-3748 python -m experiments.grug.moe.inspect_checkpoint
"""

import os

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

_CKPT = os.environ["CKPT"]


def main():
    print(f"inspecting {_CKPT}")
    ckptr = ocp.PyTreeCheckpointer()
    tree = ckptr.restore(_CKPT)
    flat, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda x: hasattr(x, "shape"))
    total = bad = 0
    bad_leaves = []
    for kp, arr in flat:
        if not hasattr(arr, "shape"):
            continue
        total += 1
        finite = bool(jnp.all(jnp.isfinite(jnp.asarray(arr))))
        if not finite:
            bad += 1
            nan_frac = float(jnp.mean(~jnp.isfinite(jnp.asarray(arr))))
            bad_leaves.append((jax.tree_util.keystr(kp), tuple(arr.shape), nan_frac))
    print(f"leaves={total}  non-finite={bad}")
    for path, shape, frac in bad_leaves:
        print(f"  NON-FINITE {path}  shape={shape}  bad_frac={frac:.3e}")
    if bad == 0:
        print("ALL LEAVES FINITE -> params + opt-state are clean (NaN was eval-forward only)")


if __name__ == "__main__":
    main()

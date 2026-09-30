# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Localize gradient differences of the expert-side router gradient under remat.

Runs the rematted layer scan of `scan_compare.py` under several remat settings and compares each
candidate configuration against main under the same setting. Each configuration also runs twice
to test repeatability.

Usage (GB200x4): python autoresearch/loop-260930-mfu30/b/scan_diagnose.py
"""

import json
import sys

import jax
import scan_compare as sc

CONFIGS = {
    # name: (variant, remat policy or "none" for no checkpoint)
    "control_remat": ("control", None),
    "candidate_remat_save": ("candidate", jax.checkpoint_policies.save_only_these_names(sc.MOE_OUTPUT)),
    "candidate_remat_nosave": ("candidate", None),
    "control_noremat": ("control", "none"),
    "candidate_noremat": ("candidate", "none"),
}
PAIRS = {
    "candidate_remat_save": "control_remat",
    "candidate_remat_nosave": "control_remat",
    "candidate_noremat": "control_noremat",
}


def main():
    sc.TOKENS_PER_SHARD, sc.HIDDEN, sc.INTER = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[2])
    mesh = sc._mesh()
    inp = sc._inputs(mesh)
    results = {}
    for name, (variant, policy) in CONFIGS.items():
        local_fn, _policy, backward = sc.VARIANTS[variant]
        exe, args = sc._build(mesh, inp, local_fn, policy, backward)
        first = exe(*args)
        second = exe(*args)
        results[name] = first
        repeat = sc._compare(first, second)
        print(json.dumps(dict(config=name, repeatable={k: v["equal"] for k, v in repeat.items()})), flush=True)
    for name, reference in PAIRS.items():
        print(
            json.dumps(dict(config=name, vs=reference, compare=sc._compare(results[reference], results[name]))),
            flush=True,
        )


if __name__ == "__main__":
    main()

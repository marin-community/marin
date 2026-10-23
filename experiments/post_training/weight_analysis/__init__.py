# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weight analysis and grafting over WP2's fp32 exports (CL15 grafting, CL16).

- :mod:`.components`: Grug component groups, graft groups, metric keys.
- :mod:`.checkpoint_io`: byte-range sharded safetensors reads and streaming writes over fsspec.
- :mod:`.delta_stats`: delta streams, norms, changed fractions, conflict, subspace overlap.
- :mod:`.expert_metrics`: expert Gini, additivity, seed ceilings.
- :mod:`.graft`: the ``B + Δ·1[S]`` builder.
"""

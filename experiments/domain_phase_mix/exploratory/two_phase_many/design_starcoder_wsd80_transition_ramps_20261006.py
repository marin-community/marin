# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build matched-total ramp staircases for the 7.41B two-phase winners and verify realized StarCoder counts."""

import json
import math
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from levanter.data.mixture import MixtureDataset

from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base

BATCH, BLOCK = base.BATCH_SIZE, base.MIXTURE_BLOCK_SIZE
STEPS_PER_BLOCK = BLOCK // BATCH  # 16
TOTAL, BOUNDARY = 28260, 22608
names = [f"nemotron_{i}" for i in range(len(base.NEMOTRON_TOKEN_COUNTS))] + ["starcoder"]
tot_nem = sum(base.NEMOTRON_TOKEN_COUNTS.values())


def leaf(p):
    w = [(1 - p) * c / tot_nem for c in base.NEMOTRON_TOKEN_COUNTS.values()] + [p]
    return dict(zip(names, w, strict=False))


def realized_starcoder(stages, total_steps=TOTAL, seed=20260821):
    weights = [(s * BATCH, leaf(p)) for s, p in stages]
    key, _ = jax.random.split(jax.random.PRNGKey(seed))
    md = MixtureDataset({n: object() for n in names}, weights, block_size=BLOCK, key=key)
    sc = list(md.dataset_index).index("starcoder")
    counts = 0
    per_stage = []
    starts = [s for s, _ in stages] + [total_steps]
    for k in range(len(stages)):
        nblocks, rem = divmod((starts[k + 1] - starts[k]) * BATCH, BLOCK)
        c = nblocks * int(md._counts_per_block_per_stage[k][sc])
        if rem:
            blk = md._get_block(starts[k] * BATCH // BLOCK + nblocks)
            c += int(np.count_nonzero((blk[:rem] >> 16) == sc))
        per_stage.append(c)
        counts += c
    return counts, per_stage, md


def abrupt(p0, p1):
    return [(0, p0), (BOUNDARY, p1)]


def ramp(p0, p1, width, target_total):
    """Linear ramp centred on the boundary as a staircase of 16-step stages; nudge ramp-stage weights
    by 1/BLOCK (one sequence per block) until the realized StarCoder sequence total equals target_total."""
    half = width // 2
    start, end = BOUNDARY - half, BOUNDARY + half
    assert start % STEPS_PER_BLOCK == 0 and end % STEPS_PER_BLOCK == 0, (start, end)
    stages = [(0, p0)]
    for s in range(start, end, STEPS_PER_BLOCK):
        mid = s + STEPS_PER_BLOCK / 2
        stages.append((s, p0 + (p1 - p0) * (mid - start) / (end - start)))
    stages.append((end, p1))
    linear = [p for _, p in stages]
    total, per_stage, _ = realized_starcoder(stages)
    residual = target_total - total
    n_ramp = len(stages) - 2
    i = 0
    while residual != 0:
        k = 1 + (i % n_ramp)
        s, p = stages[k]
        stages[k] = (s, p + math.copysign(1.0 / BLOCK, residual))
        total, per_stage, _ = realized_starcoder(stages)
        residual = target_total - total
        i += 1
        if i > 20000:
            raise RuntimeError("ramp matching did not converge")
    nudge = max(abs(p - q) for (_, p), q in zip(stages, linear, strict=False))
    return stages, total, per_stage, nudge


for block, p0, p1 in [("m100", 0.01008, 0.85968), ("m200", 0.0, 0.5), ("m400", 0.0, 0.2814080848021789)]:
    tot_ab, per_ab, _ = realized_starcoder(abrupt(p0, p1))
    print(f"{block}: abrupt {p0}->{p1} realized StarCoder sequences {tot_ab} (phase0 {per_ab[0]}, phase1 {per_ab[1]})")
    for width in (512, 2048, 5632):
        stages, tot, per, nudge = ramp(p0, p1, width, tot_ab)
        pre = sum(c for (s, _), c in zip(stages, per, strict=False) if s < BOUNDARY)
        print(
            f"  ramp width {width}: {len(stages)} stages [{stages[1][0]}, {stages[-1][0]}], "
            f"realized {tot} (match={tot == tot_ab}), pre-boundary StarCoder {pre} vs abrupt {per_ab[0]}, "
            f"max weight nudge {nudge:.5f}, weights clipped? {any(p < 0 or p > 1 for _, p in stages)}"
        )

# export the frozen stage lists

design = {}
for block, p0, p1 in [("m100", 0.01008, 0.85968), ("m200", 0.0, 0.5), ("m400", 0.0, 0.2814080848021789)]:
    tot_ab, per_ab, _ = realized_starcoder(abrupt(p0, p1))
    design[block] = {
        "p0": p0,
        "p1": p1,
        "abrupt_stages": abrupt(p0, p1),
        "abrupt_realized_starcoder_sequences": tot_ab,
        "abrupt_phase_sequences": per_ab,
        "ramps": {},
    }
    for width in (512, 2048, 5632):
        stages, tot, per, nudge = ramp(p0, p1, width, tot_ab)
        design[block]["ramps"][str(width)] = {
            "stages": stages,
            "realized_starcoder_sequences": tot,
            "max_weight_nudge": nudge,
            "pre_boundary_starcoder_sequences": sum(c for (s, _), c in zip(stages, per, strict=False) if s < BOUNDARY),
        }
with open(sys.argv[1], "w") as handle:
    json.dump(design, handle, indent=1)
print("wrote", sys.argv[1])

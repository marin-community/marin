# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Identify each ragged all-to-all instruction of a profiled train step by its place in a layer.

Reads rows.pkl (tfop_dump.py) and opnames.pkl (hlo_opnames.py). Per ragged all-to-all HLO instruction:
instances per step, median duration, exposed time (not under any compute kernel), the step fraction
where its instances sit (forward, recompute or backward), and its order among the ragged all-to-alls of
one layer. For one middle forward layer and one middle backward layer it prints the sequence of ragged
all-to-alls with the compute scopes that run under each and between consecutive ones.

Usage: python ragged_order.py <rows.pkl> <opnames.pkl>
"""

import collections
import pickle
import re
import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from anatomy_lib import intersect, length, merge, phase, scope

PS = 1e-12
RAGGED = re.compile(r"RaggedAllToAll", re.I)
COLLECTIVE = re.compile(r"nccl|RaggedAllToAll", re.I)
COPY = re.compile(r"memcpy|memset", re.I)


def _label(row, ops):
    rec = ops.get(row[5])
    op = (rec[1] if rec else "") or ""
    return f"{scope(op) if op else 'unmapped'}/{phase(op)}"


def main():
    rows_path, ops_path = sys.argv[1], sys.argv[2]
    with open(rows_path, "rb") as fh:
        data = pickle.load(fh)
    with open(ops_path, "rb") as fh:
        ops = pickle.load(fh)
    rows = sorted((r for r in data["rows"] if r[6] == "jit_train_step"), key=lambda r: r[2])
    launches = data["launches"]
    end = max(r[3] for r in rows)
    steps = list(zip(launches, [*launches[1:], end]))
    nstep = len(steps)
    compute = [r for r in rows if not COLLECTIVE.search(r[1]) and not COPY.search(r[1])]
    comp_iv = merge([(r[2], r[3]) for r in compute])
    ragged = [r for r in rows if RAGGED.search(r[1])]

    def step_of(t):
        for i, (lo, hi) in enumerate(steps):
            if lo <= t < hi:
                return i, (t - lo) / (hi - lo)
        return None, None

    per = collections.defaultdict(list)
    for r in ragged:
        per[r[5]].append(r)
    table = []
    for hlo, rs in per.items():
        iv = merge([(r[2], r[3]) for r in rs])
        exposed = length(iv) - length(intersect(iv, comp_iv))
        fracs = [step_of(r[2])[1] for r in rs if step_of(r[2])[0] is not None]
        table.append(
            (
                statistics.median(fracs),
                hlo,
                len(rs) / nstep,
                statistics.median(r[3] - r[2] for r in rs) * 1e-6,
                length(iv) * PS / nstep,
                exposed * PS / nstep,
                min(fracs),
                max(fracs),
            )
        )
    table.sort()
    print("ragged all-to-all instructions, by median position in the step:")
    print("  pos(med,min-max)   n/step  med_us   busy_s  exposed_s  instruction")
    for med, hlo, n, med_us, busy, exposed, lo, hi in table:
        print(f"  {med:.3f} ({lo:.2f}-{hi:.2f})  {n:5.1f}  {med_us:7.0f}  {busy:7.3f}  {exposed:8.3f}  {hlo}")
    print(f"  total busy {sum(t[4] for t in table):.3f}s/step exposed {sum(t[5] for t in table):.3f}s/step")

    # One forward layer and one backward layer of the middle step: the ragged sequence with what runs under it.
    lo, hi = steps[nstep // 2]
    in_step = sorted((r for r in ragged if lo <= r[2] < hi), key=lambda r: r[2])
    comp_step = [r for r in compute if lo <= r[2] < hi]
    for title, frac in (("forward", 0.15), ("backward", 0.65)):
        t0 = lo + frac * (hi - lo)
        start = next(i for i, r in enumerate(in_step) if r[2] >= t0)
        print(f"\n{title} sequence from step fraction {frac}:")
        prev_end = None
        for r in in_step[start : start + 12]:
            if prev_end is not None:
                between = collections.Counter()
                for c in comp_step:
                    ov = min(c[3], r[2]) - max(c[2], prev_end)
                    if ov > 0:
                        between[_label(c, ops)] += ov
                gap = (r[2] - prev_end) * 1e-6
                top = ", ".join(f"{k} {v * 1e-6:.0f}us" for k, v in between.most_common(4))
                print(f"      between ({gap:.0f}us): {top}")
            under = collections.Counter()
            for c in comp_step:
                ov = min(c[3], r[3]) - max(c[2], r[2])
                if ov > 0:
                    under[_label(c, ops)] += ov
            covered = length(intersect([(r[2], r[3])], comp_iv)) / (r[3] - r[2])
            top = ", ".join(f"{k} {v * 1e-6:.0f}us" for k, v in under.most_common(3))
            print(f"  {r[5]:24s} {(r[3] - r[2]) * 1e-6:6.0f}us covered {covered:4.0%}  under: {top}")
            prev_end = r[3]


if __name__ == "__main__":
    main()

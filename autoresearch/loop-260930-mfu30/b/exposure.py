# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Break a profiled train step's non-compute time into idle gaps and exposed collectives.

Reads rows.pkl (tfop_dump.py) and opnames.pkl (hlo_opnames.py). For each step it separates the
head (launch to first kernel), tail (last kernel to next launch) and internal idle gaps, lists the
largest internal gaps with the kernels on either side, and groups exposed collective time by
collective instruction. For each collective instruction it also reports the spread of its
per-instance durations: instances of one instruction move the same bytes each layer, so time
above the fastest instance is mostly waiting for other ranks.

Usage: python exposure.py <rows.pkl> <opnames.pkl> [compute_stream_prefix]
"""

import collections
import pickle
import re
import statistics
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from anatomy_lib import intersect, length, merge, phase, scope

PS = 1e-12
COLLECTIVE = re.compile(r"nccl|RaggedAllToAll", re.I)
COPY = re.compile(r"memcpy|memset", re.I)


def _iv(pairs):
    return merge([(s, e) for s, e in pairs])


def _describe(row, ops):
    rec = ops.get(row[5])
    op = (rec[1] if rec else "") or ""
    return dict(
        stream=row[0][:10],
        kernel=row[1][:48],
        hlo=row[5][:36],
        scope=scope(op) if op else "unmapped",
        phase=phase(op),
    )


def main():
    rows_path, ops_path = sys.argv[1], sys.argv[2]
    compute_stream = sys.argv[3] if len(sys.argv) > 3 else "Stream #17"
    data = pickle.load(open(rows_path, "rb"))
    ops = pickle.load(open(ops_path, "rb"))
    rows = sorted((r for r in data["rows"] if r[6] == "jit_train_step"), key=lambda r: r[2])
    launches = data["launches"]
    end = max(r[3] for r in rows)
    steps = list(zip(launches, [*launches[1:], end]))
    nstep = len(steps)

    compute = [
        r for r in rows if r[0].startswith(compute_stream) and not COLLECTIVE.search(r[1]) and not COPY.search(r[1])
    ]
    other_compute = [
        r for r in rows if not r[0].startswith(compute_stream) and not COLLECTIVE.search(r[1]) and not COPY.search(r[1])
    ]
    colls = [r for r in rows if COLLECTIVE.search(r[1])]
    copies = [r for r in rows if COPY.search(r[1])]
    comp_iv = _iv([(r[2], r[3]) for r in compute + other_compute])
    coll_iv = _iv([(r[2], r[3]) for r in colls])
    copy_iv = _iv([(r[2], r[3]) for r in copies])
    busy = merge([tuple(x) for x in comp_iv + coll_iv + copy_iv])

    print(f"steps={nstep}")
    print(
        f"per step: span {sum(b - a for a, b in steps) * PS / nstep:.3f}s  compute {length(comp_iv) * PS / nstep:.3f}  "
        f"collective busy {length(coll_iv) * PS / nstep:.3f}  exposed {(length(coll_iv) - length(intersect(coll_iv, comp_iv))) * PS / nstep:.3f}  "
        f"copies exposed {(length(copy_iv) - length(intersect(copy_iv, merge([tuple(x) for x in comp_iv + coll_iv])))) * PS / nstep:.3f}  "
        f"idle {(sum(b - a for a, b in steps) - length(busy)) * PS / nstep:.3f}"
    )

    # Idle: head, tail and internal gaps per step.
    by_start = {r[2]: r for r in rows}
    head = tail = 0
    gaps = []
    for lo, hi in steps:
        inside = [iv for iv in busy if iv[1] > lo and iv[0] < hi]
        if not inside:
            continue
        head += max(0, inside[0][0] - lo)
        tail += max(0, hi - inside[-1][1])
        for (a, b), (c, _d) in zip(inside, inside[1:]):
            if c > b:
                gaps.append((c - b, b, c, (b - lo) / (hi - lo)))
    print(
        f"idle head {head * PS / nstep:.4f}s/step  tail {tail * PS / nstep:.4f}  internal {sum(g[0] for g in gaps) * PS / nstep:.4f}"
    )
    bins = collections.Counter()
    for g in gaps:
        us = g[0] * 1e-6
        key = "<5us" if us < 5 else "<50us" if us < 50 else "<500us" if us < 500 else "<5ms" if us < 5000 else ">=5ms"
        bins[key] += g[0]
    print("internal idle by gap length:", {k: round(v * PS / nstep, 4) for k, v in sorted(bins.items())})
    deciles = collections.Counter()
    for g in gaps:
        deciles[min(9, int(10 * g[3]))] += g[0]
    print("internal idle by step decile:", {k: round(v * PS / nstep, 4) for k, v in sorted(deciles.items())})

    # Context of gaps: what ended before and what starts after.
    ends = sorted(rows, key=lambda r: r[3])
    end_times = [r[3] for r in ends]
    import bisect

    context = collections.Counter()
    samples = collections.defaultdict(list)
    for g, b, c, frac in gaps:
        i = bisect.bisect_right(end_times, b) - 1
        before = _describe(ends[i], ops) if i >= 0 else {}
        after_row = by_start.get(c)
        after = _describe(after_row, ops) if after_row else {}
        key = (
            f"{before.get('scope', '?')}/{before.get('phase', '?')}/{before.get('stream', '?')}",
            f"{after.get('scope', '?')}/{after.get('phase', '?')}/{after.get('stream', '?')}:{after.get('kernel', '?')[:30]}",
        )
        context[key] += g
        samples[key].append(g)
    print("\ninternal idle by (kernel before -> kernel after), s/step, count/step, median gap us:")
    for key, v in context.most_common(25):
        print(
            f"  {v * PS / nstep:.4f}  n={len(samples[key]) / nstep:6.1f}  med={statistics.median(samples[key]) * 1e-6:8.1f}us  {key[0]} -> {key[1]}"
        )

    # Exposed collectives per instruction, with per-instance duration spread.
    per = collections.defaultdict(list)
    for r in colls:
        per[r[5]].append(r)
    table = []
    for hlo, rs in per.items():
        iv = _iv([(r[2], r[3]) for r in rs])
        exposed = length(iv) - length(intersect(iv, comp_iv))
        durations = sorted(r[3] - r[2] for r in rs)
        rec = ops.get(hlo)
        op = (rec[1] if rec else "") or ""
        table.append(
            (
                exposed,
                hlo,
                rs[0][1][:40],
                scope(op) if op else "unmapped",
                phase(op),
                len(rs) / nstep,
                length(iv),
                durations[0],
                statistics.median(durations),
                sum(d - durations[0] for d in durations),
            )
        )
    table.sort(reverse=True)
    print("\nexposed collectives per instruction (s/step): exposed busy n/step min_us med_us above_min")
    for ex, hlo, kernel, sc, ph, n, b, dmin, dmed, above in table[:30]:
        print(
            f"  {ex * PS / nstep:.4f} {b * PS / nstep:.4f} n={n:5.1f} min={dmin * 1e-6:8.1f} med={dmed * 1e-6:8.1f} "
            f"above_min={above * PS / nstep:.4f}  {hlo[:30]:30s} {kernel[:34]:34s} {sc}/{ph}"
        )
    total_ex = sum(t[0] for t in table)
    total_above = sum(t[9] for t in table)
    print(
        f"collective instructions: exposed sum {total_ex * PS / nstep:.3f}s/step, time above each instruction's fastest instance {total_above * PS / nstep:.3f}s/step"
    )


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate for reordering D's backward chunks: bitwise values, scan timing, and the scheduled transport order.

Variants (all with the SwiGLU backward in the dh GEMM epilogue, all saving the MoE output under remat):
  dme:                D + mirror parameters + E, sequential chunks (the frozen pre-reorder module).
  candidate:          the branch's live module (dme with the backward in forward chunk order).
  pipe:               dme with #9481's pipelined forward chunks (64909b24d0).
  pipe_forward_order: pipe with the backward in forward chunk order.

Two parts, selected by the first argument:
  single: one rematless MoE layer forward and backward per routing case (the routing gate's cases); every
          output and gradient of each reordered variant must be bitwise equal to its base.
  scan <a> <b>: a 3-layer rematted scan of two variants: bitwise gradients of b against a, the median step
          time with the order rotated, and per variant the scheduled order of transports, expert GEMMs
          and barriers in the backward loop body.

Usage (GB200x4): python reorder_gate.py single; python reorder_gate.py scan dme candidate
"""

import json
import re
import statistics
import sys
import time

import jax
import numpy as np
import routing_gate
import scan_compare

PAIRS = (("dme", "candidate"), ("pipe", "pipe_forward_order"), ("dme", "pipe"))
FROZEN = ("dme", "pipe", "pipe_forward_order")


def _register():
    for name in FROZEN:
        fn = scan_compare._load_frozen(f"{name}_ep_ragged_all_to_all")._moe_mlp_ep_ragged_a2a_local
        routing_gate.VARIANTS[name] = fn
        routing_gate.BACKWARDS[name] = routing_gate.FUSED_BACKWARD
        scan_compare.VARIANTS[name] = (fn, scan_compare._SAVE_OUTPUT, scan_compare._BACKWARD)


def _single():
    mesh = routing_gate._mesh()
    shards = mesh.shape["expert"]
    base = dict(hidden=3072, inter=3072, experts=6 * shards, topk=8, capacity_factor=1.15, padded=False, seed=0)
    cases = [
        dict(base, name="small-uniform", tokens_per_shard=2048, hidden=512, inter=512, routing="uniform"),
        dict(base, name="small-skewed-drops", tokens_per_shard=2048, hidden=512, inter=512, routing="skewed"),
        dict(base, name="small-padded", tokens_per_shard=2048, hidden=512, inter=512, routing="uniform", padded=True),
        dict(base, name="small-one-hot", tokens_per_shard=2048, hidden=512, inter=512, routing="one_hot"),
        dict(base, name="hero-uniform", tokens_per_shard=65536, routing="uniform"),
        dict(base, name="hero-skewed-drops", tokens_per_shard=65536, routing="skewed", padded=True),
    ]
    failures = 0
    for case in cases:
        inp = routing_gate._inputs(case, mesh)
        args = (inp["x"], inp["weights"], inp["w13"], inp["w2"], inp["ct"])
        names = ("dme", "candidate", "pipe", "pipe_forward_order")
        results = {}
        for name in names:
            exe = routing_gate._build(name, case, mesh, inp)
            results[name] = jax.device_get(exe(*args))
            del exe
        record = dict(case=case["name"], dropped=int(results["dme"][1]))
        for a, b in PAIRS:
            cmp = routing_gate._compare(results[a], results[b])
            equal = all(v["bitwise_equal"] for v in cmp.values())
            failures += not equal
            record[f"{b}_vs_{a}"] = dict(
                bitwise=equal, differing={k: v["max_rel_diff"] for k, v in cmp.items() if not v["bitwise_equal"]}
            )
        print(json.dumps(record), flush=True)
    print(json.dumps(dict(failures=failures)), flush=True)
    return failures


def _schedule(hlo_text):
    """Scheduled order in the while body with the most ragged all-to-alls: transports, GEMMs, barriers."""
    bodies, name, lines = {}, None, []
    for line in hlo_text.splitlines():
        header = re.match(r"^%?([\w.\-]+) .*\{\s*$", line)
        if header:
            name, lines = header.group(1), []
            continue
        if line.startswith("}") and name:
            bodies[name] = lines
            name = None
            continue
        if name:
            lines.append(line)
    body = max(bodies.values(), key=lambda b: sum("ragged-all-to-all-start(" in line for line in b))
    events = []
    for line in body:
        m = re.match(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*(.*)$", line)
        if not m:
            continue
        inst, rest = m.groups()
        if "ragged-all-to-all-start(" in rest:
            shapes = re.findall(r"\[(\d+),(\d+)\]", rest.split("ragged-all-to-all-start(")[0])
            direction = f"{shapes[0][0]}->{shapes[1][0]}x{shapes[1][1]}" if len(shapes) >= 2 else "?"
            events.append(f"S:{inst.replace('ragged-all-to-all-start', 'a2a')}[{direction}]")
        elif "ragged-all-to-all-done(" in rest:
            events.append(f"D:{inst.replace('ragged-all-to-all-done', 'a2a')}")
        elif "CutlassCall" in rest:
            events.append("G")
        elif " opt-barrier(" in rest:
            events.append("B")
    compact = []
    for e in events:
        if compact and e == "G" and compact[-1].startswith("G"):
            compact[-1] = f"G{int(compact[-1][1:] or 1) + 1}"
        else:
            compact.append(e)
    return " ".join(compact)


def _scan(a, b):
    mesh = scan_compare._mesh()
    inp = scan_compare._inputs(mesh)
    compiled = {name: scan_compare._build(mesh, inp, *scan_compare.VARIANTS[name]) for name in (a, b)}
    results = {}
    for name, (exe, args) in compiled.items():
        results[name] = exe(*args)
        stats = exe.memory_analysis()
        print(
            json.dumps(
                dict(
                    variant=name,
                    temp_bytes=None if stats is None else int(stats.temp_size_in_bytes),
                    backward_schedule=_schedule(exe.as_text()),
                )
            ),
            flush=True,
        )
    cmp = scan_compare._compare(results[a], results[b])
    equal = all(v["equal"] for v in cmp.values())
    print(json.dumps(dict(pair=[a, b], bitwise=equal, detail=cmp)), flush=True)
    times = {name: [] for name in compiled}
    names = list(compiled)
    for rotation in range(4):
        order = names if rotation % 2 == 0 else names[::-1]
        for name in order:
            exe, args = compiled[name]
            for _ in range(2):
                jax.block_until_ready(exe(*args))
            samples = []
            for _ in range(10):
                start = time.perf_counter()
                jax.block_until_ready(exe(*args))
                samples.append(time.perf_counter() - start)
            times[name].append(statistics.median(samples))
    medians = {name: statistics.median(t) for name, t in times.items()}
    print(json.dumps(dict(median_seconds=medians, all_samples=times)), flush=True)
    return 0 if equal else 1


def main():
    _register()
    np.set_printoptions(linewidth=200)
    if sys.argv[1] == "single":
        sys.exit(1 if _single() else 0)
    sys.exit(_scan(sys.argv[2], sys.argv[3]))


if __name__ == "__main__":
    main()

"""Merged per-stream interval decomposition of one GPU's XProf trace (method of marin#8317).

Usage: uv run python overlap.py <xplane.pb> [--gpu 0] [--inventory]

Each device kernel is classified as compute, collective (by type), or memcpy. Per step
(one execution of the train-step XLA module), intervals of each class are merged and:
  compute busy        = |U compute|
  collective busy     = |U collective|
  exposed collective  = |U collective \\ U compute|
  realized overlap    = |U collective ∩ U compute| / |U collective|
  device idle         = span - |U all kernels and memcpys|
"""

import argparse
import collections
import json
import re
import sys

from marin.profiling.xplane import _xspace_message_class

PS = 1e-12


def load(path):
    xspace = _xspace_message_class()()
    with open(path, "rb") as f:
        xspace.ParseFromString(f.read())
    return xspace


def stat_value(stat, names):
    field = stat.WhichOneof("value")
    if field is None:
        return None
    if field == "ref_value":
        return names.get(int(stat.ref_value))
    v = getattr(stat, field)
    return v.decode(errors="replace") if isinstance(v, bytes) else v


def plane_events(plane):
    """Yield (line_name, name, start_ps, end_ps, stats) for every event in a plane."""
    names = {int(k): v.name for k, v in plane.stat_metadata.items()}
    meta = {int(k): v for k, v in plane.event_metadata.items()}
    meta_stats = {}
    for line in plane.lines:
        lname = line.display_name or line.name
        base = line.timestamp_ns * 1000
        for ev in line.events:
            m = meta.get(int(ev.metadata_id))
            if m is None:
                continue
            key = int(ev.metadata_id)
            if key not in meta_stats:
                meta_stats[key] = {names.get(int(s.metadata_id)): stat_value(s, names) for s in m.stats}
            stats = dict(meta_stats[key])
            for s in ev.stats:
                stats[names.get(int(s.metadata_id))] = stat_value(s, names)
            start = base + ev.offset_ps
            yield lname, (m.display_name or m.name), start, start + ev.duration_ps, stats


RAGGED = re.compile(r"ragged|RaggedAllToAll", re.I)
NCCL = re.compile(r"nccl|ncclDevKernel|ncclKernel", re.I)
BARRIER = re.compile(r"barrier|signal|wait_?value|cuStreamWait|flag", re.I)
MEMCPY = re.compile(r"memcpy|memset|Memcpy|Memset|MEMCPY", re.I)


def collective_type(name, hlo):
    """Map a kernel to a collective family, or None for compute."""
    text = f"{name} {hlo or ''}"
    if RAGGED.search(text):
        return "ragged_a2a"
    if re.search(r"all-gather|all_gather|AllGather", text):
        return "all_gather"
    if re.search(r"reduce-scatter|reduce_scatter|ReduceScatter", text):
        return "reduce_scatter"
    if re.search(r"all-reduce|all_reduce|AllReduce", text):
        return "all_reduce"
    if re.search(r"all-to-all|all_to_all|AllToAll", text):
        return "all_to_all"
    if re.search(r"collective-permute|collective_permute|SendRecv|Send|Recv", text):
        return "permute_sendrecv"
    if NCCL.search(text):
        return "nccl_other"
    if BARRIER.search(name):
        return "barrier_sync"
    return None


def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def length(iv):
    return sum(e - s for s, e in iv)


def intersect(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append([s, e])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def clip(iv, lo, hi):
    return [[max(s, lo), min(e, hi)] for s, e in iv if e > lo and s < hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xplane")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--inventory", action="store_true", help="print lines and top kernel names, then exit")
    ap.add_argument("--module", default=None, help="regex for the train-step module name")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    xs = load(args.xplane)
    planes = {p.name: p for p in xs.planes}
    dev = planes.get(f"/device:GPU:{args.gpu}")
    if dev is None:
        sys.exit(f"no GPU:{args.gpu} plane; planes: {list(planes)}")
    events = list(plane_events(dev))

    by_line = collections.defaultdict(list)
    for ev in events:
        by_line[ev[0]].append(ev)
    if args.inventory:
        for lname, evs in by_line.items():
            tot = collections.Counter()
            for _, n, s, e, _st in evs:
                tot[n] += e - s
            print(f"== {lname}: {len(evs)} events, {sum(tot.values()) * PS:.3f} s")
            for n, d in tot.most_common(12):
                print(f"   {d * PS:9.4f}  {n[:140]}")
        stat_keys = collections.Counter(k for ev in events[:20000] for k in ev[4])
        print("stat keys:", stat_keys.most_common(60))
        return

    # Steps: device windows bracketed by successive host launches of the train-step executable.
    # The last window closes at the last device event of that module.
    host = planes["/host:CPU"]
    launches = sorted(
        s for _, n, s, _e, _st in plane_events(host) if n.startswith("CommonPjRtLoadedExecutable::Execute (jit_train_step")
    )
    train_mod = "jit_train_step"
    dev_end = max(e for _, _n, _s, e, st in events if st.get("hlo_module") == train_mod)
    steps = list(zip(launches, launches[1:] + [dev_end]))
    mod_line = None

    kernel_lines = [k for k in by_line if k.startswith("Stream")]
    classes = collections.defaultdict(list)  # class -> intervals
    per_kernel = collections.defaultdict(lambda: collections.Counter())
    kernel_records = []
    for lname in kernel_lines:
        for _, name, s, e, st in by_line[lname]:
            hlo = st.get("hlo_op") or st.get("long_name") or st.get("tf_op") or ""
            if MEMCPY.search(name):
                cls = "memcpy"
            else:
                ctype = collective_type(name, hlo)
                cls = f"coll:{ctype}" if ctype else "compute"
            classes[cls].append((s, e))
            per_kernel[cls][name[:120]] += e - s
            kernel_records.append((s, e, cls, name, lname, st))

    report = {"train_module": train_mod, "kernel_lines": kernel_lines, "steps": []}
    for lo, hi in steps:
        span = hi - lo
        m = {c: merge(clip(iv, lo, hi)) for c, iv in classes.items()}
        comp = m.get("compute", [])
        coll_all = merge([x for c, iv in m.items() if c.startswith("coll:") for x in iv])
        everything = merge([x for iv in m.values() for x in iv])
        coll_exposed = length(coll_all) - length(intersect(coll_all, comp))
        row = {
            "span_s": span * PS,
            "compute_busy_s": length(comp) * PS,
            "collective_busy_s": length(coll_all) * PS,
            "collective_exposed_s": coll_exposed * PS,
            "realized_overlap_pct": 100 * (1 - coll_exposed / length(coll_all)) if coll_all else None,
            "memcpy_busy_s": length(m.get("memcpy", [])) * PS,
            "memcpy_exposed_s": (length(merge(m.get("memcpy", []) + [])) - length(intersect(merge(m.get("memcpy", [])), merge(comp + coll_all)))) * PS,
            "device_idle_s": (span - length(everything)) * PS,
            "by_type": {},
        }
        for c, iv in sorted(m.items()):
            if not c.startswith("coll:"):
                continue
            ex = length(iv) - length(intersect(iv, comp))
            row["by_type"][c[5:]] = {"busy_s": length(iv) * PS, "exposed_s": ex * PS}
        report["steps"].append(row)
    report["top_kernels"] = {c: [(n, d * PS) for n, d in cnt.most_common(15)] for c, cnt in per_kernel.items()}
    print(json.dumps(report, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()

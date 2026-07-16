"""Parse mfu-32g-nvlink job logs: per-GPU NVLink GB/s from counter deltas, per arm.

Input: raw `iris job logs` text on stdin or as argv[1]. NVLINK sample lines carry
unix timestamps from the pods; ARM_START/ARM_OK markers carry `date +%s` stamps
from task 0's shell, so windows and samples share a clock (pod wall clock, NTP).

For each arm we take the window [first steady step, arm end] — approximated as
[arm_start + lead, arm_end] with lead trimming compile + warmup — then report,
per task (node) and per GPU, mean TX and RX GB/s over the window, plus the peak
1-sample rate. GB = 1e9 bytes.
"""

import re
import sys
from collections import defaultdict

LINE = re.compile(r"task=(\S+?)/(\d+) \| NVLINK ([0-9.]+) (\S+) (.+)$", re.M)
ARM = re.compile(r"task=\S+/0 \| ARM_(START|OK|FAIL) (\S+)(?: rc=\d+)? (\d+)$", re.M)
STEP = re.compile(r'task=\S+?/0 \| \{"achieved.*"duration": ([0-9.]+).*"step": (\d+)', re.M)

text = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()

# arm windows from task-0 markers
arms = []  # (name, t_start, t_end)
pending = {}
for m in ARM.finditer(text):
    kind, name, ts = m.group(1), m.group(2), int(m.group(3))
    if kind == "START":
        pending[name] = ts
    elif name in pending:
        arms.append((name, pending.pop(name), ts))

# samples[(task, gpu)] = list of (ts, tx_bytes, rx_bytes)
samples = defaultdict(list)
for m in LINE.finditer(text):
    task, ts, fields = int(m.group(2)), float(m.group(3)), m.group(5)
    for f in fields.split():
        g, tx, rx = f.split(":")
        samples[(task, int(g))].append((ts, int(tx), int(rx)))

# steady-step durations per arm for context
for name, t0, t1 in arms:
    print(f"\n=== arm {name}: window {t0}..{t1} ({t1 - t0}s total incl. compile) ===")
    stats = []
    for (task, gpu), pts in sorted(samples.items()):
        pts = [p for p in pts if t0 <= p[0] <= t1]
        if len(pts) < 5:
            continue
        # trim to the active period: drop leading samples with ~zero delta (compile)
        rates = []
        for (ta, txa, rxa), (tb, txb, rxb) in zip(pts, pts[1:]):
            dt = tb - ta
            if dt <= 0:
                continue
            rates.append((ta, (txb - txa) / dt, (rxb - rxa) / dt))
        active = [r for r in rates if r[1] + r[2] > 1e9]  # >1 GB/s combined = training
        if not active:
            stats.append((task, gpu, 0.0, 0.0, 0.0, 0.0, len(rates), 0))
            continue
        mean_tx = sum(r[1] for r in active) / len(active) / 1e9
        mean_rx = sum(r[2] for r in active) / len(active) / 1e9
        peak_tx = max(r[1] for r in active) / 1e9
        peak_rx = max(r[2] for r in active) / 1e9
        stats.append((task, gpu, mean_tx, mean_rx, peak_tx, peak_rx, len(rates), len(active)))
    if not stats:
        print("no samples in window")
        continue
    print("task gpu  mean_tx  mean_rx  peak_tx  peak_rx  n_all n_active  (GB/s)")
    for s in stats:
        print(f"{s[0]:>4} {s[1]:>3}  {s[2]:7.1f}  {s[3]:7.1f}  {s[4]:7.1f}  {s[5]:7.1f}  {s[6]:5d} {s[7]:8d}")
    act = [s for s in stats if s[7] > 0]
    if act:
        n = len(act)
        print(
            f"fleet mean over {n} GPUs: tx {sum(s[2] for s in act)/n:.1f} GB/s, "
            f"rx {sum(s[3] for s in act)/n:.1f} GB/s; "
            f"max single-GPU peak tx {max(s[4] for s in act):.1f}, rx {max(s[5] for s in act):.1f} GB/s"
        )

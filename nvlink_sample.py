"""NVLink data-counter sampler: prints per-GPU cumulative TX/RX bytes at a fixed interval.

Runs alongside a benchmark; analysis computes GB/s from counter deltas. Output lines:
    NVLINK <unix_ts> <host> <gpu>:<tx_bytes>:<rx_bytes> ...
The first sample also dumps the raw `nvidia-smi nvlink -gt d` output (NVLINK_RAW prefix)
so parsing failures are diagnosable from the job log.
"""

import re
import socket
import subprocess
import sys
import time

GPU_HEADER = re.compile(r"^GPU (\d+):")
COUNTER = re.compile(r"Data (Tx|Rx):\s*([0-9]+)\s*(KiB|KBytes|KB|MiB|GiB|Bytes)", re.I)
UNIT = {"kib": 1024, "kbytes": 1024, "kb": 1024, "mib": 1024**2, "gib": 1024**3, "bytes": 1}
INTERVAL = 1.0


def sample() -> tuple[dict[int, list[int]], str]:
    out = subprocess.run(
        ["nvidia-smi", "nvlink", "-gt", "d"], capture_output=True, text=True, timeout=15
    ).stdout
    per: dict[int, list[int]] = {}
    cur = None
    for line in out.splitlines():
        m = GPU_HEADER.match(line.strip())
        if m:
            cur = int(m.group(1))
            per.setdefault(cur, [0, 0])
            continue
        m = COUNTER.search(line)
        if m and cur is not None:
            per[cur][0 if m.group(1).lower() == "tx" else 1] += int(m.group(2)) * UNIT[m.group(3).lower()]
    return per, out


def main() -> None:
    host = socket.gethostname()
    per, raw = sample()
    if not per or all(tx == 0 and rx == 0 for tx, rx in per.values()):
        for line in raw.splitlines()[:40]:
            print(f"NVLINK_RAW {line}", flush=True)
    while True:
        per, _ = sample()
        fields = " ".join(f"{g}:{v[0]}:{v[1]}" for g, v in sorted(per.items()))
        print(f"NVLINK {time.time():.2f} {host} {fields}", flush=True)
        sys.stdout.flush()
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()

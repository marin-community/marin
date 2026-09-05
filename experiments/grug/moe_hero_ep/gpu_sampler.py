# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""A per-node GPU and NVLink health sampler for the #8870 hunt.

The barrier-wait marks in the ragged all-to-all kernel showed one rank per rack arriving
50-160 ms late at a random layer, on three racks, with nothing upstream of the GPU to blame.
A flapping NVLink (``s9jvxs64`` GPU 0 link 14: 505 link recoveries and counting) is the one
counter that moved in the whole fleet while the hero ran on those racks. Link recovery is a
retrain, and a retrain stalls whatever is crossing that link for tens to hundreds of
milliseconds -- the right shape. This thread samples ``nvidia-smi`` once a second so a
recovery event and a barrier-wait event can be joined at one-second resolution.

Enabled by ``MARIN_DEBUG_GPU_SAMPLER_SECONDS`` (the period). Runs on one process per node
(``IRIS_MULTIGPU_PROCESS_INDEX == "0"``) because ``nvidia-smi`` reports all four GPUs. Logs a
clock/throttle line every sample and a link line only when a link error counter moves, so a
healthy node costs ~1 line/s and a flapping one names its link.
"""

import logging
import os
import re
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

_LINK_COUNTERS = (
    "Total link recovery events",
    "Symbol Errors",
    "Rx Errors",
    "Rx remote Errors",
    "Rx General Errors",
    "Tx discards",
)
_GPU_QUERY = "index,clocks_throttle_reasons.active,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,power.draw"


def _run(args: list[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _link_counters() -> dict[tuple[int, int, str], int]:
    out: dict[tuple[int, int, str], int] = {}
    gpu = -1
    for line in _run(["nvidia-smi", "nvlink", "-e"]).splitlines():
        m = re.match(r"GPU (\d+): ", line)
        if m:
            gpu = int(m.group(1))
            continue
        m = re.match(r"\s*Link (\d+): (.+?): (\d+)\s*$", line)
        if m and gpu >= 0 and m.group(2) in _LINK_COUNTERS:
            out[(gpu, int(m.group(1)), m.group(2))] = int(m.group(3))
    return out


def _sample_loop(period: float, hostname: str) -> None:
    prev = _link_counters()
    while True:
        time.sleep(period)
        stamp = time.strftime("%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1000):03d}"
        gpus = _run(["nvidia-smi", f"--query-gpu={_GPU_QUERY}", "--format=csv,noheader,nounits"])
        logger.info("GPUSAMPLE %s %s | %s", stamp, hostname, " ; ".join(l.strip() for l in gpus.splitlines()))
        cur = _link_counters()
        for key, value in cur.items():
            delta = value - prev.get(key, value)
            if delta:
                gpu, link, name = key
                logger.warning("GPUSAMPLE %s %s | LINK gpu%d link%d %s +%d (now %d)", stamp, hostname, gpu, link, name, delta, value)
        prev = cur or prev


def maybe_start_gpu_sampler() -> None:
    """Start the sampler thread if enabled for this node's leader process; otherwise no-op."""
    period = os.environ.get("MARIN_DEBUG_GPU_SAMPLER_SECONDS")
    if not period:
        return
    if os.environ.get("IRIS_MULTIGPU_PROCESS_INDEX", "0") != "0":
        return
    try:
        seconds = max(0.2, float(period))
    except ValueError:
        logger.warning("MARIN_DEBUG_GPU_SAMPLER_SECONDS=%r is not a number; sampler disabled", period)
        return
    hostname = os.uname().nodename
    if not _run(["nvidia-smi", "-L"]):
        logger.warning("GPU sampler requested but nvidia-smi is unavailable on %s", hostname)
        return
    threading.Thread(target=_sample_loop, args=(seconds, hostname), name="gpu-sampler", daemon=True).start()
    logger.info("GPU sampler started on %s every %.1fs (link counters logged on change)", hostname, seconds)

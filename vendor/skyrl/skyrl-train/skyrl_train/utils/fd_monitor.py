"""Self-contained file-descriptor monitor for the SkyRL RL driver.

This is a minimal, dependency-free port of the FileDescriptorMonitor used in
the OT-Agent datagen path (`hpc/local_runner_utils.py`). It is duplicated here
on purpose so it does NOT need to import anything from OT-Agent — the RL conda
env may not have OT-Agent on its path.

Goal: log file-descriptor usage of the *driver* process (the one that
FD-aborts with `uv__epoll_ctl_prep` SIGABRT on long a3 RL chains) every
`interval` seconds on a daemon thread. Output uses the same `[fd-monitor]`
prefix/format as the datagen monitor so existing greps keep working, and uses
`print(..., flush=True)` so it lands in the SLURM `.out`.

Only start this on the driver / main entrypoint process (not every Ray
worker) to avoid log spam.
"""

from __future__ import annotations

import os
import resource
import threading
import time
from pathlib import Path

DEFAULT_FD_MONITOR_INTERVAL = 120  # 2 minutes


def _get_fd_usage() -> tuple:
    """Get current file descriptor usage.

    Returns:
        Tuple of (current_open_fds, soft_limit, hard_limit, percent_used).
        Returns (-1, -1, -1, 0.0) on any failure (e.g. /proc unavailable).
    """
    try:
        pid = os.getpid()
        fd_dir = Path(f"/proc/{pid}/fd")
        if fd_dir.exists():
            current_fds = len(list(fd_dir.iterdir()))
        else:
            # Fallback for non-Linux systems (no /proc) — count via fstat.
            current_fds = 0
            for fd in range(1024):
                try:
                    os.fstat(fd)
                    current_fds += 1
                except OSError:
                    pass

        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        percent_used = (current_fds / soft_limit * 100) if soft_limit > 0 else 0
        return current_fds, soft_limit, hard_limit, percent_used
    except Exception:
        return -1, -1, -1, 0.0


def _get_mem_usage() -> tuple:
    """Get this process's RSS + system available memory (Linux /proc).

    Added 2026-05-28 to test the OOM hypothesis: the driver aborts in
    `uv__epoll_ctl_prep` with plenty of FD headroom, so the suspected cause is
    host memory exhaustion (ENOMEM), not file descriptors. Self-contained
    (no psutil) — reads /proc directly.

    Returns:
        Tuple of (rss_kb, mem_avail_kb, mem_total_kb, system_percent_used).
        Returns (-1, -1, -1, 0.0) on any failure (e.g. /proc unavailable).
    """
    try:
        rss_kb = -1
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    break
        mem_avail_kb = mem_total_kb = -1
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    mem_avail_kb = int(line.split()[1])
                elif line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
        system_pct = (mem_total_kb - mem_avail_kb) / mem_total_kb * 100 if mem_total_kb > 0 else 0.0
        return rss_kb, mem_avail_kb, mem_total_kb, system_pct
    except Exception:
        return -1, -1, -1, 0.0


def _get_cgroup_mem() -> tuple:
    """Get this container's CGROUP memory current + limit (bytes).

    Added 2026-07-11 for the 80B naive-map policy-node host-RAM OOM: the pod is
    OOM-killed at the ``--memory`` CGROUP cap (~1303 GiB for ``--memory 1400GB``),
    which is SMALLER than the node's physical RAM (~2014 GiB). So the
    ``/proc/meminfo`` node view (``_get_mem_usage``) reads "healthy" (~65% of the
    node) right up to the kernel cgroup-OOM SIGKILL — it never sees the binding
    cap. This reads the cgroup's OWN accounting (v2 ``memory.current`` /
    ``memory.max``, falling back to v1 ``memory.usage_in_bytes`` /
    ``memory.limit_in_bytes``) so the log shows RSS/usage vs the ACTUAL cap.

    Returns:
        Tuple of (cur_bytes, max_bytes). ``max_bytes`` is -1 when unlimited
        ("max" on v2, or a sentinel-huge value on v1). (-1, -1) on any failure.
    """
    try:
        # cgroup v2 (unified hierarchy) — the container sees its own cgroup root.
        v2_cur = "/sys/fs/cgroup/memory.current"
        v2_max = "/sys/fs/cgroup/memory.max"
        if os.path.exists(v2_cur):
            with open(v2_cur) as f:
                cur = int(f.read().strip())
            mx = -1
            try:
                with open(v2_max) as f:
                    raw = f.read().strip()
                mx = -1 if raw == "max" else int(raw)
            except Exception:
                mx = -1
            return cur, mx
        # cgroup v1 fallback.
        v1_cur = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        v1_max = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        if os.path.exists(v1_cur):
            with open(v1_cur) as f:
                cur = int(f.read().strip())
            mx = -1
            try:
                with open(v1_max) as f:
                    lim = int(f.read().strip())
                # v1 "unlimited" is a near-INT64 sentinel; treat >~1 PiB as no cap.
                mx = -1 if lim > (1 << 50) else lim
            except Exception:
                mx = -1
            return cur, mx
        return -1, -1
    except Exception:
        return -1, -1


def _get_cgroup_pids() -> list:
    """Return the pids in THIS container's cgroup.

    Added 2026-07-13 for the 80B host-RAM breakdown: the cgroup TOTAL
    (``_get_cgroup_mem``) has ZERO attribution, so a ~750 GiB/node OOM is
    unexplained. This reads the cgroup's own process list (v2
    ``cgroup.procs``, falling back to v1 ``memory/cgroup.procs``) so we can sum
    + bucket per-process RSS. Returns [] on any failure (non-Linux, no cgroupfs).
    """
    for p in (
        "/sys/fs/cgroup/cgroup.procs",  # v2 unified hierarchy (container root)
        "/sys/fs/cgroup/memory/cgroup.procs",
    ):  # v1 memory controller
        try:
            if os.path.exists(p):
                with open(p) as f:
                    return [int(x) for x in f.read().split()]
        except Exception:
            continue
    return []


def _proc_rss_kb(pid: int) -> int:
    """VmRSS (KiB) for ``pid`` from /proc/<pid>/status, or 0 on any failure."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def _proc_cmdline(pid: int) -> str:
    """NUL-joined /proc/<pid>/cmdline as a space-joined string (Ray sets the
    proctitle here via setproctitle, e.g. ``ray::MegatronPolicyWorkerBase.forward``),
    or "" on failure."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except Exception:
        return ""


def _bucket_for_cmdline(cmd: str) -> str:
    """Bucket a process by its cmdline/proctitle into the host-RAM breakdown
    groups. Order matters (raylet/plasma before the generic ray-worker match)."""
    c = cmd.lower()
    if "raylet" in c:
        return "raylet"
    if "plasma" in c:  # separate plasma_store_server (older Ray); embedded in raylet on newer
        return "plasma"
    if "rolloutcoordinator" in c or "rollout_coordinator" in c or "skyrl_entrypoint" in c:
        return "coord"
    if "policyworker" in c or "policy_worker" in c:
        return "workers"
    return "other"


def _get_shm_bytes() -> int:
    """/dev/shm bytes in use (the plasma object store mmaps here — shmem is
    charged to the cgroup but is NOT in any single process's VmRSS). -1 on
    failure."""
    try:
        st = os.statvfs("/dev/shm")
        return (st.f_blocks - st.f_bfree) * st.f_frsize
    except Exception:
        return -1


def _get_pinned_bytes() -> int:
    """torch host-pinned (page-locked) allocator current bytes, or -1 if not
    cheaply available. Best-effort: reads the caching-host-allocator counters
    (torch >=2.x ``host_memory_stats``); guarded so it never costs more than a
    dict lookup and never raises. Only invoked on the policy worker (torch is
    already imported there)."""
    try:
        import torch

        stats = torch.cuda.host_memory_stats()
        cur = stats.get("allocated_bytes.all.current")
        if cur is not None:
            return int(cur)
    except Exception:
        pass
    return -1


def _log_host_ram_breakdown(peaks: dict | None = None) -> None:
    """Attribute the cgroup host-RAM total across the cgroup's processes + shm +
    torch host-pinned, emitting ONE compact ``HOST_RAM_BREAKDOWN`` line/sample.

    WHY (2026-07-13, the 80B naive-map gs1 host-RAM OOM): the monitor above prints
    the cgroup TOTAL and only the monitor process's OWN RSS — zero attribution. The
    gs1 ~750 GiB/node peak needs attribution. Buckets by
    /proc/<pid>/cmdline: policy workers / raylet / plasma-store /
    rollout-coordinator / other; plus /dev/shm (plasma) and host-pinned bytes.

    CAVEAT: summed VmRSS double-counts shared pages (shm, shared libs), so the
    per-bucket figures are an ATTRIBUTION, not an exact partition — the
    authoritative total is ``cgroup=cur/cap`` (also on the OK/WARN line above).
    Best-effort; never raises into the sampler.
    """
    import socket

    gib_b = 1073741824.0
    buckets = {"workers": 0, "raylet": 0, "plasma": 0, "coord": 0, "other": 0}
    pids = _get_cgroup_pids()
    for pid in pids:
        rss_kb = _proc_rss_kb(pid)
        if rss_kb <= 0:
            continue
        buckets[_bucket_for_cmdline(_proc_cmdline(pid))] += rss_kb
    shm_b = _get_shm_bytes()
    pinned_b = _get_pinned_bytes()
    cg_cur, cg_max = _get_cgroup_mem()

    def _g(kb: int) -> float:  # KiB -> GiB
        return kb / 1048576.0

    if cg_cur >= 0:
        cg_str = f"{cg_cur / gib_b:.1f}/{cg_max / gib_b:.1f}" if cg_max > 0 else f"{cg_cur / gib_b:.1f}/nocap"
    else:
        cg_str = "n/a"
    shm_str = f"{shm_b / gib_b:.1f}" if shm_b >= 0 else "n/a"
    pinned_str = f"{pinned_b / gib_b:.1f}" if pinned_b >= 0 else "n/a"

    # Track the peak of the dominant buckets across the process lifetime (the
    # numbers that matter for the OOM post-mortem — the instantaneous sample
    # rarely lands on the pre-kill peak).
    if peaks is not None:
        peaks["bd_workers_kb"] = max(peaks.get("bd_workers_kb", 0), buckets["workers"])
        if shm_b > 0:
            peaks["bd_shm_b"] = max(peaks.get("bd_shm_b", 0), shm_b)

    ts = time.strftime("%H:%M:%S")
    print(
        f"[fd-monitor] [{ts}] HOST_RAM_BREAKDOWN node={socket.gethostname()} "
        f"cgroup={cg_str} GiB pids={len(pids)} | "
        f"workers={_g(buckets['workers']):.1f} raylet={_g(buckets['raylet']):.1f} "
        f"plasma={_g(buckets['plasma']):.1f} coord={_g(buckets['coord']):.1f} "
        f"shm={shm_str} pinned={pinned_str} other={_g(buckets['other']):.1f} GiB",
        flush=True,
    )


def _log_status(peaks: dict | None = None, breakdown: bool = False) -> None:
    """Log current file descriptor status with the [fd-monitor] prefix.

    When ``breakdown`` is True, ALSO emit the ``HOST_RAM_BREAKDOWN`` attribution
    line (cgroup total bucketed across pids + shm + pinned). ``breakdown`` is
    only passed True by the POLICY worker's gated monitor
    (``SKYRL_POLICY_HOST_RAM_MONITOR``); the gen-coordinator / head monitors keep
    the default False, so their output is byte-identical to before."""
    current, soft, hard, percent = _get_fd_usage()

    if current < 0:
        print("[fd-monitor] Unable to read file descriptor usage", flush=True)
        return

    if percent >= 90:
        level = "CRITICAL"
    elif percent >= 75:
        level = "WARNING"
    elif percent >= 50:
        level = "INFO"
    else:
        level = "OK"

    timestamp = time.strftime("%H:%M:%S")
    print(
        f"[fd-monitor] [{timestamp}] {level}: {current:,} / {soft:,} FDs open "
        f"({percent:.1f}% of soft limit, hard limit: {hard:,})",
        flush=True,
    )

    if percent >= 75:
        print(
            "[fd-monitor] Consider reducing --n_concurrent or increasing ulimit -n",
            flush=True,
        )

    # Memory telemetry (RSS + system available) — the suspected cause of the
    # driver SIGABRT once FDs were ruled out (see
    # agent_logs/2026-05-28_fresh_a3_chain_crash_not_fd.md).
    rss_kb, avail_kb, total_kb, sys_pct = _get_mem_usage()
    if rss_kb >= 0 and total_kb > 0:
        if sys_pct >= 95:
            mlevel = "CRITICAL"
        elif sys_pct >= 85:
            mlevel = "WARNING"
        elif sys_pct >= 70:
            mlevel = "INFO"
        else:
            mlevel = "OK"
        gib = 1048576.0  # KiB per GiB
        # Track the peak RSS across the process lifetime (the number that matters
        # for a cgroup-OOM post-mortem — the instantaneous sample rarely catches
        # the exact pre-kill peak).
        peak_str = ""
        if peaks is not None:
            peaks["rss_kb"] = max(peaks.get("rss_kb", 0), rss_kb)
            peak_str = f", peak RSS {peaks['rss_kb'] / gib:.2f} GiB"
        print(
            f"[fd-monitor] [{timestamp}] {mlevel}: RSS {rss_kb / gib:.2f} GiB{peak_str} | "
            f"node mem {(total_kb - avail_kb) / gib:.1f}/{total_kb / gib:.1f} GiB used "
            f"({sys_pct:.1f}%), avail {avail_kb / gib:.1f} GiB",
            flush=True,
        )
        if sys_pct >= 85:
            print(
                "[fd-monitor] Node memory pressure HIGH — consider reducing "
                "harbor.n_concurrent_trials / trainer.rollout_buffer.max_in_flight",
                flush=True,
            )

    # CGROUP telemetry — the ACTUAL binding cap on CoreWeave (the `--memory`
    # limit is smaller than the node's physical RAM, so `node mem` above never
    # sees the wall that OOM-kills the pod). This is the decisive host-RAM signal
    # for the 80B GDN-scan policy-node OOM.
    gib_b = 1073741824.0  # bytes per GiB
    cg_cur, cg_max = _get_cgroup_mem()
    if cg_cur >= 0:
        cg_peak_str = ""
        if peaks is not None:
            peaks["cg_cur"] = max(peaks.get("cg_cur", 0), cg_cur)
            cg_peak_str = f", peak {peaks['cg_cur'] / gib_b:.2f} GiB"
        if cg_max > 0:
            cg_pct = cg_cur / cg_max * 100
            if cg_pct >= 95:
                clevel = "CRITICAL"
            elif cg_pct >= 85:
                clevel = "WARNING"
            elif cg_pct >= 70:
                clevel = "INFO"
            else:
                clevel = "OK"
            print(
                f"[fd-monitor] [{timestamp}] {clevel}: cgroup mem "
                f"{cg_cur / gib_b:.2f}/{cg_max / gib_b:.2f} GiB ({cg_pct:.1f}% of cap)"
                f"{cg_peak_str}",
                flush=True,
            )
            if cg_pct >= 85:
                print(
                    "[fd-monitor] CGROUP memory pressure HIGH — approaching the "
                    "--memory cap; OOM-kill imminent (reduce policy ranks/node, "
                    "GDN scan working set, or n_concurrent_trials)",
                    flush=True,
                )
        else:
            print(
                f"[fd-monitor] [{timestamp}] OK: cgroup mem {cg_cur / gib_b:.2f} GiB (no cap){cg_peak_str}",
                flush=True,
            )

    # Host-RAM allocation BREAKDOWN — only on the gated policy-worker monitor
    # (breakdown=True). This is the decisive signal for the 80B gs1 host-RAM OOM:
    # WHICH bucket (policy workers / plasma / shm / pinned / ...) holds the ~750 GiB.
    if breakdown:
        try:
            _log_host_ram_breakdown(peaks)
        except Exception:  # pragma: no cover - best-effort telemetry
            pass


def _run(stop_event: threading.Event, interval: int, breakdown: bool = False) -> None:
    """Background thread loop: log immediately, then every `interval` seconds.

    Maintains a per-thread ``peaks`` dict so each sample can report the running
    peak RSS + peak cgroup usage (the numbers that matter for a cgroup-OOM
    post-mortem — the instantaneous sample rarely lands on the pre-kill peak).
    ``breakdown`` is threaded to ``_log_status`` (policy-worker monitor only).
    """
    peaks: dict = {}
    _log_status(peaks, breakdown)
    while not stop_event.is_set():
        stop_event.wait(interval)
        if not stop_event.is_set():
            _log_status(peaks, breakdown)


def start_fd_monitor(interval_seconds: int = DEFAULT_FD_MONITOR_INTERVAL, breakdown: bool = False) -> threading.Event:
    """Start a daemon thread that periodically logs FD usage of this process.

    Self-contained and best-effort: never raises into the caller. Intended to
    be started once in the RL driver entrypoint (skyrl_entrypoint), not in
    every Ray worker.

    Args:
        interval_seconds: How often to log FD usage (default: 120s). A value
            <= 0 disables the monitor (no-op).
        breakdown: When True, ALSO emit the ``HOST_RAM_BREAKDOWN`` cgroup
            allocation-attribution line each sample. Passed True only by the
            gated policy-worker monitor; default False keeps the gen/head
            monitors byte-identical.

    Returns:
        The threading.Event used to stop the loop. Set it to stop early; the
        thread is a daemon so it does not need to be joined for shutdown.
    """
    stop_event = threading.Event()
    if interval_seconds <= 0:
        print("[fd-monitor] Disabled (interval <= 0)", flush=True)
        return stop_event

    thread = threading.Thread(
        target=_run,
        args=(stop_event, interval_seconds, breakdown),
        daemon=True,
        name="fd-monitor",
    )
    thread.start()
    print(f"[fd-monitor] Started monitoring (every {interval_seconds}s)", flush=True)
    return stop_event

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Start mimalloc-pprof before the training application imports JAX."""

import atexit
import ctypes
import os
import re
import sys
import threading
from pathlib import Path
from typing import ClassVar

_PROCESS_INDEX_ENV = "IRIS_MULTIGPU_PROCESS_INDEX"
_LD_PRELOAD_ENV = "LD_PRELOAD"
_SAMPLE_INTERVAL_BYTES = 128 * 1024
_SAMPLE_INTERVAL = 60
_PROFILE_PROCESS_INDICES = frozenset((0, 4))
_TOP_STACKS = 10
_STACK_FRAMES = 12
_MISSING_PRELOAD_EXIT_CODE = 86
_STARTUP_FAILURE_EXIT_CODE = 87
_PERIODIC_FAILURE_EXIT_CODE = 88
_SAMPLE_LINE = re.compile(r"^\s*\d+:\s+(\d+)\s+\[")
# AArch64 return addresses may carry pointer-authentication bits above the userspace VA.
_USERSPACE_ADDRESS_MASK = (1 << 48) - 1


class _PprofStats(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("size", ctypes.c_size_t),
        ("version", ctypes.c_int),
        ("enabled", ctypes.c_bool),
        ("accum", ctypes.c_bool),
        ("sample_rate", ctypes.c_size_t),
        ("live_samples", ctypes.c_size_t),
        ("live_bytes", ctypes.c_size_t),
        ("accum_samples", ctypes.c_size_t),
        ("accum_bytes", ctypes.c_size_t),
        ("unique_stacks", ctypes.c_size_t),
        ("arena_committed", ctypes.c_size_t),
        ("stack_table_overflows", ctypes.c_size_t),
        ("dropped_samples", ctypes.c_size_t),
        ("heap_committed", ctypes.c_size_t),
        ("heap_reserved", ctypes.c_size_t),
        ("heap_malloc_requested", ctypes.c_size_t),
        ("heap_pages", ctypes.c_size_t),
        ("heap_pages_abandoned", ctypes.c_size_t),
        ("heap_count", ctypes.c_size_t),
        ("theap_count", ctypes.c_size_t),
        ("heap_purged", ctypes.c_size_t),
        ("heap_stats_detailed", ctypes.c_bool),
    ]


class _DlInfo(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("filename", ctypes.c_char_p),
        ("base", ctypes.c_void_p),
        ("symbol_name", ctypes.c_char_p),
        ("symbol_address", ctypes.c_void_p),
    ]


def _log(message: str) -> None:
    os.write(sys.stderr.fileno(), f"[mimalloc-pprof] {message}\n".encode())


def _rss_kib() -> int | None:
    """Return resident memory in KiB, or ``None`` when procfs lacks VmRSS."""
    for line in Path("/proc/self/status").read_text().splitlines():
        name, separator, value = line.partition(":")
        pieces = value.split()
        if separator and name == "VmRSS" and pieces and pieces[0].isdigit():
            return int(pieces[0])
    return None


class _Sampler:
    def __init__(self, library_path: Path, interval: int, process_index: int):
        self.library = ctypes.CDLL(str(library_path))
        self.interval = interval
        self.process_index = process_index
        self.sequence = 0
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.previous_samples: dict[tuple[int, ...], int] = {}
        self.output_dir = Path("/tmp/mimalloc-pprof") / f"rank-{process_index:03d}-pid-{os.getpid()}"
        self.output_dir.mkdir(parents=True)

        start = self.library.mi_prof_start
        start.argtypes = [ctypes.c_size_t]
        start.restype = ctypes.c_bool
        if not start(_SAMPLE_INTERVAL_BYTES):
            raise RuntimeError("mi_prof_start failed")

    def _dump(self, path: Path) -> None:
        dump = self.library.mi_prof_dump
        dump.argtypes = [ctypes.c_char_p]
        dump.restype = ctypes.c_bool
        if not dump(os.fsencode(path)):
            raise RuntimeError("mi_prof_dump failed")

    @staticmethod
    def _samples(path: Path) -> dict[tuple[int, ...], int]:
        samples: dict[tuple[int, ...], int] = {}
        for line in path.read_text().splitlines():
            prefix, separator, raw_stack = line.partition(" @ ")
            if not separator:
                continue
            match = _SAMPLE_LINE.match(prefix)
            if match is None:
                continue
            stack = tuple(
                int(address, 16) & _USERSPACE_ADDRESS_MASK for address in raw_stack.split() if address.startswith("0x")
            )
            samples[stack] = samples.get(stack, 0) + int(match.group(1))
        return samples

    @staticmethod
    def _symbolize(address: int) -> str:
        process = ctypes.CDLL(None)
        process.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
        process.dladdr.restype = ctypes.c_int
        info = _DlInfo()
        if process.dladdr(ctypes.c_void_p(address), ctypes.byref(info)) == 0:
            return hex(address)
        module = Path(os.fsdecode(info.filename)).name if info.filename else "unknown"
        if info.symbol_name:
            symbol = os.fsdecode(info.symbol_name)
            offset = address - int(info.symbol_address or 0)
            return f"{module}:{symbol}+0x{offset:x}"
        offset = address - int(info.base or 0)
        return f"{module}+0x{offset:x}"

    def _log_diff(self, label: str, path: Path) -> None:
        samples = self._samples(path)
        deltas = sorted(
            (
                (live_bytes - self.previous_samples.get(stack, 0), live_bytes, stack)
                for stack, live_bytes in samples.items()
            ),
            reverse=True,
        )
        for index, (delta, live_bytes, stack) in enumerate(deltas[:_TOP_STACKS], start=1):
            frames = " <- ".join(self._symbolize(frame) for frame in stack[:_STACK_FRAMES])
            _log(
                f"diff rank={self.process_index} label={label} index={index} "
                f"sampled_delta_bytes={delta} sampled_live_bytes={live_bytes} stack={frames}"
            )
        self.previous_samples = samples

    def _stats(self) -> _PprofStats:
        stats = _PprofStats()
        stats.size = ctypes.sizeof(_PprofStats)
        stats.version = 3
        stats_get = self.library.mi_prof_stats_get
        stats_get.argtypes = [ctypes.POINTER(_PprofStats)]
        stats_get.restype = ctypes.c_bool
        if not stats_get(ctypes.byref(stats)):
            raise RuntimeError("mi_prof_stats_get failed")
        return stats

    def sample(self, label: str) -> None:
        with self.lock:
            self.sequence += 1
            path = self.output_dir / f"sample-{self.sequence:04d}-{label}.heap"
            self._dump(path)
            self._log_diff(label, path)
            stats = self._stats()
            cgroup_path = Path("/sys/fs/cgroup/memory.current")
            cgroup_bytes = int(cgroup_path.read_text()) if cgroup_path.exists() else None
            _log(
                f"stats rank={self.process_index} label={label} rss_kib={_rss_kib()} "
                f"cgroup_bytes={cgroup_bytes} sample_rate={stats.sample_rate} "
                f"live_samples={stats.live_samples} live_bytes={stats.live_bytes} "
                f"unique_stacks={stats.unique_stacks} dropped_samples={stats.dropped_samples} "
                f"heap_committed={stats.heap_committed} heap_reserved={stats.heap_reserved} profile={path}"
            )

    def run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                self.sample("periodic")
            except Exception as error:
                _log(f"periodic sample failed: {error!r}")
                os._exit(_PERIODIC_FAILURE_EXIT_CODE)

    def finish(self) -> None:
        self.stop_event.set()
        self.sample("final")
        self.library.mi_prof_stop()


def _start() -> None:
    raw_process_index = os.environ.get(_PROCESS_INDEX_ENV)
    if raw_process_index is None:
        return
    process_index = int(raw_process_index)
    if process_index not in _PROFILE_PROCESS_INDICES:
        return

    library_path = Path(os.environ[_LD_PRELOAD_ENV].split(":", 1)[0])
    if "libmimalloc" not in Path("/proc/self/maps").read_text():
        _log(f"mimalloc is not loaded; {_LD_PRELOAD_ENV}={os.environ.get(_LD_PRELOAD_ENV)}")
        os._exit(_MISSING_PRELOAD_EXIT_CODE)

    sampler = _Sampler(library_path, _SAMPLE_INTERVAL, process_index)
    sampler.sample("start")
    atexit.register(sampler.finish)
    threading.Thread(target=sampler.run, name="mimalloc-pprof", daemon=True).start()


try:
    _start()
except BaseException as error:
    _log(f"startup failed: {error!r}")
    os._exit(_STARTUP_FAILURE_EXIT_CODE)

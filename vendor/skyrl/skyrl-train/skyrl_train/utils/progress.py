"""Log-capture-safe progress bars for SkyRL.

Motivation
----------
On captured / non-interactive logging pipelines (e.g. CoreWeave / Iris container
logs, SLURM ``.out`` files) ``tqdm`` progress bars are effectively invisible: tqdm
refreshes in place using carriage returns (``\\r``) and the capture layer typically
only records newline (``\\n``)-terminated lines. loguru / ``print`` lines DO survive,
so this module provides a drop-in ``tqdm`` replacement that, when stderr is NOT an
interactive TTY, emits *newline-terminated* progress lines (throttled) via loguru
(falling back to ``print(..., flush=True)``) instead of ``\\r`` in-place updates.

When stderr IS a TTY (interactive local dev) it delegates to the real ``tqdm`` so
nothing regresses.

Gate
----
By default the mode is auto-detected from ``sys.stderr.isatty()`` — this covers
CoreWeave AND SLURM with no launcher wiring. ``trainer.progress.mode`` can force
``tqdm``, ``logging``, or ``auto``.

Throttling (bounds log volume — this environment has had disk-brick incidents from
log floods): a progress line is emitted only when a new percent-bucket is crossed OR
a heartbeat interval elapses (and a per-emit minimum interval is respected), PLUS a
guaranteed final 100% line. The thresholds live under ``trainer.progress``.

API
---
Drop-in for the subset of the tqdm API SkyRL uses:
  * construction: ``tqdm(iterable=None, total=..., initial=..., desc=..., disable=...,
    position=...)``
  * ``.update(n=1)``, ``.set_postfix(dict_or_kwargs)``, ``.close()``
  * iterator form: ``for x in tqdm(iterable): ...``
  * context manager: ``with tqdm(...) as pbar: ...``
  * async classmethod: ``await tqdm.gather(*coros, desc=..., total=..., disable=...)``
    (drop-in for ``tqdm.asyncio.tqdm.gather``)
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from tqdm import tqdm as _std_tqdm
from tqdm.asyncio import tqdm as _std_atqdm

try:  # loguru is a first-class dep of skyrl_train and its lines are provably captured
    from loguru import logger as _loguru_logger
except Exception:  # pragma: no cover - defensive
    _loguru_logger = None


@dataclass(frozen=True)
class _ProgressSettings:
    mode: str
    min_interval_seconds: float
    heartbeat_seconds: float
    percent_step: float
    count_step: float


_SETTINGS: _ProgressSettings | None = None


def configure_progress(config) -> None:
    """Install process-local progress settings from ``trainer.progress``."""
    global _SETTINGS
    _SETTINGS = _ProgressSettings(
        mode=str(config.mode),
        min_interval_seconds=float(config.min_interval_seconds),
        heartbeat_seconds=float(config.heartbeat_seconds),
        percent_step=float(config.percent_step),
        count_step=float(config.count_step),
    )


def _use_real_tqdm() -> bool:
    """Return True iff we should delegate to the real tqdm (interactive TTY)."""
    if _SETTINGS is None:
        return True
    mode = _SETTINGS.mode
    if mode == "tqdm":
        return True
    if mode == "logging":
        return False
    # auto
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def _emit(line: str) -> None:
    if _loguru_logger is not None:
        # opt(depth=...) keeps loguru's source location off this helper; failures fall through
        try:
            _loguru_logger.opt(depth=1).info(line)
            return
        except Exception:
            pass
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass


def _fmt_postfix(postfix: dict) -> str:
    parts = []
    for k, v in postfix.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3g}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


class _LoggingProgress:
    """Newline-terminated, throttled progress reporter (non-TTY fallback)."""

    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        total: Optional[int] = None,
        initial: int = 0,
        desc: Optional[str] = None,
        disable: bool = False,
        **_ignored: Any,
    ):
        self.iterable = iterable
        if total is None and iterable is not None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                total = None
        self.total = total
        self.n = initial or 0
        self.desc = desc or "progress"
        self.disable = disable
        self._postfix = ""
        self._start = time.time()
        self._last_emit_t = 0.0
        self._last_emit_n = -1
        self._last_bucket: Optional[int] = None
        self._closed = False
        self._lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        assert _SETTINGS is not None, "configure_progress must run before logging progress is constructed"
        self._settings = _SETTINGS
        # Initial line so operators immediately see the bar exists (and its total).
        self._maybe_emit(force=True)
        if not self.disable and self._settings.heartbeat_seconds > 0:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"skyrl-progress-{self.desc}",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self._settings.heartbeat_seconds):
            self._maybe_emit()

    # --- bucketing ---------------------------------------------------------
    def _bucket(self) -> Optional[int]:
        if self.total and self.total > 0:
            return int((self.n * 100.0 / self.total) // self._settings.percent_step)
        return int(self.n // self._settings.count_step)

    def _should_emit(self, now: float, force: bool) -> bool:
        if self.disable:
            return False
        if force:
            return True
        if self._last_emit_t == 0.0:
            return True
        if (now - self._last_emit_t) < self._settings.min_interval_seconds:
            return False
        # Heartbeat is checked BEFORE the unchanged-counter early-return below: a
        # FROZEN counter must still emit periodically, otherwise a stalled bar goes
        # silent and becomes indistinguishable in the log from a healthy idle one.
        # That silence is exactly what hid the v0d generation-buffer wedge (stuck at
        # 31/64 for ~4h with zero log lines). Emitting "31/64 [Xs]" every heartbeat
        # keeps a stall visible + greppable (same n, growing elapsed). Env-tunable via
        # trainer.progress.heartbeat_seconds.
        if (now - self._last_emit_t) >= self._settings.heartbeat_seconds:
            return True
        if self.n == self._last_emit_n:
            return False
        bucket = self._bucket()
        if bucket != self._last_bucket:
            return True
        return False

    def _maybe_emit(self, force: bool = False) -> None:
        with self._lock:
            if self.disable:
                return
            now = time.time()
            if not self._should_emit(now, force):
                return
            elapsed = now - self._start
            if self.total and self.total > 0:
                pct = 100.0 * self.n / self.total
                line = f"{self.desc}: {self.n}/{self.total} ({pct:.0f}%) [{elapsed:.0f}s]"
            else:
                line = f"{self.desc}: {self.n} [{elapsed:.0f}s]"
            if self._postfix:
                line = f"{line} | {self._postfix}"
            _emit(line)
            self._last_emit_t = now
            self._last_emit_n = self.n
            self._last_bucket = self._bucket()

    # --- tqdm-compatible surface ------------------------------------------
    def update(self, n: int = 1) -> None:
        with self._lock:
            self.n += n
            self._maybe_emit()

    def set_postfix(self, ordered_dict: Optional[dict] = None, refresh: bool = True, **kwargs: Any) -> None:
        merged = {}
        if ordered_dict:
            merged.update(ordered_dict)
        if kwargs:
            merged.update(kwargs)
        with self._lock:
            self._postfix = _fmt_postfix(merged)

    def set_description(self, desc: Optional[str] = None, refresh: bool = True) -> None:
        if desc is not None:
            with self._lock:
                self.desc = desc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._heartbeat_stop.set()
            # Guaranteed final line (100% when total is known).
            self._maybe_emit(force=True)
        if self._heartbeat_thread is not None and self._heartbeat_thread is not threading.current_thread():
            self._heartbeat_thread.join()

    def refresh(self) -> None:
        self._maybe_emit()

    def __iter__(self):
        if self.iterable is None:
            return
        try:
            for obj in self.iterable:
                yield obj
                self.update(1)
        finally:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class tqdm:
    """Drop-in tqdm replacement: real tqdm on a TTY, log-safe fallback otherwise."""

    def __init__(self, iterable: Optional[Iterable] = None, *args: Any, **kwargs: Any):
        if _use_real_tqdm():
            self._impl = _std_tqdm(iterable, *args, **kwargs)
            self._real = True
        else:
            # Positional beyond `iterable` is not used anywhere in the SkyRL call
            # sites; pull the fields we render from kwargs.
            self._impl = _LoggingProgress(
                iterable=iterable,
                total=kwargs.get("total"),
                initial=kwargs.get("initial", 0),
                desc=kwargs.get("desc"),
                disable=kwargs.get("disable", False),
            )
            self._real = False

    # proxy the surface we use
    def update(self, n: int = 1):
        return self._impl.update(n)

    def set_postfix(self, *args: Any, **kwargs: Any):
        return self._impl.set_postfix(*args, **kwargs)

    def set_description(self, *args: Any, **kwargs: Any):
        return self._impl.set_description(*args, **kwargs)

    def close(self):
        return self._impl.close()

    def refresh(self):
        return self._impl.refresh()

    def __iter__(self):
        return iter(self._impl)

    def __enter__(self):
        self._impl.__enter__()
        return self

    def __exit__(self, *exc):
        return self._impl.__exit__(*exc)

    def __getattr__(self, item):
        # Fallback for any tqdm attribute we didn't explicitly proxy (real-tqdm mode).
        return getattr(self._impl, item)

    # --- async gather (drop-in for tqdm.asyncio.tqdm.gather) ---------------
    @classmethod
    async def gather(cls, *fs: Any, **kwargs: Any):
        if _use_real_tqdm():
            return await _std_atqdm.gather(*fs, **kwargs)

        disable = kwargs.get("disable", False)
        if disable:
            return await asyncio.gather(*fs)

        total = kwargs.get("total")
        if total is None:
            total = len(fs)
        pbar = _LoggingProgress(
            total=total,
            initial=0,
            desc=kwargs.get("desc") or "gather",
            disable=disable,
        )

        results: list = [None] * len(fs)

        async def _run(idx: int, coro: Any):
            res = await coro
            return idx, res

        tasks = [asyncio.ensure_future(_run(i, f)) for i, f in enumerate(fs)]
        try:
            for fut in asyncio.as_completed(tasks):
                idx, res = await fut
                results[idx] = res
                pbar.update(1)
        finally:
            pbar.close()
        return results


# convenience alias mirroring `from tqdm.asyncio import tqdm`
tqdm_asyncio = tqdm

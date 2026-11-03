"""CUDA allocator samples and interval peaks around a policy worker's phases."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

import torch
from loguru import logger

from skyrl_train.telemetry import WORKER_ROLE, record_event


# Allocator peaks are per device. Each busy device maps to whether another span overlapped its owner.
_busy: dict[int, bool] = {}
_busy_lock = Lock()


class LearnerCudaMetrics:
    """Record CUDA allocator samples and interval peaks around worker phases."""

    def __init__(self, *, enabled: bool, rank: int) -> None:
        self.enabled = enabled
        self._rank = rank
        self._device: int | None = None
        self._identity: dict[str, str] = {}

    def _identified_device(self) -> int:
        if self._device is None:
            device = torch.cuda.current_device()
            allocator_backend = torch.cuda.get_allocator_backend()
            if allocator_backend != "native":
                raise RuntimeError(f"learner allocator peaks require native CUDA allocator, got {allocator_backend}")
            self._identity = {
                "backend": "megatron",
                "role": WORKER_ROLE,
                "worker_role": "policy",
                "rank": str(self._rank),
                "cuda_device": str(device),
                "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
                "allocator_backend": allocator_backend,
            }
            self._device = device
        return self._device

    def _record(self, *, phase: str, boundary: str, outcome: str, step: int | None, overlapped: bool = False) -> None:
        device = self._identified_device()
        stats = torch.cuda.memory_stats(device)
        free, total = torch.cuda.mem_get_info(device)
        fields = {
            "allocated_bytes": stats["allocated_bytes.all.current"],
            "reserved_bytes": stats["reserved_bytes.all.current"],
            "device_free_bytes": free,
            "device_total_bytes": total,
        }
        if boundary == "exit" and not overlapped:
            fields.update(
                peak_allocated_bytes=stats["allocated_bytes.all.peak"],
                peak_reserved_bytes=stats["reserved_bytes.all.peak"],
            )
        attributes = {**self._identity, "phase": phase, "boundary": boundary, "outcome": outcome}
        if step is not None:
            attributes["step"] = str(step)
        if overlapped:
            attributes["scope_overlap"] = "true"
        record_event("cuda_memory_observation", fields, attributes=attributes)

    def _disable(self, phase: str, error: Exception) -> None:
        self.enabled = False
        logger.warning("Disabling learner CUDA memory observations after phase {} failed: {}", phase, error)

    @contextmanager
    def span(self, phase: str, *, step: int | None) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        owner = False
        try:
            device = self._identified_device()
            with _busy_lock:
                owner = device not in _busy
                _busy[device] = not owner
            if owner:
                torch.cuda.reset_peak_memory_stats(device)
                self._record(phase=phase, boundary="enter", outcome="started", step=step)
        except Exception as error:
            self._disable(phase, error)
        outcome = "success"
        try:
            yield
        except BaseException:
            outcome = "failure"
            raise
        finally:
            if owner:
                # Holding the lock keeps a new owner from resetting the peaks before this exit reads them.
                with _busy_lock:
                    try:
                        if self.enabled:
                            self._record(
                                phase=phase, boundary="exit", outcome=outcome, step=step, overlapped=_busy[device]
                            )
                    except Exception as error:
                        self._disable(phase, error)
                    finally:
                        del _busy[device]

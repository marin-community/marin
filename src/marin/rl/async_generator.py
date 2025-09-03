"""Async batch generator that mirrors env.fetch() into a local queue.

This thin wrapper builds the environment (pull-based) and repeatedly calls
``env.fetch(min_count=1, timeout=...)`` on a background thread, buffering up to
``queue_size`` batches locally for a same-process consumer.
"""

from typing import Optional
import queue
import threading
import time

import ray

from .datatypes import Rollout, InferenceEndpoint
from .config import AbstractEnvConfig


class AsyncGenerator:

    def __init__(
        self,
        env_cfg: AbstractEnvConfig,
        inference: InferenceEndpoint,
        *,
        seed: int = 0,
        queue_size: int = 4,
    ):
        self._env_cfg = env_cfg
        self._inference = inference
        self._seed = seed

        self._batch_queue: queue.Queue[list[Rollout]] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._paused = False
        self._worker_thread: threading.Thread | None = None
        self._started = False

        self._env_actor = None  # Ray env actor

    def start(self):
        """Starts Ray env and begins buffering batches in a thread."""
        if self._started:
            raise RuntimeError("AsyncGenerator is already started.")

        # Build the pull-based env actor (rollout_sink arg is ignored)
        self._env_actor = self._env_cfg.build(self._inference, self._seed)

        # Spawn worker thread to shuttle from Ray buffer -> local queue
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AsyncGeneratorWorker",
            daemon=True,
        )
        self._worker_thread.start()
        self._started = True

    def stop(self):
        """Stops the worker and asks the env to shutdown."""
        if not self._started:
            raise RuntimeError("AsyncGenerator is not started.")

        self._stop_event.set()
        # Best-effort: tell env to shutdown
        try:
            if self._env_actor is not None:
                self._env_actor.shutdown.remote()  # fire-and-forget
        except Exception:
            pass

        if self._worker_thread:
            self._worker_thread.join()
            self._worker_thread = None
        self._started = False

    def pause(self):
        self._paused = True

    def unpause(self):
        self._paused = False

    def get(self, *, block: bool = True, timeout: Optional[float] = None) -> list[Rollout] | None:
        """Fetch the next batch from the local queue."""
        try:
            return self._batch_queue.get(block=block, timeout=timeout)  # type: ignore[no-any-return]
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _worker_loop(self):
        """Fetch batches from Ray buffer and stage up to K locally."""
        assert self._env_actor is not None

        while not self._stop_event.is_set():
            if self._paused:
                time.sleep(0.05)
                continue

            try:
                # Issue one step and wait with timeout
                fut = self._env_actor.step.remote()
                ready, _ = ray.wait([fut], timeout=0.5)
                if not ready:
                    continue
                batch = ray.get(ready[0])
            except Exception:
                # If Ray errors (e.g., actor died), backoff and retry
                time.sleep(0.1)
                continue

            # Enqueue batch, keeping at most K most recent
            if not batch:
                continue
            try:
                self._batch_queue.put_nowait(batch)
            except queue.Full:
                try:
                    _ = self._batch_queue.get_nowait()  # drop oldest
                except queue.Empty:
                    pass
                try:
                    self._batch_queue.put_nowait(batch)
                except queue.Full:
                    # If still full, drop this batch to avoid blocking
                    pass

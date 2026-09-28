"""Early release of a TPP-10 sweep behind its p=100 execution check.

The execution check exists to catch systematic failures (code pins, cache identities, compilation,
checkpointing) before the remaining grid points spend their compute. Waiting for the whole canary
run wastes the accelerator capacity available while it trains, so the remaining points are released
as soon as the canary has committed a checkpoint: from then on a late failure is recoverable from
that checkpoint and is no cheaper to discover by waiting.
"""

import contextvars
import logging
import threading
from collections.abc import Callable

import fsspec
from marin.execution.lazy import ArtifactStep, run
from marin.training.training import LevanterCheckpoint, temporary_checkpoint_base_path

logger = logging.getLogger(__name__)

CHECKPOINT_MARKER = "metadata.json"
POLL_SECONDS = 60.0


def committed_checkpoint_exists(output_path: str) -> bool:
    """Whether the run at ``output_path`` has committed a checkpoint, temporary or permanent."""
    fs, _ = fsspec.core.url_to_fs(output_path)
    bases = (temporary_checkpoint_base_path(output_path), LevanterCheckpoint(path=output_path).checkpoint_dir)
    return any(fs.glob(f"{base}/step-*/{CHECKPOINT_MARKER}") for base in bases)


def run_with_early_release(
    canary: tuple[ArtifactStep, ...],
    remaining: tuple[ArtifactStep, ...],
    *,
    release_ready: Callable[[], bool],
    poll_seconds: float = POLL_SECONDS,
    runner: Callable[..., object] = run,
) -> None:
    """Run the pending ``canary`` steps and release ``remaining`` once ``release_ready`` holds or the canary succeeded.

    The canary runs on a helper thread that inherits this thread's execution context (the Fray
    client). A canary failure before release raises without releasing anything; a failure after
    release is raised once the released steps have finished, so their results are not lost. An
    empty ``canary`` (already built) releases immediately.
    """
    if not canary:
        if remaining:
            runner(*remaining, max_concurrent=len(remaining), force_run_failed=True)
        return
    succeeded = threading.Event()
    failures: list[Exception] = []

    def check() -> None:
        try:
            runner(*canary, max_concurrent=len(canary), force_run_failed=True)
            succeeded.set()
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=contextvars.copy_context().run, args=(check,), name="tpp10-execution-check")
    thread.start()
    while True:
        thread.join(poll_seconds)
        if failures:
            raise RuntimeError("Execution check failed before release; nothing else was submitted") from failures[0]
        if succeeded.is_set() or release_ready():
            break
    try:
        if remaining:
            reason = "succeeded" if succeeded.is_set() else "committed a checkpoint"
            logger.info("Execution check %s; releasing %d remaining points", reason, len(remaining))
            runner(*remaining, max_concurrent=len(remaining), force_run_failed=True)
    finally:
        thread.join()
    if failures:
        message = "Execution check failed after release; the released points ran to completion"
        raise RuntimeError(message) from failures[0]

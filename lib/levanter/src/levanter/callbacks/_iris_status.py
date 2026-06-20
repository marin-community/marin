# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Push a human-readable training-status row to the running Iris task.

Mirrors the technique Zephyr uses (``ZephyrCoordinator._report_task_stats``):
periodically write a markdown ``(detail, summary)`` pair to the
``iris.task_status`` namespace via ``Client.report_task_status_text`` so the
current step, throughput, loss, and a link to the W&B run show up on the Iris
dashboard for the job. It is a no-op when not running inside an Iris job.
"""

import logging
from datetime import timedelta

import jax
from fray.device_flops import device_flops_for_jax_device
from iris.client import get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from rigging.timing import RateLimiter

import levanter.tracker
from levanter.callbacks._core import StepInfo
from levanter.schedule import BatchSchedule

logger = logging.getLogger(__name__)

# The Iris task-status row is a single dashboard cell; keep it short. Matches the
# cap Zephyr uses for the same namespace.
MAX_STATUS_TEXT_LENGTH = 1000


def _wandb_run_url() -> str | None:
    """Best-effort W&B run URL from the active tracker, or ``None`` if unavailable."""
    try:
        return levanter.tracker.get_tracker("wandb").run.url
    except (KeyError, RuntimeError, AttributeError):
        return None


def _format_status(
    step: int,
    total_steps: int | None,
    loss: float,
    step_duration: float,
    tokens_per_second: float | None,
    examples_per_second: float | None,
    mfu: float | None,
    wandb_url: str | None,
) -> tuple[str, str]:
    """Build the ``(detail_md, summary_md)`` markdown pair for one status push."""
    if total_steps:
        progress = f" ({100 * step / total_steps:.1f}%)"
        step_str = f"{step:,}/{total_steps:,}"
    else:
        progress = ""
        step_str = f"{step:,}"

    detail_lines = [f"**step {step_str}**{progress}", "", f"- **loss**: {loss:.4f}"]
    if tokens_per_second is not None and examples_per_second is not None:
        detail_lines.append(f"- **throughput**: {tokens_per_second:,.0f} tok/s · {examples_per_second:,.1f} ex/s")
    if mfu is not None:
        detail_lines.append(f"- **MFU**: {mfu:.1f}%")
    if total_steps and step_duration > 0.0:
        eta = timedelta(seconds=round((total_steps - step) * step_duration))
        detail_lines.append(f"- **ETA**: {eta}")
    detail_lines.append(f"- **step time**: {step_duration:.2f}s")
    if wandb_url:
        detail_lines += ["", f"[W&B run]({wandb_url})"]
    detail_md = "\n".join(detail_lines)[:MAX_STATUS_TEXT_LENGTH]

    summary_parts = [f"step {step_str}{progress}", f"loss {loss:.3f}"]
    if tokens_per_second is not None:
        summary_parts.append(f"{tokens_per_second / 1000:,.1f}k tok/s")
    summary_md = " · ".join(summary_parts)

    return detail_md, summary_md


def iris_status_reporter(
    tokens_per_example: int,
    batch_schedule: int | BatchSchedule,
    total_steps: int | None = None,
    flops_per_example: float | None = None,
    interval_seconds: float = 30.0,
):
    """Build a hook that pushes training status to the running Iris task.

    The returned callback is a no-op unless the process is running inside an Iris
    job; inside one it reports at most once per ``interval_seconds`` (and always
    on the final, ``force``-d step). It needs the same throughput inputs as
    :func:`log_performance_stats` so it can report tokens/sec and MFU.

    Args:
        tokens_per_example: Tokens per training example (sequence length).
        batch_schedule: Batch size schedule, or a fixed batch size.
        total_steps: Total planned steps, used for progress percent and ETA.
        flops_per_example: Model FLOPs per example; enables MFU reporting.
        interval_seconds: Minimum wall-clock gap between pushes.
    """
    if isinstance(batch_schedule, int):
        batch_schedule = BatchSchedule(batch_schedule)

    limiter = RateLimiter(interval_seconds=interval_seconds)

    device_count = jax.device_count()
    flops_per_device = device_flops_for_jax_device(jax.devices()[0].device_kind)
    theoretical_flops = flops_per_device * device_count if flops_per_device is not None else None

    def report(step_info: StepInfo, force: bool = False):
        ctx = get_iris_ctx()
        job_info = get_job_info()
        if ctx is None or ctx.client is None or job_info is None:
            return
        if not force and not limiter.should_run():
            return

        step = step_info.step
        step_duration = step_info.step_duration
        batch_size = batch_schedule.batch_size_at_step(step)

        tokens_per_second = examples_per_second = mfu = None
        if step_duration > 0.0:
            examples_per_second = batch_size / step_duration
            tokens_per_second = tokens_per_example * examples_per_second
            if flops_per_example is not None and theoretical_flops is not None:
                model_flops_instant = flops_per_example / step_duration * batch_size
                mfu = model_flops_instant / theoretical_flops * 100.0

        detail_md, summary_md = _format_status(
            step=step,
            total_steps=total_steps,
            loss=float(step_info.loss),
            step_duration=step_duration,
            tokens_per_second=tokens_per_second,
            examples_per_second=examples_per_second,
            mfu=mfu,
            wandb_url=_wandb_run_url(),
        )

        try:
            ctx.client.report_task_status_text(job_info.task_id, job_info.attempt_id, detail_md, summary_md)
        except Exception:
            logger.warning("Failed to report training status to Iris controller", exc_info=True)

    return report

# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import collections
import copy
import logging as pylogging
import statistics
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import jax
from fray.device_flops import device_flops_for_jax_device
from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging

import levanter.tracker
from levanter.callbacks import StepInfo
from levanter.schedule import BatchSchedule
from levanter.tracker import log_optimizer_hyperparams
from levanter.utils.jax_utils import jnp_to_python

logger = pylogging.getLogger(__name__)


@dataclass
class InstantThroughput:
    """Per-step throughput derived from a single step's wall-clock duration.

    Fields are ``None`` when they cannot be computed: all rates require a
    nonzero ``step_duration``; ``model_flops_per_second`` additionally needs
    ``flops_per_example``; ``mfu`` additionally needs the device's theoretical
    peak FLOPs.
    """

    examples_per_second: float | None = None
    tokens_per_second: float | None = None
    model_flops_per_second: float | None = None
    mfu: float | None = None


def aggregate_device_flops() -> float | None:
    """Theoretical peak FLOPs summed over all local JAX devices, or ``None`` if unknown."""
    flops_per_device = device_flops_for_jax_device(jax.devices()[0].device_kind)
    if flops_per_device is None:
        return None
    return flops_per_device * jax.device_count()


def compute_instant_throughput(
    batch_size: int,
    step_duration: float,
    tokens_per_example: int,
    flops_per_example: float | None = None,
    theoretical_flops: float | None = None,
) -> InstantThroughput:
    """Compute examples/sec, tokens/sec, model FLOPs/sec, and MFU for one step."""
    if step_duration <= 0.0:
        return InstantThroughput()
    examples_per_second = batch_size / step_duration
    tokens_per_second = tokens_per_example * examples_per_second
    model_flops_per_second = mfu = None
    if flops_per_example is not None:
        model_flops_per_second = flops_per_example / step_duration * batch_size
        if theoretical_flops is not None:
            mfu = model_flops_per_second / theoretical_flops * 100.0
    return InstantThroughput(examples_per_second, tokens_per_second, model_flops_per_second, mfu)


def log_step_info(total_steps: Optional[int], batch_schedule: Optional[BatchSchedule] = None):
    def log_step_info_inner(step: StepInfo):
        metrics = {"train/loss": step.loss, "global_step": step.step}
        if total_steps:
            if batch_schedule is not None:
                total_examples = batch_schedule.global_data_offset_by_step(total_steps)
                current_examples = batch_schedule.global_data_offset_by_step(step.step + 1)
                metrics["run_progress"] = current_examples / total_examples
            else:
                metrics["run_progress"] = step.step / total_steps
        log_optimizer_hyperparams(step.opt_state, step=step.step, prefix="optim")
        levanter.tracker.log(metrics, step=step.step)

    return log_step_info_inner


def log_performance_stats(
    tokens_per_example: int,
    batch_schedule: int | BatchSchedule,
    flops_per_example: Optional[float] = None,
    prefix: Optional[str] = "throughput",
):
    if isinstance(batch_schedule, int):
        batch_schedule = BatchSchedule(batch_schedule)

    def wrap_key(key):
        if prefix:
            return f"{prefix}/{key}"
        return key

    device_count = jax.device_count()
    device = jax.devices()[0]

    flops_per_device = device_flops_for_jax_device(device.device_kind)
    theoretical_flops = flops_per_device * device_count if flops_per_device is not None else None
    levanter.tracker.log_summary(
        {
            wrap_key("device_kind"): device.device_kind,
        }
    )

    if flops_per_device is not None:
        levanter.tracker.log_summary(
            {
                wrap_key("theoretical_flops_per_device"): flops_per_device,
                wrap_key("theoretical_flops"): theoretical_flops,
            }
        )

    if flops_per_example is not None:
        levanter.tracker.log_summary({wrap_key("flops_per_example"): flops_per_example})

    # Accumulate MFU samples over a trailing window for robust distribution stats.
    # 500 steps is large enough for stable percentiles while bounded in memory.
    mfu_window: collections.deque[float] = collections.deque(maxlen=500)

    def log_performance_stats(step_info: StepInfo):
        dict_to_log: dict[str, float | int] = {}

        # log these totals because it's useful for comparing different seqlens, batch sizes, etc
        # TODO: if we add seqlen schedules this will get even more complex
        this_batch_size = batch_schedule.batch_size_at_step(step_info.step)
        total_examples = batch_schedule.global_data_offset_by_step(step_info.step + 1)
        total_tokens = tokens_per_example * total_examples
        dict_to_log["total_tokens"] = total_tokens

        if flops_per_example:
            total_flops = flops_per_example * total_examples
            dict_to_log["total_gflops"] = total_flops / 1e9

        throughput = compute_instant_throughput(
            this_batch_size, step_info.step_duration, tokens_per_example, flops_per_example, theoretical_flops
        )
        if throughput.examples_per_second is not None and throughput.tokens_per_second is not None:
            dict_to_log["examples_per_second"] = throughput.examples_per_second
            dict_to_log["tokens_per_second"] = throughput.tokens_per_second
            dict_to_log["duration"] = step_info.step_duration
        if throughput.model_flops_per_second is not None:
            dict_to_log["gflops_per_second"] = throughput.model_flops_per_second / 1e9
        if throughput.mfu is not None:
            dict_to_log["mfu"] = throughput.mfu
            mfu_window.append(throughput.mfu)

        dict_to_log = {wrap_key(k): v for k, v in dict_to_log.items()}
        levanter.tracker.log(dict_to_log, step=step_info.step)

        if len(mfu_window) > 0:
            n = len(mfu_window)
            mean = statistics.mean(mfu_window)
            med = statistics.median(mfu_window)
            if n >= 2:
                deciles = statistics.quantiles(mfu_window, n=10)
                p10, p90 = deciles[0], deciles[8]
                sd = statistics.stdev(mfu_window)
            else:
                p10 = p90 = mfu_window[0]
                sd = 0.0
            levanter.tracker.log_summary(
                {
                    wrap_key("p10_mfu"): p10,
                    wrap_key("p50_mfu"): med,
                    wrap_key("p90_mfu"): p90,
                    wrap_key("mean_mfu"): mean,
                    wrap_key("stddev_mfu"): sd,
                    wrap_key("mfu_sample_count"): n,
                }
            )

    return log_performance_stats


def pbar_logger(iterable=None, desc="train", **tqdm_mkwargs):
    kwargs = copy.copy(tqdm_mkwargs)
    if "desc" not in kwargs:
        kwargs["desc"] = desc
    if "iterable" not in kwargs:
        kwargs["iterable"] = iterable

    _tqdm_logging_one_time_setup()
    pbar = tqdm(**kwargs)

    def update_pbar(step: StepInfo):
        pbar.update(step.next_step - pbar.n)
        pbar.set_postfix(loss=jnp_to_python(step.loss))

    return update_pbar


_did_tqdm_logging_one_time_setup = False


def _tqdm_logging_one_time_setup():
    global _did_tqdm_logging_one_time_setup
    if _did_tqdm_logging_one_time_setup:
        return
    _did_tqdm_logging_one_time_setup = True
    tqdm_logging.set_log_rate(timedelta(seconds=60))

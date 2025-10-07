# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import logging as pylogging
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Optional

import jax
from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging

import levanter.tracker
from levanter.callbacks import StepInfo
from levanter.data import AsyncDataset
from levanter.schedule import BatchSchedule
from levanter.tracker import log_optimizer_hyperparams
from levanter.utils import flop_utils
from levanter.utils.jax_utils import jnp_to_python


logger = pylogging.getLogger(__name__)


def log_epoch_progress(total_tokens_future, tokens_per_example, batch_size, max_epochs: Optional[int] = None):
    total_tokens = None

    def log_epoch(step_info: StepInfo):
        nonlocal total_tokens
        if total_tokens is None:
            if not total_tokens_future.done():
                if step_info.step % 1000 == 0:
                    logger.info("Dataset not finished. Can't compute epochs.")
                return  # We don't have the total tokens yet, so we can't calculate epoch
            total_tokens = total_tokens_future.result()

        # Get the total processed tokens from the metrics logged by log_performance_stats
        processed_tokens = tokens_per_example * batch_size * step_info.step

        # If we're doing multiple epochs, adjust the denominator
        total_tokens_for_epochs = total_tokens * max_epochs if max_epochs else total_tokens
        current_epoch = processed_tokens / total_tokens_for_epochs

        levanter.tracker.log({"train/current_epoch": current_epoch}, step=step_info.step)

    return log_epoch


def get_total_dataset_tokens(ds: AsyncDataset, seq_length: int):
    def log_length():
        # If ds.async_len() is the only option, run it in an event loop inside the thread
        import asyncio

        async def compute_length():
            length = await ds.dataset.async_len()
            return length

        # Run the async function synchronously in this thread
        length = asyncio.run(compute_length())
        total_tokens = length * seq_length
        levanter.tracker.log_summary({"dataset/total_tokens": total_tokens})
        return total_tokens

    # Create a ThreadPoolExecutor with a single worker thread
    executor = ThreadPoolExecutor(max_workers=1)
    # Submit the log_length function to be executed in a separate thread
    future = executor.submit(log_length)
    return future


def log_step_info(total_steps: Optional[int]):
    def log_step_info_inner(step: StepInfo):
        metrics = {"train/loss": step.loss, "global_step": step.step}
        if total_steps:
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

    flops_per_device = flop_utils.device_hardware_flops(device)
    levanter.tracker.log_summary(
        {
            wrap_key("device_kind"): device.device_kind,
        }
    )

    if flops_per_device is not None:
        theoretical_flops = flops_per_device * device_count

        levanter.tracker.log_summary(
            {
                wrap_key("theoretical_flops_per_device"): flops_per_device,
                wrap_key("theoretical_flops"): theoretical_flops,
            }
        )

    if flops_per_example is not None:
        levanter.tracker.log_summary({wrap_key("flops_per_example"): flops_per_example})

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

        if step_info.step_duration != 0.0:
            dict_to_log["examples_per_second"] = float(this_batch_size) / step_info.step_duration
            dict_to_log["tokens_per_second"] = float(tokens_per_example) / step_info.step_duration * this_batch_size
            dict_to_log["duration"] = step_info.step_duration

            if flops_per_example is not None:
                model_flops_instant = flops_per_example / step_info.step_duration * this_batch_size
                dict_to_log["gflops_per_second"] = model_flops_instant / 1e9

                if flops_per_device is not None:
                    mfu_instant = model_flops_instant / theoretical_flops * 100.0
                    dict_to_log["mfu"] = mfu_instant

        dict_to_log = {wrap_key(k): v for k, v in dict_to_log.items()}
        levanter.tracker.log(dict_to_log, step=step_info.step)

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

# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Optional

import wandb

import jax
from tqdm_loggable.auto import tqdm

import levanter.tracker
from levanter.callbacks._core import Callback, CBInfo, JitCallback, LambdaCallback, StepInfo
from levanter.callbacks._metrics import (
    _tqdm_logging_one_time_setup,
    log_performance_stats,
    log_step_info,
    logger,
    pbar_logger,
)
from levanter.callbacks._iris_status import iris_status_reporter
from levanter.callbacks.state_adapter import CallbackStateView, StateCallbackRunner
from levanter.callbacks.profiler import profile, profile_ctx
from levanter.data import DataLoader
from levanter.metrics import LossFunctionWithMetrics, unwrap_metrics
from levanter.metrics import fold as fold_metric
from levanter.tracker.wandb import WandbConfig
from levanter.utils.logging import save_xla_dumps_to_wandb


def eval_loss_loop(
    loss_fn: LossFunctionWithMetrics, model, dataset, max_batches: Optional[int] = None, name: Optional[str] = None
) -> tuple[float, dict[str, float]]:

    total_loss = 0.0
    total_load_time = 0.0
    total_loss_time = 0.0
    accumulated_metrics: dict = {}
    n = 0

    desc = f"eval {name}" if name is not None else "eval"

    _tqdm_logging_one_time_setup()
    pbar = tqdm(dataset, desc=desc, position=1, leave=False, total=max_batches)

    iter_ = iter(pbar)
    with jax.named_scope(desc):
        while True:
            time_in = time.time()
            batch = next(iter_, None)
            if batch is None:
                break
            load_time = time.time() - time_in
            total_load_time += load_time

            # loss_fn returns (loss, wrapped_metrics) where wrapped_metrics is Dict[str, Metric]
            loss, wrapped_metrics = loss_fn(model, batch)

            # Use fold() to accumulate Metric objects
            for key, metric in wrapped_metrics.items():
                if key not in accumulated_metrics:
                    accumulated_metrics[key] = metric
                else:
                    accumulated_metrics[key] = fold_metric(accumulated_metrics[key], metric)

            total_loss += loss.item()
            n += 1
            loss_time = time.time() - time_in - load_time
            total_loss_time += loss_time

            pbar.set_postfix(loss=total_loss / n)

            if max_batches is not None and n >= max_batches:
                break

    if n > 0:
        total_loss /= n

    plain_metrics = unwrap_metrics(accumulated_metrics)
    plain_metrics["eval/timing/load_time"] = total_load_time
    plain_metrics["eval/timing/loss_time"] = total_loss_time
    plain_metrics["eval/timing/num_batches"] = float(n)
    return total_loss, plain_metrics


def compute_validation_loss(
    loss_fn: LossFunctionWithMetrics,
    dataset: DataLoader,
    max_batches: Optional[int] = None,
    name: Optional[str] = None,
):
    def compute_loss(info: StepInfo):
        loss, metrics = eval_loss_loop(loss_fn, info.eval_model, dataset, max_batches=max_batches, name=name)

        prefix = "eval"
        if name:
            prefix += "/" + name

        # Log loss and metrics. eval_loss_loop already namespaces its loop-timing
        # keys under "eval/"; strip it so this prefix (e.g. "eval/<name>") is applied
        # once, yielding "eval/<name>/timing/..." instead of "eval/eval/timing/...".
        to_log = {f"{prefix}/loss": loss}
        to_log.update({f"{prefix}/{k.removeprefix('eval/')}": v for k, v in metrics.items()})
        levanter.tracker.log(to_log, step=info.step)

        if name:
            logger.info(f"{name} validation loss: {loss:.3f}")
        else:
            logger.info(f"validation loss: {loss:.3f}")

        return loss

    return compute_loss


def wandb_xla_logger(config: WandbConfig):
    last_mtime = wandb.run and wandb.run.start_time or time.time()

    def log_xla_to_wandb(step: StepInfo):
        nonlocal last_mtime
        save_xla_dumps_to_wandb(last_mtime)
        # update time to now
        last_mtime = time.time()

    if config.save_xla_dumps:
        return log_xla_to_wandb
    else:
        return lambda x: None


__all__ = [
    "eval_loss_loop",
    "compute_validation_loss",
    "wandb_xla_logger",
    "profile",
    "profile_ctx",
    "Callback",
    "CBInfo",
    "JitCallback",
    "LambdaCallback",
    "StepInfo",
    "log_performance_stats",
    "iris_status_reporter",
    "log_step_info",
    "pbar_logger",
    "CallbackStateView",
    "StateCallbackRunner",
]

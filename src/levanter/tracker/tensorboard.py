# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import numbers
import os
import typing
from dataclasses import dataclass
from typing import Any

import jax
from iris.marin_fs import url_to_fs
import numpy as np

from levanter.tracker import Tracker, TrackerConfig
from levanter.tracker.histogram import Histogram

pylogger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from tensorboardX import SummaryWriter


def _is_scalar(v) -> bool:
    return isinstance(v, numbers.Number) or (isinstance(v, np.ndarray | jax.Array) and v.ndim == 0)


class TensorboardTracker(Tracker):
    name: str = "tensorboard"

    def __init__(self, writer: "SummaryWriter"):
        self.writer = writer

    def log_hyperparameters(self, hparams: typing.Mapping[str, Any]):
        self.writer.add_hparams(hparams, {"dummy": 0})

    def log(self, metrics: typing.Mapping[str, Any], *, step, commit=None):
        # Don't log metrics from non-primary workers.
        if jax.process_index() != 0:
            return

        del commit
        metrics = _flatten_nested_dict(metrics)
        for k, value in metrics.items():
            try:
                if isinstance(value, jax.Array):
                    if value.ndim == 0:
                        value = value.item()
                    else:
                        value = np.array(value)

                if isinstance(value, Histogram):
                    num = value.num
                    if hasattr(num, "item"):
                        num = num.item()
                    self.writer.add_histogram_raw(
                        k,
                        min=value.min.item(),
                        max=value.max.item(),
                        num=num,
                        sum=value.sum.item(),
                        sum_squares=value.sum_squares.item(),
                        bucket_limits=np.array(value.bucket_limits).tolist(),
                        bucket_counts=np.concatenate([[0], np.array(value.bucket_counts)]).tolist(),
                        global_step=step,
                    )
                elif isinstance(value, str):
                    self.writer.add_text(k, value)
                elif isinstance(value, np.ndarray):
                    if np.dtype(value.dtype).kind in ("U", "S", "O"):
                        self.writer.add_text(k, str(value))
                    elif np.issubdtype(value.dtype, np.number):
                        if value.ndim == 0:
                            self.writer.add_scalar(k, value.item(), global_step=step)
                        else:
                            self.writer.add_histogram(k, value.ravel(), global_step=step)
                    else:
                        pylogger.error(f"Unsupported metric type: {type(value)} for key {k}")
                else:
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(k, value, global_step=step)
                    else:
                        self.writer.add_text(k, str(value), global_step=step)
            except Exception:
                pylogger.exception(f"Error logging metric {k} with value {value}")

    def log_summary(self, metrics: dict[str, Any]):
        for k, v in metrics.items():
            if _is_scalar(v):
                self.writer.add_scalar(k, v, global_step=None)
            elif isinstance(v, str):
                self.writer.add_text(k, v, global_step=None)
            else:
                pylogger.error(f"Unsupported metric type: {type(v)} for key {k}")

    def log_artifact(self, artifact_path, *, name: str | None = None, type: str | None = None):
        log_path = self.writer.logdir
        # sync the artifact to the logdir via fsspec
        try:
            fs, fs_path = url_to_fs(log_path)
            fs.put(artifact_path, os.path.join(fs_path, name or os.path.basename(artifact_path)), recursive=True)
        except Exception:
            pylogger.exception(f"Error logging artifact {artifact_path} to {log_path}")
            return

    def finish(self):
        self.writer.close()


@TrackerConfig.register_subclass("tensorboard")
@dataclass
class TensorboardConfig(TrackerConfig):
    logdir: str = "tblogs"
    comment: str | None = ""
    purge_step: int | None = None
    max_queue: int | None = 10
    flush_secs: int | None = 120
    filename_suffix: str | None = ""
    write_to_disk: bool | None = True

    def init(self, run_id: str | None) -> TensorboardTracker:
        dir_to_write = self.logdir
        if run_id is not None:
            dir_to_write = os.path.join(dir_to_write, run_id)

        pylogger.info(f"Writing Tensorboard logs to {dir_to_write}")

        from tensorboardX import SummaryWriter

        writer = SummaryWriter(
            dir_to_write,
            comment=self.comment,
            purge_step=self.purge_step,
            max_queue=self.max_queue,
            flush_secs=self.flush_secs,
            filename_suffix=self.filename_suffix,
            write_to_disk=self.write_to_disk,
        )

        return TensorboardTracker(writer)


def _flatten_nested_dict(d):
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in _flatten_nested_dict(value).items():
                    yield key + "/" + subkey, subvalue
            else:
                yield key, value

    return dict(items())

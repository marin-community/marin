# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging as pylogging
import os
import time
from pathlib import Path
from typing import Iterable, Iterator, TypeVar, Union

import jax
import rigging.log_setup as iris_logging

pylogger = pylogging.getLogger(__name__)

T = TypeVar("T")


def init_logging(log_dir: Union[str, Path], run_id: str, level: int = pylogging.INFO) -> None:
    """
    Initialize logging with iris (stderr + ring buffer) plus a levanter-format file log.

    The file log uses a verbose format that includes process index, filename, and line number,
    which is useful for post-hoc debugging of distributed runs.

    :param log_dir: Directory for writing the log file
    :param run_id: Used as the log file basename
    :param level: Default logging level
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{run_id}.log"

    # Set up iris logging (stderr handler + ring buffer).
    iris_logging.configure_logging(level)

    # Add a file handler with levanter's verbose format, which includes process index,
    # source location, and level — useful when aggregating logs from multiple processes.
    process_index = jax.process_index()
    log_format = f"%(asctime)s - {process_index} - %(name)s - %(filename)s:%(lineno)d - %(levelname)s :: %(message)s"
    date_format = "%Y-%m-%dT%H:%M:%S"

    file_handler = pylogging.FileHandler(path, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(pylogging.Formatter(fmt=log_format, datefmt=date_format))
    pylogging.getLogger().addHandler(file_handler)

    pylogging.getLogger("levanter").setLevel(level)
    pylogging.getLogger("tqdm_loggable").setLevel(level)

    silence_transformer_nag()


class LoadingTimeTrackerIterator(Iterator[T]):
    def __init__(self, items: Iterable[T]):
        self.total_time = 0.0
        self.this_load_time = 0.0
        self.items = iter(items)

    def __next__(self) -> T:
        start = time.perf_counter()
        item = next(self.items)
        self.this_load_time = time.perf_counter() - start
        self.total_time += self.this_load_time
        return item


def silence_transformer_nag():
    # this is a hack to silence the transformers' "None of PyTorch, TensorFlow 2.0 or Flax have been found..." thing
    # which is annoying and not useful
    # Often we won't call this early enough, but it helps with multiprocessing stuff
    if os.getenv("TRANSFORMERS_VERBOSITY") is None:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

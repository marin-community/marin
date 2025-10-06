# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging as pylogging
import os
import time
from pathlib import Path
from typing import Iterable, Iterator, List, TypeVar, Union

import jax


pylogger = pylogging.getLogger(__name__)

T = TypeVar("T")


def init_logging(log_dir: Union[str, Path], run_id: str, level: int = pylogging.INFO) -> None:
    """
    Initialize logging.Logger with the appropriate name, console, and file handlers.

    :param path: Path for writing log file
    :param level: Default logging level
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{run_id}.log"

    process_index = jax.process_index()
    log_format = f"%(asctime)s - {process_index} - %(name)s - %(filename)s:%(lineno)d - %(levelname)s :: %(message)s"
    # use ISO 8601 format for timestamps, except no TZ, because who cares
    date_format = "%Y-%m-%dT%H:%M:%S"

    handlers: List[pylogging.Handler] = [pylogging.FileHandler(path, mode="a"), pylogging.StreamHandler()]

    # Create Root Logger w/ Base Formatting
    pylogging.basicConfig(format=log_format, datefmt=date_format, handlers=handlers, force=True)
    pylogging.getLogger("levanter").setLevel(level)

    # Silence Transformers' "None of PyTorch, TensorFlow 2.0 or Flax have been found..." thing
    silence_transformer_nag()


def save_xla_dumps_to_wandb(initial_time: float):
    import os

    from levanter.tracker.wandb import is_wandb_available

    if not is_wandb_available():
        pylogger.warning("Wandb is not available, so we can't save XLA dumps")
        return

    import wandb

    # attempt to parse xla_flags to see if we're dumping assembly files
    flags = os.getenv("XLA_FLAGS", None)
    if flags is not None and "xla_dump_to" in flags:
        # parse the path
        # this isn't robust to quotes
        path = flags.split("xla_dump_to=")[1].split(" ")[0]
        pylogger.info(f"Found xla_dump_to={path}, logging to wandb")
        if wandb.run:
            # only want to save the files that were generated during this run
            # XLA_FLAGS has to be set before the first jax call, so we can't just set it in the middle of the run
            # which means it's a pain to control where the files are saved
            # so we just save all the files that were generated during this run
            # this is a bit hacky, but it works
            def include_file(path: str):
                return os.path.getmtime(path) > initial_time

            wandb.run.log_code(root=path, name="xla_dumps", include_fn=include_file)
    else:
        pylogger.warning("XLA_FLAGS is not set to dump to a path, so we can't save the dumps to wandb")


class LoadingTimeTrackerIterator(Iterator[T]):
    def __init__(self, items: Iterable[T]):
        self.total_time = 0.0
        start = time.perf_counter()
        self.items = iter(items)
        self.total_time += time.perf_counter() - start
        self.this_load_time = 0.0

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

    import transformers  # noqa: F401

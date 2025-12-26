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
    pylogging.getLogger("levanter.tensorstore_serialization").setLevel(pylogging.ERROR)
    pylogging.getLogger("tqdm_loggable").setLevel(level)

    # Silence Transformers' "None of PyTorch, TensorFlow 2.0 or Flax have been found..." thing
    silence_transformer_nag()


def save_xla_dumps_to_wandb(initial_time: float):
    import os
    import traceback
    from pathlib import Path as _Path

    from levanter.tracker.wandb import is_wandb_available

    # Debug: show initial time and current flags
    flags_env = os.getenv("XLA_FLAGS", None)
    msg = f"[XLA_DUMP] initial_time={initial_time:.3f} XLA_FLAGS={flags_env}"
    pylogger.info(msg)
    try:
        print(msg, flush=True)
    except Exception:
        pass

    if not is_wandb_available():
        pylogger.warning("[XLA_DUMP] Wandb unavailable or run not initialized; skipping upload")
        return

    import wandb

    # attempt to parse xla_flags to see if we're dumping assembly files
    flags = flags_env
    if flags is not None and "xla_dump_to" in flags:
        # parse the path (not robust to quotes)
        try:
            path = flags.split("xla_dump_to=")[1].split(" ")[0]
        except Exception:
            pylogger.warning("[XLA_DUMP] Failed to parse xla_dump_to from XLA_FLAGS; skipping upload")
            return

        _msg = f"[XLA_DUMP] Parsed xla_dump_to={path}"
        pylogger.info(_msg)
        try:
            print(_msg, flush=True)
        except Exception:
            pass

        if not path:
            pylogger.warning("[XLA_DUMP] Empty xla_dump_to path; skipping upload")
            return

        # Pre-flight checks and debug listing
        root = _Path(path)
        if not root.exists():
            pylogger.warning(f"[XLA_DUMP] Dump path does not exist: {root}")
            return
        if not root.is_dir():
            pylogger.warning(f"[XLA_DUMP] Dump path is not a directory: {root}")
            return

        # Count files and those newer than initial_time
        total_files = 0
        recent_files = 0
        latest_mtime = 0.0
        sample_list = []
        try:
            for p in root.rglob("*"):
                if p.is_file():
                    total_files += 1
                    m = p.stat().st_mtime
                    if m > latest_mtime:
                        latest_mtime = m
                    if m > initial_time:
                        recent_files += 1
                        if len(sample_list) < 10:
                            sample_list.append(str(p))
        except Exception as e:
            pylogger.warning(f"[XLA_DUMP] Failed to scan dump dir {root}: {e}")

        _msg = (
            f"[XLA_DUMP] Found total_files={total_files}, recent_files={recent_files}, "
            f"latest_mtime={latest_mtime:.3f}"
        )
        pylogger.info(_msg)
        try:
            print(_msg, flush=True)
        except Exception:
            pass
        if sample_list:
            pylogger.info("[XLA_DUMP] Sample recent files:\n" + "\n".join(sample_list))

        if wandb.run:
            def include_file(p: str):
                try:
                    return os.path.getmtime(p) > initial_time
                except FileNotFoundError:
                    return False
                except Exception:
                    return False

            try:
                _msg = f"[XLA_DUMP] Uploading to W&B via log_code(root={root}, name='xla_dumps')"
                pylogger.info(_msg)
                try:
                    print(_msg, flush=True)
                except Exception:
                    pass
                wandb.run.log_code(root=str(root), name="xla_dumps", include_fn=include_file)
                pylogger.info("[XLA_DUMP] W&B log_code request submitted.")
                try:
                    print("[XLA_DUMP] W&B log_code request submitted.", flush=True)
                except Exception:
                    pass
            except Exception:
                pylogger.error("[XLA_DUMP] Exception during W&B upload:\n" + traceback.format_exc())
        else:
            pylogger.warning("[XLA_DUMP] No active wandb.run; skipping upload")
    else:
        pylogger.warning("[XLA_DUMP] XLA_FLAGS missing xla_dump_to; cannot upload dumps")


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

import contextlib
import os
import sys
import time
from dataclasses import dataclass


def logical_cpu_core_count():
    """Returns the number of logical CPU cores available to the process."""
    num_cpus = os.getenv("SLURM_CPUS_ON_NODE", None)
    if num_cpus is not None:
        return int(num_cpus)

    try:
        return os.cpu_count()
    except NotImplementedError:
        return 1


def logical_cpu_memory_size():
    """Returns the total amount of memory in GB available to the process or logical memory for SLURM."""
    mem = os.getenv("SLURM_MEM_PER_NODE", None)
    tasks = os.getenv("SLURM_NTASKS_PER_NODE", None)
    if mem is not None and tasks is not None:
        return float(mem) / int(tasks) / 1024.0  # MEM_PER_NODE is in MB

    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return total / (1024.0**3)
    except ValueError:
        import psutil

        return psutil.virtual_memory().total / (1024.0**3)


def non_caching_cycle(iterable):
    """Like itertools.cycle, but doesn't cache the iterable."""
    while True:
        yield from iterable


# https://stackoverflow.com/a/58336722/1736826 CC-BY-SA 4.0
def dataclass_with_default_init(_cls=None, *args, **kwargs):
    def wrap(cls):
        # Save the current __init__ and remove it so dataclass will
        # create the default __init__.
        user_init = getattr(cls, "__init__")
        delattr(cls, "__init__")

        # let dataclass process our class.
        result = dataclass(cls, *args, **kwargs)

        # Restore the user's __init__ save the default init to __default_init__.
        setattr(result, "__default_init__", result.__init__)
        setattr(result, "__init__", user_init)

        # Just in case that dataclass will return a new instance,
        # (currently, does not happen), restore cls's __init__.
        if result is not cls:
            setattr(cls, "__init__", user_init)

        return result

    # Support both dataclass_with_default_init() and dataclass_with_default_init
    if _cls is None:
        return wrap
    else:
        return wrap(_cls)


def actual_sizeof(obj):
    """similar to sys.getsizeof, but recurses into dicts and lists and other objects"""
    seen = set()
    size = 0
    objects = [obj]
    while objects:
        need_to_see = []
        for obj in objects:
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            size += sys.getsizeof(obj)
            if isinstance(obj, dict):
                need_to_see.extend(obj.values())
            elif hasattr(obj, "__dict__"):
                need_to_see.extend(obj.__dict__.values())
            elif isinstance(obj, (list, tuple, set, frozenset)):
                need_to_see.extend(obj)
        objects = need_to_see
    return size


class Stopwatch:
    """Resumable stop watch for tracking time per call"""

    def __init__(self):
        self._start_time = time.time()
        self._elapsed = 0.0
        self._n = 0

    def start(self):
        self._start_time = time.time()
        self._n += 1

    def stop(self):
        self._elapsed += time.time() - self._start_time

    def reset(self):
        self._elapsed = 0.0

    def elapsed(self):
        return self._elapsed

    def average(self):
        if self._n == 0:
            return 0.0
        return self._elapsed / self._n

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


@contextlib.contextmanager
def set_global_rng_seeds(seed):
    import numpy as np

    current_np_seed = np.random.get_state()
    np.random.seed(seed)

    import random

    current_random_seed = random.getstate()
    random.seed(seed)

    try:
        import torch

        current_torch_seed = torch.random.get_rng_state()
        torch.manual_seed(seed)
    except ImportError:
        torch = None
        current_torch_seed = None
        pass

    try:
        yield
    finally:
        np.random.set_state(current_np_seed)
        random.setstate(current_random_seed)
        if current_torch_seed is not None:
            torch.random.set_rng_state(current_torch_seed)

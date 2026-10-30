"""Clean children for native checkers that enforce address-space limits."""

from collections.abc import Callable
from multiprocessing.process import BaseProcess
from typing import Any

import numpy  # noqa: F401  # Load before the native guard disables NumPy's import-time environment writes.
from loky.backend.context import get_context


def bounded_process(*, target: Callable[..., None], args: tuple[Any, ...]) -> BaseProcess:
    # Fork inherits the persistent grader's libraries and corpus mappings. Stdlib
    # spawn reimports the CLI. Either can exceed the checker's native memory cap.
    return get_context("loky").Process(
        target=target, args=args,
        env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )

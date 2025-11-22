# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Function thunk helper for executing cloudpickled callables.

This module provides utilities for serializing and executing callables via cloudpickle.
Used by cluster implementations to support callable entrypoints uniformly across
different execution environments.

Usage as CLI:
    python -m fray.fn_thunk <fsspec-path>

Usage as library:
    >>> from fray.fn_thunk import create_thunk_entrypoint
    >>> def my_fn():
    ...     return "hello"
    >>> entrypoint = create_thunk_entrypoint(my_fn, prefix="/tmp/job")
    >>> # entrypoint is an Entrypoint(module="fray.fn_thunk", args=["/tmp/job_abc123.pkl"])

Example:
    python -m fray.fn_thunk /tmp/my_function.pkl
    python -m fray.fn_thunk gs://bucket/path/to/function.pkl
"""

import logging
import sys
import tempfile
from collections.abc import Callable
from typing import Any

import click
import cloudpickle
import fsspec

from fray.cluster.base import Entrypoint

logger = logging.getLogger(__name__)


def create_thunk_entrypoint(
    callable_fn: Callable[..., Any], prefix: str, function_args: dict[str, Any] | None = None
) -> Entrypoint:
    """Serialize a callable and its arguments to a temporary file and return an Entrypoint.

    Args:
        callable_fn: Callable to serialize
        prefix: File prefix for the pickled callable
        function_args: Keyword arguments to pass to callable

    Returns:
        Entrypoint configured to execute the callable via fray.fn_thunk
    """
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", prefix=prefix + "_", delete=False) as f:
        cloudpickle.dump((callable_fn, function_args), f, protocol=cloudpickle.DEFAULT_PROTOCOL)
        pickle_path = f.name

    return Entrypoint(binary="python", args=["-m", "fray.fn_thunk", pickle_path])


@click.command()
@click.argument("path", type=str)
def main(path: str):
    """Execute a cloudpickled callable from an fsspec-compatible path.

    Args:
        path: fsspec-compatible path to the cloudpickled callable
    """
    logging.basicConfig(
        level=logging.INFO, format="%(filename)s:%(lineno)d %(asctime)s %(levelname)s %(message)s", stream=sys.stderr
    )
    try:
        logger.info("Loading callable from %s", path)
        with fsspec.open(path, "rb") as f:
            payload = cloudpickle.load(f)
            if isinstance(payload, tuple):
                callable_fn, function_args = payload
            else:
                callable_fn = payload
                function_args = None
    except Exception as e:
        logger.error("Failed to load callable from %s: %s", path, e)
        raise

    try:
        logger.info("Calling user entrypoint %s", callable_fn)
        logger.info("About to execute callable_fn()")
        sys.stdout.flush()
        sys.stderr.flush()
        if function_args:
            result = callable_fn(**function_args)
        else:
            result = callable_fn()
        logger.info("Callable executed successfully. Result: %s", result)
        return 0
    except Exception as e:
        logger.error("Failed to run callable: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    main()

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


def create_thunk_entrypoint(callable_fn: Callable[[], Any], prefix: str | None = None) -> Entrypoint:
    """Serialize a callable to a temporary file and return an Entrypoint.

    Args:
        callable_fn: Zero-argument callable to serialize
        prefix: Optional file prefix for the pickled callable (default: uses tempfile)

    Returns:
        Entrypoint configured to execute the callable via fray.fn_thunk

    Example:
        >>> def my_job():
        ...     print("Running job")
        >>> entrypoint = create_thunk_entrypoint(my_job, prefix="/tmp/myjob")
        >>> # Returns: Entrypoint(binary="python", args=["-m", "fray.fn_thunk", "/tmp/myjob_abc123.pkl"])
    """
    if prefix:
        # Use prefix with tempfile naming for uniqueness
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", prefix=prefix + "_", delete=False) as f:
            cloudpickle.dump(callable_fn, f)
            pickle_path = f.name
    else:
        # Use default tempfile location
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            cloudpickle.dump(callable_fn, f)
            pickle_path = f.name

    return Entrypoint(binary="python", args=["-m", "fray.fn_thunk", pickle_path])


@click.command()
@click.argument("path", type=str)
def main(path: str):
    """Execute a cloudpickled callable from an fsspec-compatible path.

    Args:
        path: fsspec-compatible path to the cloudpickled callable
    """
    try:
        logger.info("Loading callable from %s", path)
        with fsspec.open(path, "rb") as f:
            callable_fn = cloudpickle.load(f)

        logger.info("Executing callable...")
        result = callable_fn()

        logger.info("Callable executed successfully. Result: %s", result)

    except FileNotFoundError:
        click.echo(f"Error: File not found: {path}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error executing function: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

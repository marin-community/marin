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

This is primarily used to convert function invocations into Ray jobs -- the Ray
"job" entrypoint only accepts a command to run, not callables.

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
from collections.abc import Callable, Sequence
from typing import Any

import click
import cloudpickle
import fsspec

from fray.cluster.base import Entrypoint

logger = logging.getLogger(__name__)


def create_thunk_entrypoint(
    callable_fn: Callable[..., Any],
    prefix: str,
    args: Sequence[Any] = (),
    kwargs: dict[str, Any] | None = None,
) -> Entrypoint:
    """Serialize a callable and its arguments to a temporary file and return an Entrypoint.

    Args:
        callable_fn: Callable to serialize
        prefix: File prefix for the pickled callable
        args: Positional arguments to pass to callable
        kwargs: Keyword arguments to pass to callable

    Returns:
        Entrypoint configured to execute the callable via fray.fn_thunk
    """
    if " " in prefix:
        raise ValueError("prefix must not contain spaces")

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", prefix=prefix + "_", delete=False) as f:
        cloudpickle.dump((callable_fn, args, kwargs or {}), f, protocol=cloudpickle.DEFAULT_PROTOCOL)
        pickle_path = f.name

    return Entrypoint.from_binary("python", ["-m", "fray.fn_thunk", pickle_path])


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
            if isinstance(payload, tuple) and len(payload) == 3:
                callable_fn, args, kwargs = payload
            elif isinstance(payload, tuple) and len(payload) == 2:
                # Legacy format: (callable, function_args_dict)
                callable_fn, kwargs = payload
                args = ()
            else:
                callable_fn = payload
                args = ()
                kwargs = {}
    except Exception as e:
        logger.error("Failed to load callable from %s: %s", path, e)
        raise

    try:
        logger.info("Calling user entrypoint %s with args=%s, kwargs=%s", callable_fn, args, kwargs)
        result = callable_fn(*args, **(kwargs or {}))
        logger.info("Callable executed successfully. Result: %s", result)
        return 0
    except Exception as e:
        logger.error("Failed to run callable: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    main()

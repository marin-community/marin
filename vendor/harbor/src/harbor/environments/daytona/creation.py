"""Isolate SDK creation: Daytona temporarily clears the process environment.

Pickle preserves Image's private build context and SDK exception attributes.
The pipes are private to a child running this same installed package; never
accept these payloads from an external caller.
"""

from __future__ import annotations

import asyncio
import pickle
import sys
from contextlib import redirect_stdout

from daytona import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    CreateSnapshotParams,
    DaytonaConfig,
)

from harbor.environments.daytona.utils import daytona_connection_pool_maxsize


async def create_in_subprocess(
    params: CreateSnapshotParams
    | CreateSandboxFromImageParams
    | CreateSandboxFromSnapshotParams,
    *,
    timeout: float,
    target: str | None = None,
    api_key: str | None = None,
) -> str:
    """Create a Daytona resource without mutating Harbor's environment."""
    launch = asyncio.create_task(
        asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import asyncio; from harbor.environments.daytona.creation import main; asyncio.run(main())",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
    )
    try:
        process = await asyncio.shield(launch)
        output, _ = await process.communicate(
            pickle.dumps((params, timeout, target, api_key))
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"Daytona creation subprocess exited with status {process.returncode}"
            )
        result = pickle.loads(output)
        if isinstance(result, Exception):
            raise result
        return result
    finally:
        # A cancellation during spawn must still collect and reap the child.
        process = await launch
        if process.returncode is None:
            process.kill()
        await process.wait()


async def main() -> None:
    params, timeout, target, api_key = pickle.loads(sys.stdin.buffer.read())
    with redirect_stdout(sys.stderr):
        try:
            config = DaytonaConfig(
                api_key=api_key,
                target=target,
                connection_pool_maxsize=daytona_connection_pool_maxsize(),
            )
            async with AsyncDaytona(config) as daytona:
                if isinstance(params, CreateSnapshotParams):
                    resource = await daytona.snapshot.create(params, timeout=timeout)
                else:
                    resource = await daytona.create(params, timeout=timeout)
                result = resource.id
        except Exception as error:
            result = error
    sys.stdout.buffer.write(pickle.dumps(result))

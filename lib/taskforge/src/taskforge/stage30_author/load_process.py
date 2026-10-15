# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load a program's source in a child process, so a module body that never returns cannot hold the caller.

``load_program_source`` runs the module body in the calling process. The build child bounds the
steps it runs, but a caller that only loads a program (the agentic author's loader, build and submit
tools) would block its event loop on a module body that spins. ``load_program_in_process`` runs the
loader in a child process for at most ``wall_timeout`` seconds, with the source on its stdin, and
reads back the program's manifest (``author.program_manifest``) or the loader's refusal.
"""

import asyncio
import json
import sys
from asyncio.subprocess import PIPE

from pydantic import BaseModel

from taskforge.child_process import stop_process_group
from taskforge.stage30_author.author import program_manifest
from taskforge.stage30_author.loader import load_program_source

MODULE = "taskforge.stage30_author.load_process"
STDERR_TAIL = 2000


class StepManifest(BaseModel):
    inputs: list[str]
    source_digest: str


class ProgramManifest(BaseModel):
    """``program_manifest`` as a model: the digest, the sink, the run order, and each step's inputs and source digest."""

    digest: str
    sink: str
    order: list[str]
    steps: dict[str, StepManifest]


class LoadResult(BaseModel):
    """What the child writes: the manifest of a program that loads, or the loader's refusal."""

    manifest: ProgramManifest | None = None
    error: str | None = None


async def load_program_in_process(source: str, *, wall_timeout: float) -> ProgramManifest:
    """The manifest of ``source``, loaded by ``load_program_source`` in a child process.

    Raises:
        ValueError: the loader refuses ``source`` (its message), or the module body ran past
            ``wall_timeout`` seconds.
        RuntimeError: the child exited without a result.
    """
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", MODULE, stdin=PIPE, stdout=PIPE, stderr=PIPE, start_new_session=True
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(source.encode()), wall_timeout)
    except TimeoutError:
        raise ValueError(
            f"the program's module body ran for more than {wall_timeout:.0f} seconds without returning; keep "
            "the module level to imports, steps and the program, and every computation inside a step"
        ) from None
    finally:
        await stop_process_group(process)
    if process.returncode != 0 or not stdout:
        tail = stderr.decode(errors="replace")[-STDERR_TAIL:]
        raise RuntimeError(f"the program loader process exited with status {process.returncode}; stderr ends:\n{tail}")
    result = LoadResult.model_validate_json(stdout)
    if result.error is not None:
        raise ValueError(result.error)
    assert result.manifest is not None
    return result.manifest


def main() -> None:
    source = sys.stdin.read()
    try:
        result = LoadResult(manifest=ProgramManifest.model_validate(program_manifest(load_program_source(source))))
    except ValueError as error:
        result = LoadResult(error=str(error))
    sys.stdout.write(json.dumps(result.model_dump()))


if __name__ == "__main__":
    main()

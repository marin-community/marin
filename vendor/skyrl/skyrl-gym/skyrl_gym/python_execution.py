"""Persistent Python tools executed through the task's Shellbox machine."""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from pydantic import TypeAdapter
from rolloutengine.cleanup import finish_cleanup
from shellbox.machine import Command, ExitReason, Machine, Result

from skyrl_gym.python_kernel import FRAME_LIMIT_BYTES, KERNEL_TERMINATED_EXIT_CODE, KERNEL_TIMEOUT_EXIT_CODE

KERNEL_SCRIPT = Path(__file__).with_name("python_kernel.py")
KERNEL_STARTUP_TIMEOUT = 40.0
KERNEL_CLEANUP_TIMEOUT = 10.0
KERNEL_OUTPUT_LIMIT_BYTES = 65536
KERNEL_RESULT = TypeAdapter(Result)


class PythonKernel:
    """One task-local interpreter with bounded execution and output."""

    def __init__(self, machine: Machine, *, memory_bytes: int | None = None):
        self.machine = machine
        self.memory_bytes = memory_bytes
        self.directory = f"/tmp/skyrl-python-{uuid4().hex}"
        self.script = f"{self.directory}/kernel.py"
        self.started = False
        self.failure: Result | None = None

    async def start(self) -> None:
        await self.machine.upload(KERNEL_SCRIPT, self.script)
        result = await self.machine.run(
            Command(
                ("python", self.script, "start", self.directory, str(self.memory_bytes or 0)),
                env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                timeout=KERNEL_STARTUP_TIMEOUT,
            )
        )
        if result.reason != ExitReason.EXITED or result.exit_code != 0:
            raise RuntimeError(f"Python kernel startup failed: {result.stderr.decode(errors='replace')}")
        self.started = True

    async def execute(
        self, code: str, *, timeout: float, output_limit_bytes: int = KERNEL_OUTPUT_LIMIT_BYTES
    ) -> Result:
        """Execute Python in the same namespace; return the terminal failure after process loss."""
        if self.failure is not None:
            return self.failure
        assert self.started
        command = Command(
            ("python", self.script, "call", self.directory),
            stdin=json.dumps(
                {"code": code, "timeout": timeout, "output_limit_bytes": output_limit_bytes}, allow_nan=False
            ).encode(),
            timeout=timeout + 10.0,
            output_limit_bytes=FRAME_LIMIT_BYTES,
        )
        try:
            result = await self.machine.run(command)
        except asyncio.CancelledError as interruption:
            try:
                await finish_cleanup(self._close_interrupted_kernel, timeout=KERNEL_CLEANUP_TIMEOUT)
            except (Exception, asyncio.CancelledError) as cleanup_error:
                raise interruption from cleanup_error
            raise
        if result.reason == ExitReason.TIMED_OUT or result.exit_code in (
            KERNEL_TIMEOUT_EXIT_CODE,
            KERNEL_TERMINATED_EXIT_CODE,
        ):
            timed_out = result.reason == ExitReason.TIMED_OUT or result.exit_code == KERNEL_TIMEOUT_EXIT_CODE
            self.failure = Result(
                None if timed_out else result.exit_code,
                result.stdout,
                result.stderr,
                result.stdout_truncated,
                result.stderr_truncated,
                ExitReason.TIMED_OUT if timed_out else ExitReason.EXITED,
            )
            await finish_cleanup(self._close_interrupted_kernel, timeout=KERNEL_CLEANUP_TIMEOUT)
            return self.failure
        if result.reason != ExitReason.EXITED or result.exit_code != 0 or result.stdout_truncated:
            raise RuntimeError(f"Python kernel execution failed: {result.stderr.decode(errors='replace')}")
        return KERNEL_RESULT.validate_json(result.stdout, strict=True)

    async def _close_interrupted_kernel(self) -> None:
        try:
            await self.close()
        except (Exception, asyncio.CancelledError) as error:
            # A live candidate must not share the machine with subsequent grading.
            await self.machine.close()
            self.started = False
            error.add_note("The task machine was closed because Python kernel cleanup failed.")
            raise

    async def close(self) -> None:
        if not self.started:
            return
        result = await self.machine.run(
            Command(("python", self.script, "close", self.directory), timeout=KERNEL_CLEANUP_TIMEOUT)
        )
        if result.reason != ExitReason.EXITED or result.exit_code != 0:
            raise RuntimeError(f"Python kernel cleanup failed: {result.stderr.decode(errors='replace')}")
        self.started = False

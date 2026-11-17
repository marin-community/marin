"""Bounded Lean compiler execution inside a Shellbox machine."""

import json
from pathlib import Path
from uuid import uuid4

from pydantic import TypeAdapter
from shellbox.machine import Command, ExitReason, Machine, Result

from skyrl_gym.lean_compiler import COMPILER_OUTPUT_LIMIT_BYTES

COMPILER_SCRIPT = Path(__file__).with_name("lean_compiler.py")
COMPILER_RESULT = TypeAdapter(Result)


async def compile_lean(machine: Machine, proof: str, *, project: str, timeout: float) -> Result:
    """Run the task image's Lean project and return bounded compiler diagnostics."""
    script = f"/tmp/skyrl-lean-{uuid4().hex}/compiler.py"
    await machine.upload(COMPILER_SCRIPT, script)
    result = await machine.run(
        Command(
            ("python", script),
            stdin=json.dumps({"proof": proof, "project": project, "timeout": timeout}).encode(),
            cwd=project,
            timeout=timeout + 10.0,
            output_limit_bytes=12 * COMPILER_OUTPUT_LIMIT_BYTES + 1024,
        )
    )
    if result.reason != ExitReason.EXITED or result.exit_code != 0 or result.stdout_truncated:
        raise RuntimeError(f"Lean compiler execution failed: {result.stderr.decode(errors='replace')}")
    return COMPILER_RESULT.validate_json(result.stdout, strict=True)

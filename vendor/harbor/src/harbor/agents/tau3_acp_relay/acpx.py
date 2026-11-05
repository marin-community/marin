"""Subprocess client for the persistent ACPX session."""

import asyncio
import json

from .relay import ACPX_EXIT_ERROR_PREFIX, TURN_TIMEOUT_ERROR_PREFIX


class ACPXPromptClient:
    """Prompt the persistent ACPX session configured by Harbor."""

    def __init__(self, timeout_sec: float | None) -> None:
        self._timeout_sec = timeout_sec

    async def prompt(self, message: str) -> str:
        process = await asyncio.create_subprocess_exec(
            "acpx",
            "prompt",
            "--file",
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # ACPX interprets JSON arrays as content blocks, even in user messages.
        prompt = json.dumps([{"type": "text", "text": message}]).encode()
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt), timeout=self._timeout_sec
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(
                f"{TURN_TIMEOUT_ERROR_PREFIX} {self._timeout_sec} seconds"
            ) from None

        output = stdout.decode(errors="replace").strip()
        error = stderr.decode(errors="replace").strip()
        if process.returncode != 0:
            detail = error or output
            raise RuntimeError(
                f"{ACPX_EXIT_ERROR_PREFIX} {process.returncode}: {detail}"
            )
        return output

"""The worker log bridge preserves asyncio diagnostics without an exception."""

import subprocess
import sys


def test_worker_logging_preserves_asyncio_messages_and_exceptions():
    script = """
import asyncio
from loguru import logger
from skyrl_train.utils.utils import configure_ray_worker_logging

configure_ray_worker_logging()
loop = asyncio.new_event_loop()
try:
    loop.default_exception_handler({"message": "pending worker probe"})
    try:
        raise RuntimeError("worker cleanup probe")
    except RuntimeError as error:
        loop.default_exception_handler({"message": "failed worker probe", "exception": error})
finally:
    loop.close()
    logger.complete()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=30)
    assert "pending worker probe" in result.stderr
    assert "failed worker probe" in result.stderr
    assert "RuntimeError: worker cleanup probe" in result.stderr

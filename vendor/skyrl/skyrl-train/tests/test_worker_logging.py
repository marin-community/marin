"""The worker log bridge preserves asyncio diagnostics and exception traces."""

import subprocess
import sys


def test_worker_logging_preserves_asyncio_diagnostics_and_traceback():
    script = """
import asyncio
from loguru import logger
from skyrl_train.utils.utils import configure_ray_worker_logging

def fail_worker():
    raise RuntimeError("worker cleanup probe")

configure_ray_worker_logging()
loop = asyncio.new_event_loop()
try:
    loop.call_exception_handler({"message": "pending worker probe", "worker": "worker-17"})
    try:
        fail_worker()
    except RuntimeError as error:
        loop.call_exception_handler({"message": "failed worker probe", "exception": error, "worker": "worker-29"})
finally:
    loop.close()
    logger.complete()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=30)
    assert "pending worker probe" in result.stderr
    assert "worker-17" in result.stderr
    assert "failed worker probe" in result.stderr
    assert "worker-29" in result.stderr
    assert "RuntimeError" in result.stderr
    assert "worker cleanup probe" in result.stderr
    assert "fail_worker" in result.stderr

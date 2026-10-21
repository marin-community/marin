import asyncio
import json
from pathlib import Path

import pytest
from harbor.environments.base import ExecResult
from skyrl_train.rollouts.agent_environment import _nonzero_agent_failure_code, _pi_failure_code
from skyrl_train.rollouts.diagnostic_spool import DiagnosticSpool


def test_pi_failure_code_only_classifies_fixed_safe_messages():
    assert _pi_failure_code(ValueError("Pi evidence stdout contained non-frame output")) == "non_frame_stdout"
    assert _pi_failure_code(ValueError("Pi did not return exactly one in-memory evidence frame (missing chunks)")) == "missing_frame_chunks"
    assert _pi_failure_code(ValueError("Pi evidence frame must contain one collated trajectory")) == "missing_collated_trajectory"
    assert _pi_failure_code(ValueError("secret=do-not-persist")) == "other_value_error"
    assert _pi_failure_code(RuntimeError("secret=do-not-persist")) is None


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        ("npm error code EAI_AGAIN secret=do-not-persist", "dns_temporary_failure"),
        ("npm ERR! code ETIMEDOUT", "network_timeout"),
        ("npm ERR! code ECONNRESET", "connection_reset"),
        ("npm error code E404", "npm_package_not_found"),
        ("curl: (22) service unavailable", "curl_failed"),
        ("private task content secret=do-not-persist", "unclassified_nonzero_exit"),
    ],
)
def test_nonzero_agent_exit_code_never_persists_stderr(stderr, expected):
    result = ExecResult(return_code=1, stderr=stderr)
    code = _nonzero_agent_failure_code(result)
    assert code == expected
    assert "secret" not in code


@pytest.mark.asyncio
async def test_diagnostics_stream_before_attempt_finishes_and_flush_last_event(tmp_path):
    destination = "s3://same-region/pi-smoke/attempt/engine/diagnostics.jsonl"
    published: list[bytes] = []
    arrived = asyncio.Event()

    def publish(source: str, target: str) -> None:
        assert target == destination
        published.append(Path(source).read_bytes())
        arrived.set()

    spool = DiagnosticSpool(tmp_path / "engine" / "diagnostics.jsonl", destination, interval=0.01, publisher=publish)
    stop = asyncio.Event()
    task = asyncio.create_task(spool.run(stop))
    try:
        await asyncio.wait_for(arrived.wait(), 2)
        assert json.loads(published[0].splitlines()[0])["phase"] == "attempt:started"
        spool.record("agent:run_start")
        arrived.clear()
        await asyncio.wait_for(arrived.wait(), 2)
        assert b'"agent:run_start"' in published[-1]
        # No new PUT is needed when nothing in the journal has changed.
        count = len(published)
        await asyncio.sleep(0.04)
        assert len(published) == count
        spool.record("agent:failed")
    finally:
        stop.set()
        await task
    assert b'"agent:failed"' in published[-1]
    assert all(b"s3://" not in content for content in published)


@pytest.mark.asyncio
async def test_diagnostics_upload_failure_fails_loudly(tmp_path):
    def fail(source: str, target: str) -> None:
        raise OSError("storage unavailable")

    spool = DiagnosticSpool(
        tmp_path / "engine" / "diagnostics.jsonl", "s3://same-region/attempt/engine/diagnostics.jsonl", publisher=fail
    )
    with pytest.raises(OSError, match="storage unavailable"):
        await spool.run(asyncio.Event())

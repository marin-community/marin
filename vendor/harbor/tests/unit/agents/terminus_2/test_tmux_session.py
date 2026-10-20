import re
import shlex
import subprocess
from unittest.mock import AsyncMock

import pytest

from harbor.agents.base import MalformedModelResponseError
from harbor.agents.terminus_2 import tmux_session as tmux_session_module
from harbor.agents.terminus_2.tmux_session import (
    _BatchMarkers,
    ModelInitiatedTmuxFailure,
    TmuxBatchProtocolError,
    TmuxCommandError,
    TmuxKeystrokeBatch,
    TmuxSession,
    TmuxSessionEndedError,
)
from harbor.environments.base import ExecResult
from harbor_config import ErrorCategory, error_category
from harbor_config.errors import is_resume_retryable


@pytest.fixture
def tmux_session(mock_environment):
    mock_environment.session_id = "test-session-id"
    return TmuxSession(
        session_name="test-session",
        environment=mock_environment,
    )


def _extract_send_keys_payload(command: str) -> list[str]:
    parts = shlex.split(command)
    assert parts[:5] == ["tmux", "send-keys", "-t", "test-session", "--"]
    return parts[5:]


def _extract_called_command(call) -> str:
    if "command" in call.kwargs:
        return call.kwargs["command"]
    return call.args[0]


async def test_start_configures_history_without_persisting_pane(tmux_session):
    tmux_session._attempt_tmux_installation = AsyncMock()
    tmux_session.environment.exec = AsyncMock(return_value=ExecResult(return_code=0))

    await tmux_session.start()

    command = _extract_called_command(tmux_session.environment.exec.await_args_list[0])
    set_option = "set-option -g history-limit 10000000"
    assert command.index(set_option) < command.index("new-session")
    assert "pipe-pane" not in command


def _batch_marker(command: str, suffix: str) -> str:
    match = re.search(
        rf"(__HARBOR_TMUX_[0-9a-f]+_{suffix}__)",
        command,
    )
    assert match is not None
    return match.group(1)


def _batch_output(command: str, *, visible: str, full: str) -> str:
    return (
        f"{_batch_marker(command, 'VISIBLE')}\n{visible}"
        f"{_batch_marker(command, 'FULL')}\n{full}"
    )


def _batch(
    keystrokes: str | list[str], duration_sec: float = 0.0
) -> TmuxKeystrokeBatch:
    return TmuxKeystrokeBatch(keystrokes, duration_sec)


@pytest.mark.parametrize(
    "keystrokes",
    [
        "pkill -f uvicorn\n",
        "pkill -9 -f uvicorn\n",
        "kill $(pgrep -f uvicorn)\n",
        ["pkill -f uvicorn", "Enter"],
        ["pkill -f uvicorn\n"],
    ],
)
async def test_model_process_kill_is_scoreable(tmux_session, keystrokes):
    async def execute(command, **kwargs):
        return ExecResult(
            return_code=143,
            stdout=f"{_batch_marker(command, 'SENT')}0\n",
            stderr="Terminated",
        )

    tmux_session.environment.exec = AsyncMock(side_effect=execute)

    with pytest.raises(ModelInitiatedTmuxFailure) as caught:
        await tmux_session.send_keys_and_capture([_batch(keystrokes)])

    assert isinstance(caught.value.__cause__, TmuxBatchProtocolError)
    assert caught.value.result.return_code == 143
    name = type(caught.value).__name__
    assert error_category(name) is ErrorCategory.AGENT
    assert not is_resume_retryable(name)


@pytest.mark.parametrize(
    ("keystrokes", "return_code", "sent"),
    [
        ("echo 'pkill -f uvicorn'\n", 143, True),
        ("kill 1234\n", 143, True),
        ("pkill -f uvicorn\n", 124, True),
        ("pkill -f uvicorn\n", 143, False),
        ("pkill -f uvicorn\n", 127, True),
        ("pkill -f uvicorn", 143, True),
        (["pkill", "-f", "uvicorn", "Enter"], 143, True),
        ("pkill -f\nuvicorn\n", 143, True),
        ("pkill -f '['\n", 143, True),
    ],
)
async def test_unattributed_batch_failure_stays_infrastructure(
    tmux_session, keystrokes, return_code, sent
):
    async def execute(command, **kwargs):
        return ExecResult(
            return_code=return_code,
            stdout=f"{_batch_marker(command, 'SENT')}0\n" if sent else "",
        )

    tmux_session.environment.exec = AsyncMock(side_effect=execute)

    with pytest.raises(TmuxBatchProtocolError):
        await tmux_session.send_keys_and_capture([_batch(keystrokes)])


@pytest.mark.parametrize(
    "keystrokes",
    ["exit\n", "C-d", "tmux kill-session -t test-session\n", ["exit", "Enter"]],
)
async def test_model_session_exit_is_scoreable(tmux_session, keystrokes):
    async def execute(command, **kwargs):
        return ExecResult(
            return_code=0,
            stdout=(
                f"{_batch_marker(command, 'SENT')}0\n"
                f"{_batch_marker(command, 'SESSION_ENDED')}\n"
            ),
        )

    tmux_session.environment.exec = AsyncMock(side_effect=execute)

    with pytest.raises(ModelInitiatedTmuxFailure) as caught:
        await tmux_session.send_keys_and_capture([_batch(keystrokes), _batch("pwd\n")])

    assert isinstance(caught.value.__cause__, TmuxSessionEndedError)


@pytest.mark.parametrize(
    ("keystrokes", "sent"),
    [
        ("exit\n", False),
        ("exit", True),
        ("C-d\n", True),
        ("'C-d'\n", True),
        (["exit", "0", "Enter"], True),
    ],
)
async def test_session_loss_is_not_attributed_to_unexecuted_exit(
    tmux_session, keystrokes, sent
):
    async def execute(command, **kwargs):
        return ExecResult(
            return_code=0,
            stdout=(
                (f"{_batch_marker(command, 'SENT')}0\n" if sent else "")
                + f"{_batch_marker(command, 'SESSION_ENDED')}\n"
            ),
        )

    tmux_session.environment.exec = AsyncMock(side_effect=execute)

    with pytest.raises(TmuxSessionEndedError):
        await tmux_session.send_keys_and_capture([_batch(keystrokes), _batch("pwd\n")])


@pytest.mark.parametrize("batched", [False, True])
async def test_surrogate_command_is_classified_as_model_failure(tmux_session, batched):
    command = "echo \ud842\n"

    with pytest.raises(MalformedModelResponseError) as caught:
        if batched:
            await tmux_session.send_keys_and_capture([_batch(command)])
        else:
            await tmux_session.send_keys(command, block=False, min_timeout_sec=0)

    assert isinstance(caught.value.__cause__, UnicodeEncodeError)
    assert command.rstrip() in caught.value.__cause__.object


_CAPTURE_LIMIT = tmux_session_module._MAX_CAPTURE_BYTES
_ASCII_CAPTURE_COMMAND = f"yes x | head -c {_CAPTURE_LIMIT * 2}; printf 'END\\n'"
_MULTIBYTE_LINE_BYTES = 159 * len("🙂".encode("utf-8")) + 1
_MULTIBYTE_LINE_COUNT = _CAPTURE_LIMIT * 2 // _MULTIBYTE_LINE_BYTES + 2
while (_MULTIBYTE_LINE_BYTES * _MULTIBYTE_LINE_COUNT + 2 - _CAPTURE_LIMIT) % len(
    "🙂".encode("utf-8")
) == 0:
    _MULTIBYTE_LINE_COUNT += 1
_MULTIBYTE_CAPTURE_COMMAND = (
    "python3 -c 'import sys; "
    f'sys.stdout.buffer.write((("🙂"*159+"\\n")*{_MULTIBYTE_LINE_COUNT}+"Z\\n").encode())\''
)


def _run_with_fake_tmux(
    command: str,
    full_capture_command: str,
    max_transport_bytes: int | None = None,
) -> ExecResult:
    fake_tmux = f"""
tmux() {{
  case "$1" in
    has-session|send-keys)
      return 0
      ;;
    capture-pane)
      case " $* " in
        *" -S - "*)
          {full_capture_command}
          ;;
        *)
          printf 'visible\\n'
          ;;
      esac
      ;;
  esac
}}
"""
    result = subprocess.run(
        ["bash", "-c", f"{fake_tmux}{command}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        max_transport_bytes is not None
        and len(result.stdout.encode("utf-8")) > max_transport_bytes
    ):
        raise OverflowError("terminal capture exceeded the transport memory bound")
    return ExecResult(
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _run_batch_with_has_session_error(command: str, diagnostic: str):
    fake_tmux = (
        f"tmux() {{ printf '%s\\n' {shlex.quote(diagnostic)} >&2; return 1; }}\n"
    )
    return subprocess.run(
        ["bash", "-c", f"{fake_tmux}{command}"],
        check=False,
        capture_output=True,
        text=True,
    )


def test_tmux_send_keys_chunks_long_payload_below_limit(tmux_session):
    long_key = "x" * (tmux_session._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH * 2)

    commands = tmux_session._tmux_send_keys([long_key, "Enter"])

    assert len(commands) >= 2
    assert all(
        len(command) <= tmux_session._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH
        for command in commands
    )

    all_payload = []
    for command in commands:
        all_payload.extend(_extract_send_keys_payload(command))

    assert "".join(all_payload[:-1]) == long_key
    assert all_payload[-1] == "Enter"


def test_tmux_send_keys_keeps_small_payload_single_command(tmux_session):
    commands = tmux_session._tmux_send_keys(["echo hello world", "Enter"])

    assert len(commands) == 1
    assert _extract_send_keys_payload(commands[0]) == ["echo hello world", "Enter"]


def test_tmux_send_keys_keys_starting_with_dash_are_literal(tmux_session):
    """Keys starting with ``-`` must be passed as literal keys, not parsed
    as ``tmux send-keys`` options. This is enforced by the trailing ``--``
    end-of-options marker in the command prefix."""
    commands = tmux_session._tmux_send_keys(["-x", "-Lfoo", "Enter"])

    assert len(commands) == 1
    assert _extract_send_keys_payload(commands[0]) == ["-x", "-Lfoo", "Enter"]


def test_tmux_send_keys_prefix_includes_end_of_options_marker(tmux_session):
    """The built command must contain the literal ``--`` separator between
    the ``-t <session>`` option and the keys arguments."""
    [command] = tmux_session._tmux_send_keys(["echo hi", "Enter"])

    assert " -t test-session -- " in command


def test_tmux_send_keys_chunks_quote_heavy_payload_below_limit(tmux_session):
    quote_heavy_key = ("abc' def " * 2000).strip()

    commands = tmux_session._tmux_send_keys([quote_heavy_key, "Enter"])

    assert len(commands) >= 2
    assert all(
        len(command) <= tmux_session._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH
        for command in commands
    )

    all_payload = []
    for command in commands:
        all_payload.extend(_extract_send_keys_payload(command))

    assert "".join(all_payload[:-1]) == quote_heavy_key
    assert all_payload[-1] == "Enter"


def test_tmux_send_keys_many_small_keys_split_across_commands(tmux_session):
    """Many moderate-sized keys that individually fit but collectively exceed the limit."""
    max_len = tmux_session._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH
    keys = [f"key{i:04d}" + "x" * 490 for i in range(max_len // 500 * 3)]

    commands = tmux_session._tmux_send_keys(keys)

    assert len(commands) >= 2
    assert all(len(c) <= max_len for c in commands)

    all_payload = []
    for command in commands:
        all_payload.extend(_extract_send_keys_payload(command))

    assert all_payload == keys


async def test_send_keys_and_capture_batches_commands_and_observation(tmux_session):
    async def execute(command, **kwargs):
        return ExecResult(
            return_code=0,
            stdout=_batch_output(
                command,
                visible="$ second\ntwo\n",
                full="$ first\none\n$ second\ntwo\n",
            ),
        )

    tmux_session.environment.exec = AsyncMock(side_effect=execute)

    output, session_alive = await tmux_session.send_keys_and_capture(
        [_batch("first\n", 1.0), _batch("second\n", 2.0)]
    )

    assert session_alive is True
    assert output == "Current Terminal Screen:\n$ second\ntwo\n"
    assert tmux_session.environment.exec.await_count == 1
    call = tmux_session.environment.exec.await_args_list[0]
    command = _extract_called_command(call)
    assert command.index("first") < command.index("second")
    assert call.kwargs["timeout_sec"] == 183


@pytest.mark.parametrize(
    ("full_capture_command", "expected_suffix"),
    [
        (_ASCII_CAPTURE_COMMAND, "END\n"),
        (_MULTIBYTE_CAPTURE_COMMAND, "Z\n"),
    ],
)
async def test_capture_pane_bounds_complete_scrollback_before_transport(
    tmux_session, full_capture_command, expected_suffix
):
    async def execute(command, **kwargs):
        return _run_with_fake_tmux(
            command,
            full_capture_command,
            max_transport_bytes=_CAPTURE_LIMIT + 1_000,
        )

    tmux_session.environment.exec = AsyncMock(side_effect=execute)

    output = await tmux_session.capture_pane(capture_entire=True)

    assert output.endswith(expected_suffix)
    assert len(output.encode("utf-8")) <= _CAPTURE_LIMIT


async def test_send_keys_and_capture_bounds_complete_scrollback_before_transport(
    tmux_session,
):
    async def execute(command, **kwargs):
        return _run_with_fake_tmux(
            command,
            _ASCII_CAPTURE_COMMAND,
            max_transport_bytes=_CAPTURE_LIMIT + 1_000,
        )

    tmux_session.environment.exec = AsyncMock(side_effect=execute)

    output, session_alive = await tmux_session.send_keys_and_capture(
        [_batch("echo hi\n")]
    )

    assert session_alive is True
    assert output == "Current Terminal Screen:\nvisible\n"


async def test_send_keys_and_capture_preserves_complete_capture_failure(tmux_session):
    async def execute(command, **kwargs):
        return _run_with_fake_tmux(command, "return 17")

    tmux_session.environment.exec = AsyncMock(side_effect=execute)

    with pytest.raises(TmuxCommandError):
        await tmux_session.send_keys_and_capture([_batch("echo hi\n")])


@pytest.mark.parametrize(
    ("marker_suffix", "error_type"),
    [
        ("SESSION_ENDED", TmuxSessionEndedError),
        ("COMMAND_FAILED", TmuxCommandError),
    ],
)
async def test_send_keys_and_capture_classifies_confirmed_failure(
    tmux_session, marker_suffix, error_type
):
    async def execute(command, **kwargs):
        return ExecResult(
            return_code=0,
            stdout=f"{_batch_marker(command, marker_suffix)}\n",
        )

    tmux_session.environment.exec = AsyncMock(side_effect=execute)

    with pytest.raises(error_type):
        await tmux_session.send_keys_and_capture([_batch("echo hi\n")])


@pytest.mark.parametrize(
    "diagnostic",
    [
        "can't find session: test-session",
        "no server running on /tmp/tmux-1000/default",
        "no sessions",
    ],
)
def test_batch_script_requires_absence_diagnostic_for_session_ended(
    tmux_session, diagnostic
):
    batch_markers = _BatchMarkers.create()
    command = tmux_session._build_batch_script([], batch_markers)

    result = _run_batch_with_has_session_error(command, diagnostic)

    assert result.stdout == f"{batch_markers.session_ended}\n"


def test_batch_script_does_not_treat_unknown_tmux_failure_as_session_ended(
    tmux_session,
):
    batch_markers = _BatchMarkers.create()
    command = tmux_session._build_batch_script([], batch_markers)

    result = _run_batch_with_has_session_error(command, "permission denied")

    assert result.stdout == f"{batch_markers.command_failed}\n"
    assert result.stderr == "permission denied\n"


async def test_send_keys_and_capture_treats_post_action_exit_as_normal(tmux_session):
    async def execute(command, **kwargs):
        return ExecResult(
            return_code=0,
            stdout=f"{_batch_marker(command, 'DEAD_AFTER')}\n",
        )

    tmux_session.environment.exec = AsyncMock(side_effect=execute)

    output, session_alive = await tmux_session.send_keys_and_capture([_batch("exit\n")])

    assert session_alive is False
    assert output == "Current Terminal Screen:\n"


async def test_send_keys_and_capture_classifies_missing_markers_as_protocol_error(
    tmux_session,
):
    tmux_session.environment.exec = AsyncMock(
        return_value=ExecResult(return_code=127, stderr="bash: not found")
    )

    with pytest.raises(TmuxBatchProtocolError):
        await tmux_session.send_keys_and_capture([_batch("echo hi\n")])


async def test_send_keys_and_capture_preserves_transport_exception(tmux_session):
    transport_error = ConnectionError("bridge disconnected")
    tmux_session.environment.exec = AsyncMock(side_effect=transport_error)

    with pytest.raises(ConnectionError) as exc_info:
        await tmux_session.send_keys_and_capture([_batch("echo hi\n")])

    assert exc_info.value is transport_error


def test_tmux_session_rejects_invalid_batch_timeout_margin(mock_environment):
    with pytest.raises(ValueError, match="positive and finite"):
        TmuxSession(
            session_name="test-session",
            environment=mock_environment,
            batch_exec_timeout_margin_sec=0,
        )


async def test_send_non_blocking_keys_executes_all_chunked_commands(tmux_session):
    long_key = "x" * (tmux_session._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH * 2)
    expected_commands = tmux_session._tmux_send_keys([long_key])

    tmux_session.environment.exec = AsyncMock(return_value=ExecResult(return_code=0))

    await tmux_session._send_non_blocking_keys(keys=[long_key], min_timeout_sec=0.0)

    executed_commands = [
        _extract_called_command(call)
        for call in tmux_session.environment.exec.await_args_list
    ]
    assert executed_commands == expected_commands


async def test_send_non_blocking_keys_small_payload_single_exec(tmux_session):
    tmux_session.environment.exec = AsyncMock(return_value=ExecResult(return_code=0))

    await tmux_session._send_non_blocking_keys(keys=["echo hi"], min_timeout_sec=0.0)

    assert tmux_session.environment.exec.await_count == 1
    command = _extract_called_command(tmux_session.environment.exec.await_args_list[0])
    assert _extract_send_keys_payload(command) == ["echo hi"]


async def test_send_non_blocking_keys_raises_on_failed_chunk(tmux_session):
    long_key = "x" * (tmux_session._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH * 2)
    commands = tmux_session._tmux_send_keys([long_key])
    assert len(commands) >= 2

    responses = [ExecResult(return_code=0) for _ in commands]
    responses[1] = ExecResult(return_code=1, stderr="command too long")
    tmux_session.environment.exec = AsyncMock(side_effect=responses)

    with pytest.raises(
        RuntimeError,
        match="failed to send non-blocking keys",
    ):
        await tmux_session._send_non_blocking_keys(keys=[long_key], min_timeout_sec=0.0)

    assert tmux_session.environment.exec.await_count == 2


async def test_send_blocking_keys_waits_after_chunked_send(tmux_session):
    long_key = "x" * (tmux_session._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH * 2)
    expected_commands = tmux_session._tmux_send_keys([long_key, "Enter"])
    wait_command = "timeout 1.0s tmux wait done"

    tmux_session.environment.exec = AsyncMock(
        side_effect=[
            *[ExecResult(return_code=0) for _ in expected_commands],
            ExecResult(return_code=0),
        ]
    )

    await tmux_session._send_blocking_keys(
        keys=[long_key, "Enter"],
        max_timeout_sec=1.0,
    )

    executed_commands = [
        _extract_called_command(call)
        for call in tmux_session.environment.exec.await_args_list
    ]
    assert executed_commands == [*expected_commands, wait_command]


async def test_send_blocking_keys_raises_on_failed_chunk(tmux_session):
    long_key = "x" * (tmux_session._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH * 2)
    commands = tmux_session._tmux_send_keys([long_key, "Enter"])
    assert len(commands) >= 2

    tmux_session.environment.exec = AsyncMock(
        return_value=ExecResult(return_code=1, stderr="failed to send command"),
    )

    with pytest.raises(
        RuntimeError,
        match="failed to send blocking keys",
    ):
        await tmux_session._send_blocking_keys(
            keys=[long_key, "Enter"],
            max_timeout_sec=1.0,
        )

    # Stops on first failure — never reaches tmux wait
    assert tmux_session.environment.exec.await_count == 1


async def test_send_blocking_keys_raises_timeout_on_wait_failure(tmux_session):
    tmux_session.environment.exec = AsyncMock(
        side_effect=[
            ExecResult(return_code=0),
            ExecResult(return_code=124, stderr=""),
        ],
    )

    with pytest.raises(TimeoutError, match="timed out after"):
        await tmux_session._send_blocking_keys(
            keys=["echo hello", "Enter"],
            max_timeout_sec=1.0,
        )


async def test_send_non_blocking_keys_error_message_includes_diagnostics(tmux_session):
    """When a chunk fails, the RuntimeError message must include the
    failing command, return_code, stderr and stdout to aid debugging."""
    tmux_session.environment.exec = AsyncMock(
        return_value=ExecResult(
            return_code=42, stderr="boom-stderr", stdout="boom-stdout"
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await tmux_session._send_non_blocking_keys(
            keys=["echo hi"], min_timeout_sec=0.0
        )

    message = str(exc_info.value)
    assert "failed to send non-blocking keys" in message
    assert "return_code=42" in message
    assert "boom-stderr" in message
    assert "boom-stdout" in message
    assert "command=" in message


async def test_send_blocking_keys_error_message_includes_diagnostics(tmux_session):
    """When a chunk fails, the RuntimeError message must include the
    failing command, return_code, stderr and stdout to aid debugging."""
    tmux_session.environment.exec = AsyncMock(
        return_value=ExecResult(
            return_code=7, stderr="bad-stderr", stdout="bad-stdout"
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await tmux_session._send_blocking_keys(
            keys=["echo hello", "Enter"],
            max_timeout_sec=1.0,
        )

    message = str(exc_info.value)
    assert "failed to send blocking keys" in message
    assert "return_code=7" in message
    assert "bad-stderr" in message
    assert "bad-stdout" in message
    assert "command=" in message

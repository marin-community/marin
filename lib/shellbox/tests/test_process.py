# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import sys
from pathlib import Path

import pytest
from shellbox.lifecycle import CommandLifetime, finish_cleanup
from shellbox.process import ProcessScope


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper and procfs contract")
@pytest.mark.parametrize("ending", ["complete", "timeout", "cancel", "cancel_before_start"])
def test_scope_reaps_detached_orphan_and_prevents_late_writes(tmp_path, ending):
    async def scenario():
        scope = ProcessScope(str(tmp_path / "scope"))
        script = """
import os, signal, sys
from pathlib import Path
root = Path(sys.argv[1])
r, w = os.pipe()
if os.fork() == 0:
    os.close(r)
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (root / "child").write_text(str(os.getpid()))
    os.write(w, b"ready")
    os.close(w)
    with (root / "release").open() as fifo:
        fifo.read()
    (root / "late").touch()
    os._exit(0)
os.close(w)
os.read(r, 5)
print("ready", flush=True)
if sys.argv[2] != "complete":
    signal.pause()
"""
        os.mkfifo(tmp_path / "release")
        if ending == "cancel_before_start":
            cancel = await asyncio.create_subprocess_shell(scope.cancel_command())
            assert await cancel.wait() == 0
        command = await asyncio.create_subprocess_exec(
            *scope.command((sys.executable, "-c", script, str(tmp_path), ending), 0.5 if ending == "timeout" else None),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            if ending != "cancel_before_start":
                assert await command.stdout.readline() == b"ready\n"
            if ending == "cancel":
                cancel = await asyncio.create_subprocess_shell(scope.cancel_command())
                assert await cancel.wait() == 0
            await asyncio.wait_for(command.communicate(), 10)
            assert (
                command.returncode == {"complete": 0, "timeout": 124, "cancel": 143, "cancel_before_start": 143}[ending]
            )
            if ending == "cancel_before_start":
                assert not (tmp_path / "child").exists()
            else:
                pid = int((tmp_path / "child").read_text())
                assert not Path(f"/proc/{pid}").exists()
            assert not (tmp_path / "late").exists()
        finally:
            if command.returncode is None:
                command.terminate()
                await command.communicate()

    asyncio.run(scenario())


def test_machine_close_drains_command_through_repeated_cancellation():
    async def scenario():
        lifetime = CommandLifetime()
        running, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        effects = []

        async def cleanup():
            cleaning.set()
            await release.wait()
            effects.append("stopped")

        async def command():
            with lifetime.admit():
                running.set()
                try:
                    await asyncio.Future()
                finally:
                    await finish_cleanup(cleanup())

        task = asyncio.create_task(command())
        await running.wait()
        closing = asyncio.create_task(lifetime.close())
        await cleaning.wait()
        task.cancel()
        assert not closing.done()
        release.set()
        await closing
        assert effects == ["stopped"]
        assert task.cancelled()
        with pytest.raises(RuntimeError, match="closed"):
            with lifetime.admit():
                raise AssertionError("Closed machine admitted a command")

    asyncio.run(scenario())

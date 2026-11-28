# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SIGTERM handling of the controller daemon process."""

import re
import signal
import subprocess
import sys
import textwrap

from rigging.timing import Duration, ExponentialBackoff

# Runs the real daemon entrypoint so the signal handler and interpreter exit
# behave as in production. A short periodic checkpoint interval keeps the
# checkpoint loop busy when SIGTERM arrives.
_SERVE_SCRIPT = textwrap.dedent(
    """
    import logging, sys
    from pathlib import Path
    from iris.cluster.controller import main as controller_main
    from iris.testing.config import make_controller_only_config
    from rigging.log_setup import configure_logging

    tmp = Path(sys.argv[1])
    configure_logging(level=logging.INFO)
    config = make_controller_only_config()
    config.storage.remote_state_dir = (tmp / "remote").as_uri()
    controller_main.run_controller_serve(
        config, host="127.0.0.1", port=0, state_dir=tmp / "state", fresh=True, checkpoint_interval=0.2
    )
    """
)


def test_sigterm_exits_after_final_checkpoint_and_it_is_the_newest(tmp_path):
    log_path = tmp_path / "controller.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-c", _SERVE_SCRIPT, str(tmp_path)], stdout=log, stderr=subprocess.STDOUT
        )
    try:
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: "ready to accept connections" in log_path.read_text() or proc.poll() is not None,
            timeout=Duration.from_seconds(60),
        )
        assert proc.poll() is None, log_path.read_text()
        checkpoints = tmp_path / "remote" / "controller-state"
        ExponentialBackoff(initial=0.05, maximum=0.5).wait_until(
            lambda: checkpoints.exists() and any(checkpoints.iterdir()),
            timeout=Duration.from_seconds(10),
        )

        proc.send_signal(signal.SIGTERM)
        returncode = proc.wait(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    output = log_path.read_text()
    assert returncode == 0, output
    final = re.search(r"Final checkpoint written: \S+/controller-state/(\d+) ", output)
    assert final is not None, output
    newest = max(checkpoints.iterdir(), key=lambda d: int(d.name))
    assert newest.name == final.group(1)
    assert sorted(p.name for p in newest.iterdir()) == ["auth.sqlite3.zst", "controller.sqlite3.zst"]

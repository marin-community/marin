# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import fcntl
import os
import signal
import subprocess
import sys


def main() -> None:
    mode, lock_path = sys.argv[1:]
    if mode == "interrupt":

        def cleanup(_signal, _frame):
            with open(lock_path, "w") as marker:
                marker.write("cancelled")
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, cleanup)
        os.kill(os.getppid(), signal.SIGINT)
        signal.alarm(8)
        signal.pause()
        return
    if mode == "hold":
        with open(lock_path, "r+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            print("ready", flush=True)
            signal.alarm(8)
            signal.pause()
        return

    worker = subprocess.Popen([sys.executable, "-I", __file__, "hold", lock_path], stdout=subprocess.PIPE, text=True)
    assert worker.stdout.readline().strip() == "ready"
    print("build child ready", file=sys.stderr, flush=True)
    if mode == "exit":
        raise SystemExit(7)
    worker.wait()


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hold a file lock while a test exercises native process-group cleanup."""

import fcntl
import subprocess
import sys
from pathlib import Path
from time import sleep


def main() -> int:
    mode, lock, *options = sys.argv[1:]
    if mode == "child":
        with Path(lock).open("r+") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            print("ready", flush=True)
            sleep(8)
        return 0
    worker = subprocess.Popen([sys.executable, "-I", __file__, "child", lock], stdout=subprocess.PIPE, text=True)
    assert worker.stdout is not None and worker.stdout.readline().strip() == "ready"
    print("build child ready", file=sys.stderr, flush=True)
    return 7 if options == ["True"] else worker.wait()


if __name__ == "__main__":
    raise SystemExit(main())

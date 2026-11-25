# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import fcntl
import subprocess
import sys
from time import sleep


def main() -> None:
    mode, lock_path = sys.argv[1:]
    if mode == "hold":
        with open(lock_path, "r+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            print("ready", flush=True)
            sleep(8)
        return

    worker = subprocess.Popen([sys.executable, "-I", __file__, "hold", lock_path], stdout=subprocess.PIPE, text=True)
    assert worker.stdout.readline().strip() == "ready"
    print("build child ready", file=sys.stderr, flush=True)
    if mode == "exit":
        raise SystemExit(7)
    worker.wait()


if __name__ == "__main__":
    main()

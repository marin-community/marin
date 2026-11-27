# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Child driver behaviors for the runtime watchdog tests."""

import argparse
import os
import signal
import threading


def block_abort(*_args) -> None:
    while True:
        print("[fd-monitor] process is still allocated", flush=True)
        threading.Event().wait(0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("behavior", choices=("abort", "stall", "progress", "silent"))
    parser.add_argument("ready", type=int)
    args = parser.parse_args()
    os.write(args.ready, b"r")
    os.close(args.ready)
    if args.behavior == "abort":
        os.abort()
    elif args.behavior == "stall":
        signal.signal(signal.SIGABRT, block_abort)
        print("training started", flush=True)
        os.kill(os.getpid(), signal.SIGABRT)
    elif args.behavior == "progress":
        for phase in range(6):
            print(f"completed phase {phase}", flush=True)
            threading.Event().wait(0.05)
    else:
        threading.Event().wait(0.1)


if __name__ == "__main__":
    main()

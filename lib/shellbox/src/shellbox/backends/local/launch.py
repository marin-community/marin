# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply a command's resource limits and exec bwrap.

The local backend runs this source as ``python -I -S -c <source> <limits JSON> <program> <args>...``,
so it imports only the standard library. The limits pass through bwrap to every sandboxed process.
"""

import json
import os
import resource
import signal
import sys


def main() -> None:
    for limit, value in json.loads(sys.argv[1]):
        soft, _ = resource.getrlimit(limit)
        value = value if soft == resource.RLIM_INFINITY else min(value, soft)
        resource.setrlimit(limit, (value, value))
    # Python ignores these signals at startup, and exec would pass the ignored dispositions on to the command.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
    os.execv(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()

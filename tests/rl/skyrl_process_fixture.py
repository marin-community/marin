# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
import time
from pathlib import Path


def main(mode: str, *values: str) -> int:
    if mode == "environment-value":
        print(os.environ[values[0]])
    elif mode == "environment":
        print(
            json.dumps(
                {
                    "python": list(sys.version_info[:2]),
                    "base_prefix": sys.base_prefix,
                    "cwd": str(Path.cwd()),
                    "environment": {
                        key: os.environ.get(key)
                        for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV")
                    },
                }
            )
        )
    elif mode == "stream":
        marker = Path(values[0])
        print("launcher line", file=sys.stderr, flush=True)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.01)
        sys.stdout.write("saw" if marker.exists() else "missed")
    elif mode == "invalid-stderr":
        sys.stderr.buffer.write(b"\xff bad\n")
        return 4
    else:
        raise ValueError(mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))

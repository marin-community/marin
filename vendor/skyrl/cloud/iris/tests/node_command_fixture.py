# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from pathlib import Path


if __name__ == "__main__":
    print(json.dumps({"cwd": os.getcwd(), "max_jobs": os.environ.get("MAX_JOBS"), "argv": sys.argv[1:]}))
    if os.environ.get("EXIT_TRANSPORT_FIXTURE") and Path(sys.argv[0]).name == "nccl_transport_probe.py":
        raise SystemExit(19)

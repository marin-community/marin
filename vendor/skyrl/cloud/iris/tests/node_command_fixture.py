# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys


if __name__ == "__main__":
    print(json.dumps({"cwd": os.getcwd(), "max_jobs": os.environ.get("MAX_JOBS"), "argv": sys.argv[1:]}))

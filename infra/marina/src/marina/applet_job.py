# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Child process entrypoint for a materialized immutable application job."""

import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    module = sys.argv[2]
    sys.argv = [module, *sys.argv[3:]]
    sys.path.insert(0, str(root.parent))
    runpy.run_module(f"{root.name}.{module}", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()

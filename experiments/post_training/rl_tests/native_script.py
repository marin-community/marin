# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import runpy
import sys
from pathlib import Path

import skyrl_train  # pyrefly: ignore[missing-import]
from cloud.iris import config_cli, launch_config  # pyrefly: ignore[missing-import]


def main() -> None:
    checkout, script, *args = sys.argv[1:]
    config_cli.verify_checkout(Path(checkout), config_cli, launch_config, skyrl_train)
    sys.argv = [script, *args]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()

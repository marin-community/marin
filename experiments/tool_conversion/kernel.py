# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Visible Bash commands for an explicitly provisioned persistent IPython kernel."""

import hashlib
import shlex
from pathlib import Path


def kernel_command(code: str, identity: str, *, initialize: bool) -> str:
    """Keep each cell intact and use one kernel per source conversation."""
    directory = "/tmp/marin-python-" + hashlib.sha256(identity.encode()).hexdigest()
    delimiter = "CELL_" + hashlib.sha256(code.encode()).hexdigest()
    while delimiter in code.splitlines():
        delimiter += "_"
    command = ""
    if initialize:
        client = Path(__file__).with_name("kernel_client.py").read_text()
        command = (
            "set -e\n"
            f"mkdir {shlex.quote(directory)}\n"
            f"cat > {directory}/cell.py <<'MARIN_KERNEL_CLIENT'\n{client}\nMARIN_KERNEL_CLIENT\n"
            f"nohup python3 -m ipykernel_launcher --IPKernelApp.parent_handle=0 -f {directory}/kernel.json "
            f"> {directory}/kernel.log 2>&1 < /dev/null &\n"
        )
    return command + f"python3 {directory}/cell.py {directory}/kernel.json <<'{delimiter}'\n{code}\n{delimiter}"

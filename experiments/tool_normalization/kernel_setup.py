# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Visible Bash setup for the tested local IPython execution environment."""

from pathlib import Path


def kernel_setup_calls() -> list[dict]:
    client = Path(__file__).with_name("kernel_cell.py").read_text()
    command = (
        "cat > /workspace/cell.py <<'CLIENT'\n" + client + "\nCLIENT\n"
        "nohup python3 -m ipykernel_launcher -f /workspace/kernel.json "
        ">/workspace/kernel.log 2>&1 &\n"
        "python3 - <<'READY'\n"
        "import time\nfrom pathlib import Path\n"
        "for _ in range(300):\n"
        "    if Path('/workspace/kernel.json').exists(): break\n"
        "    time.sleep(0.1)\n"
        "else: raise RuntimeError('kernel did not create connection file')\n"
        "READY"
    )
    return [{"name": "bash", "arguments": {"command": command}}]

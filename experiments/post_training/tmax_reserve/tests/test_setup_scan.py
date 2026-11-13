# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess

from experiments.post_training.tmax_reserve.setup_scan import scan_row, traced_setup
from experiments.post_training.tmax_reserve.shared_image import BuildSteps, Template


def test_traced_setup_logs_unhandled_failures_only(tmp_path):
    guest = tmp_path / "guest"
    guest.mkdir()
    (guest / "post_install.sh").write_text(
        "false || echo handled\n"
        "if false; then :; fi\n"
        "install_it() { missing-tool --flag; }\n"
        "install_it\n"
        "echo $GOPATH > " + str(tmp_path / "gopath") + "\n"
        "exit 0\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("apt-get", "rm"):
        (bin_dir / tool).write_text("#!/bin/bash\nexit 0\n")
        (bin_dir / tool).chmod(0o755)
    env = {**BuildSteps(Template.B).env, "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    result = subprocess.run(["bash", "-s"], input=traced_setup(str(guest), env), capture_output=True, text=True)
    row = scan_row("task_x", Template.B, result.stderr)

    # Bash reports a failing function body twice (in the function, then at its call); handled steps never.
    assert [(step["line"], step["command"]) for step in row["failed_steps"]] == [
        (3, "missing-tool --flag"),
        (4, "missing-tool --flag"),
    ]
    assert {step["exit"] for step in row["failed_steps"]} == {127}
    assert (row["post_install_exit"], row["tmux_exit"]) == (0, 0)
    assert (tmp_path / "gopath").read_text().strip() == "/home/user/go"

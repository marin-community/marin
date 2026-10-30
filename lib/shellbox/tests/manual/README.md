# Manual execution checks

These scripts exercise installed runtime assets that the test suite cannot bundle. They use assertions and exit nonzero on failure.

- `check_machine.py` compares command results and file transfer in a local Docker image and a prepared QEMU bundle.
- `check_shell.py` checks persistent Bash state, stdin, interruption, jobs, output limits, and reset in a PTY-enabled QEMU bundle.
- `check_prepared_image.py` runs one registry image in Docker and QEMU and compares its command output. It needs `docker` and `skopeo` on `PATH`, `umoci`, guest assets, and registry access.

Use the machine and shell checks directly. Marin's vendored Harbor rejects the
legacy `shellbox.agent:BashAgent` adapter; rollout execution uses `ShellHarness`.

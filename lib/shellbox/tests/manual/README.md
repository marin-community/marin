# Manual execution checks

These scripts exercise installed runtime assets that the test suite cannot bundle. They use assertions and exit nonzero on failure.

- `check_machine.py` compares command results and file transfer in a local Docker image and a prepared QEMU bundle.
- `check_shell.py` checks persistent Bash state, stdin, interruption, jobs, output limits, and reset in a PTY-enabled QEMU bundle.
- `check_prepared_image.py` runs one registry image in Docker and QEMU and compares its command output. It needs `docker` and `skopeo` on `PATH`, `umoci`, guest assets, and registry access.

`../test_harbor_shellsim.py` runs the local Harbor ShellSim trial automatically when Harbor and ShellSim are installed. `../harbor_smoke.py` also accepts a QEMU bundle for a manual Harbor trial. See the package README for commands and asset requirements.

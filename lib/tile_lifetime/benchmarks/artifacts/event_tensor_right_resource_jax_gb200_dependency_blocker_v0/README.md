# Event Tensor GB200 dependency blocker

Iris job `/dlwh/shuttle-event-right-resource-gb200-linkage` failed before the
dependency/source/ABI preflight on Shuttle commit `73b7a277f7`.

The isolated JAX environment was initially Torch-free. Installing
`quack-kernels==0.2.10` with its public wrapper dependencies then installed
Torch 2.10.0 and `torch-c-dlpack-ext==0.1.5`. The fail-closed preflight rejected
the environment with `the dependency-only JAX runtime environment contains
Torch`.

No Shuttle source/ABI compilation or device kernel launch occurred. The job
ran for 5.95 seconds and terminated, releasing the GB200 allocation. This is a
dependency-packaging failure, not physical linkage evidence.

The bounded correction installs every audited low-level dependency explicitly
and installs the QuACK source package with `--no-deps`. QuACK's Torch-facing
public wrappers remain excluded. A CPU-only repeat of the complete preflight is
required before another device request.


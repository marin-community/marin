# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""In-job GPU hang detection and recovery.

The package splits along the process boundary that recovery requires:

- ``types`` holds the pure, JAX-free data types shared across that boundary.
- ``supervisor`` is the long-lived parent process. It never creates a CUDA
  context (it makes no JAX device calls), so the child subprocess owns the GPUs;
  it warm-restarts the child from the host snapshot on a recoverable fault.
- ``snapshot``, ``detection`` and ``faults`` do the JAX work inside the child.

See ``experiments/grug/recovery`` for the end-to-end wiring and ablation runner.
"""

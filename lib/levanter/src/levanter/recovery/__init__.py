# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""In-job GPU hang detection and recovery.

The package splits along the process boundary that recovery requires:

- ``types`` and ``supervisor`` are import-safe without JAX. The supervisor is the
  long-lived parent process; it must never create a CUDA context, so it spawns
  the trainer as a subprocess and never imports JAX itself.
- ``snapshot``, ``detection`` and ``faults`` run inside the trainer subprocess and
  do import JAX.

See ``experiments/grug/recovery`` for the end-to-end wiring and ablation runner.
"""

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end wiring of the levanter hang-recovery system on a grug model.

``config`` holds the JAX-free run config (safe to pickle into the trainer
subprocess); ``train`` is the trainer entrypoint; ``run_ablations`` is the
supervisor-driven ablation sweep demonstrating fault injection and recovery.
"""

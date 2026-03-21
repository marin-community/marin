# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Alternating multi-host RL on a single TPU pod.

This package implements phase-alternating RL where one TPU pod allocation
alternates between independent per-host vLLM sampling and full-pod Levanter
training, with file-based handoff at phase boundaries.
"""

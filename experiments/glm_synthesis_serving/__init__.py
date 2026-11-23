# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standing GLM-5.3-Flash data-synthesis serving pool on cw-us-east-08a.

One brokered OpenAI-compatible endpoint served by 32 independent TP4 vLLM engines
(one GB200x4 tray each, 128 GPUs total) for the tool-conversion repair and later
synthesis consumers. See pool.py for the single source of serving configuration.
"""

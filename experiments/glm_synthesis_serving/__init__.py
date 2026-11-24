# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standing GLM-5.3-Flash data-synthesis serving pool on cw-us-east-08a.

Independent TP4 vLLM endpoints (one GB200x4 tray each, 128 GPUs total) feed
tool conversion and synthesis through a validated client-side pool manifest.
See pool.py for the pinned serving configuration.
"""

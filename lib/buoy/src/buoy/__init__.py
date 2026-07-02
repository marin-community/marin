# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""buoy — a web viewer for wandb runs, hosted on Iris.

wandb stays the source of truth; buoy mirrors a run's metrics, config, and TPU
profile into a refetchable GCS cache on demand and serves metric plots plus the
real xprof profile UI behind the Iris controller proxy.
"""

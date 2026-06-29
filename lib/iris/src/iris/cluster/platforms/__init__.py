# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime substrate drivers (GCP, Kubernetes, manual hosts).

Per-cloud tooling for both task-runtime substrate (worker/slice provisioning,
remote execution) and cluster bring-up (controller lifecycle, VM provisioning,
the provider factory). Worker-side and controller-side modules are kept
separate (`workers.py` vs `controller.py`); the only layering rule is that
nothing here imports a `TaskBackend` implementation from `backends/` — only the
shared `backends.protocols`/`backends.types` seam. Import from submodules
directly for concrete classes.
"""

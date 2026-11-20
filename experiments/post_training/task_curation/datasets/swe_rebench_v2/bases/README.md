# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ten authored shared base Dockerfiles for the SWE-rebench V2.1 repair.

Authoring rules (ruling 2026-11-17T16:00Z, option (b)):

- Every base carries the upstream ``base_dockerfiles/`` recipes for its member bases
  (vendored at ``upstream/`` provenance pin ``3beb41a4``), unioned where a group merges
  several upstream bases, with per-version toolchains installed side by side.
- Every base carries the declared interpreter-provisioning step: a ``python3 >= 3.9`` on
  PATH for the shared-source controller bootstrap (apt install when the upstream image
  lacks one), asserted at build time. Disclosed per-base deviation; census evidence in
  ``swe-rebench-base-python-census.json``.
- Every base is one Daytona snapshot identity: identical bytes per group, no per-task
  material. The task-start setup (converter ``setup_command``) selects the row's declared
  toolchain version inside the base.
- Build-time guards fail the build on any broken assumption rather than shipping a base
  that would fail silently at task start.

Fidelity gate: before the full population run, a seeded sample per group (>= 20 tasks per
base, all 10 groups) runs oracle + nop on BOTH the upstream digest-pinned image and the
condensed env; identical F2P/P2P outcomes are required. A mismatching group splits further
or keeps its upstream image — tests are never weakened.
"""

GROUP_COUNT = 10

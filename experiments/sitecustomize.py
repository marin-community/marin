# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime environment import-order hardening for Ray jobs.

Ray's pip runtime env uses a virtualenv that (by default) includes system
site-packages. On some clusters, the base environment is an Anaconda install
with preinstalled compiled wheels (e.g. pandas) that can be ABI-incompatible
with the numpy version pulled into the runtime env.

Python imports `sitecustomize` automatically if it is importable on `sys.path`.
Our Ray runtime env sets `PYTHONPATH` to include `experiments/`, so this module
is present only for Ray jobs (and is not imported for typical local runs).
"""

from __future__ import annotations

import sys


def _looks_like_ray_runtime() -> bool:
    return any("/tmp/ray/session_" in p or "runtime_resources" in p for p in sys.path if isinstance(p, str))


def _harden_sys_path() -> None:
    if not _looks_like_ray_runtime():
        return

    def _is_preferred(p: str) -> bool:
        return ("runtime_resources/pip" in p) or ("/virtualenv/" in p and "site-packages" in p)

    def _is_demoted(p: str) -> bool:
        return ("anaconda3" in p and "site-packages" in p) or ("conda" in p and "site-packages" in p)

    preferred: list[str] = []
    rest: list[str] = []
    demoted: list[str] = []

    for p in sys.path:
        if not isinstance(p, str) or not p:
            rest.append(p)
            continue
        if _is_preferred(p):
            preferred.append(p)
        elif _is_demoted(p):
            demoted.append(p)
        else:
            rest.append(p)

    # Preserve relative ordering within each partition, while avoiding duplicates.
    seen: set[str] = set()
    new_path: list[str] = []
    for p in [*preferred, *rest, *demoted]:
        if not isinstance(p, str) or not p:
            new_path.append(p)
            continue
        if p in seen:
            continue
        seen.add(p)
        new_path.append(p)

    sys.path[:] = new_path


_harden_sys_path()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``ray.exceptions`` -> ``fakeray.exceptions``.

A real module file (not a ``__getattr__`` shim) so that smallpond's
``import ray.exceptions`` resolves. Re-exports the Ray-compatible names.
"""

from fakeray.exceptions import (  # noqa: F401
    GetTimeoutError,
    RayError,
    RayTaskError,
    RuntimeEnvSetupError,
)

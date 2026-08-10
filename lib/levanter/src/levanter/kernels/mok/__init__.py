# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Optional JAX adapter for the Mixture-of-Kittens BF16 EP kernel."""

from levanter.kernels.mok.availability import (
    MokPreflightStatus as MokPreflightStatus,
    mok_preflight_status as mok_preflight_status,
    require_mok_available as require_mok_available,
)
from levanter.kernels.mok.ffi import mok_bf16 as mok_bf16
from levanter.kernels.mok.runtime import (
    MokBf16Config as MokBf16Config,
    MokRuntimeHandle as MokRuntimeHandle,
    close_mok_runtime as close_mok_runtime,
    initialize_mok_runtime as initialize_mok_runtime,
    mok_runtime_initialized as mok_runtime_initialized,
)

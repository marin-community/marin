# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

# DIAGNOSTIC: log every pool allocation and free on the named ranks. This must run before the PJRT
# plugin loads, because its logging initializer reads the environment once.
if os.environ.get("IRIS_MULTIGPU_PROCESS_INDEX") in os.environ.get("MEMGROW_VLOG_RANKS", "").split(","):
    os.environ["TF_CPP_VMODULE"] = "gpu_cudamallocasync_allocator=10"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

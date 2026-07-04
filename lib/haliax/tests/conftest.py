# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import os

# Must be set before JAX initializes its backend. Simulates 8 CPU devices so
# sharding tests guarded by skip_if_not_enough_devices(2|4) actually run.
os.environ.setdefault("JAX_NUM_CPU_DEVICES", "8")

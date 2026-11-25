# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import platform

platform.machine = lambda: os.environ["FIXTURE_ARCHITECTURE"]

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CLI entry point for reviewed Marin IaC imports."""

import os
import sys

# Pulumi Automation starts gRPC threads before invoking subprocesses. Its fork
# hook is unnecessary because each child immediately execs pulumi or gcloud.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from iac.import_command import main

if __name__ == "__main__":
    main()

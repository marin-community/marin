# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Entry point so ``python -m iris.cluster.controller.replay`` works."""

from iris.cluster.controller.replay.run import main

if __name__ == "__main__":
    raise SystemExit(main())

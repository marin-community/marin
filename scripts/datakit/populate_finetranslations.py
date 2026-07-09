# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Populate the finetranslations normalized output under the active MARIN_PREFIX.

Runs the full download -> fold-parallel-text -> normalize chain; already-done
sub-steps skip via the on-disk step cache.
"""

from marin.datakit.sources import all_sources
from marin.execution.step_runner import StepRunner
from rigging.log_setup import configure_logging


def main() -> None:
    configure_logging()
    StepRunner().run([all_sources()["finetranslations"].normalized])


if __name__ == "__main__":
    main()

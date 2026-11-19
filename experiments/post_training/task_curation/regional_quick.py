# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the documented curation driver with local Zephyr inside a regional Iris CPU job."""

import os

from experiments.post_training.task_curation.driver import main
from experiments.post_training.task_curation.local import local_client

if __name__ == "__main__":
    if not os.environ.get("IRIS_TASK_ID"):
        raise RuntimeError("Regional local-Zephyr preparation requires an Iris CPU job")
    with local_client():
        main()

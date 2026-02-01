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

"""Test zephyr pipeline running on Iris via fray v2.

Exercises the full ZephyrContext flow: coordinator actor, worker actors, and task execution.
"""

import json
import logging

from zephyr.dataset import Dataset
from zephyr.execution import get_default_zephyr_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    ds = Dataset.from_list([{"text": f"hello-{i}"} for i in range(10)])
    # Use a lambda to avoid module-not-found issues on workers
    ds = ds.map(lambda record: {**record, "text": record["text"].upper()})

    ctx = get_default_zephyr_context()
    results = list(ctx.execute(ds))
    logger.info("Got %d results", len(results))
    for r in results:
        logger.info("  %s", json.dumps(r))
    assert len(results) == 10
    assert all(r["text"].startswith("HELLO-") for r in results)
    print(f"SUCCESS: {len(results)} records processed")

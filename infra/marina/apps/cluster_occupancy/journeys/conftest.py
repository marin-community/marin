# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Journeys for the Cluster Occupancy app run against fakes for both sources.

The Finelog side uses an embedded hub seeded with the fixture scenario; the
live side serves the same scenario's rows through ``OCCUPANCY_LIVE_ROWS_FILE``,
the env seam the packaged backend honors inside the kernel. No network: the
walk is entirely synthetic users.
"""

import atexit
import json
import os
import tempfile

from finelog.client import LogClient
from finelog.embedded import require_embedded_server

from infra.marina.tests import cluster_occupancy_fixtures as fixtures

fixture_path = tempfile.mkstemp(prefix="occupancy-live-rows-", suffix=".json")[1]
with open(fixture_path, "w") as handle:
    json.dump(fixtures.live_rows(), handle)
os.environ["OCCUPANCY_LIVE_ROWS_FILE"] = fixture_path
atexit.register(os.unlink, fixture_path)

if "OCCUPANCY_FINELOG_URL" not in os.environ:
    server = require_embedded_server()(log_dir=tempfile.mkdtemp(prefix="occupancy-journey-hub-"))
    client = LogClient.connect(f"http://127.0.0.1:{server.port}")
    fixtures.seed_occupancy_fixtures(client)
    os.environ["OCCUPANCY_FINELOG_URL"] = f"http://127.0.0.1:{server.port}"
    os.environ.pop("OCCUPANCY_IAP_CLUSTER", None)
    atexit.register(client.close)
    atexit.register(server.stop)

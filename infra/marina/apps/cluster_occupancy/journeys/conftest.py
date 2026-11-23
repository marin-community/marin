# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Journeys for the Cluster Occupancy app run against fakes for both sources.

The Finelog side uses an embedded hub seeded with the fixture scenario; the
live side reads fixture rows through ``OCCUPANCY_LIVE_ROWS_FILE``, the same
env seam the packaged backend honors inside the kernel. No network: the walk
is entirely synthetic users.
"""

import atexit
import json
import os
import tempfile

from finelog.client import LogClient
from finelog.embedded import require_embedded_server

from infra.marina.tests import cluster_occupancy_fixtures as fixtures

LIVE_ROWS = {
    fixtures.CLUSTER_A: [
        {"root_job_id": "/alice/train-coordinator", "user_id": fixtures.ALICE, "job_id": "job-1",
         "priority_band": 1, "res_device_json": '{"gpu": {"variant": "GB200", "count": 4}}',
         "node_name": "n1", "started_at_ms": int(fixtures._at(-180).timestamp() * 1000),
         "topology_json": f'{{"ds.coreweave.com/nvlink.domain": "{fixtures.RACK_122}"}}'},
        {"root_job_id": "/alice/train-coordinator", "user_id": fixtures.ALICE, "job_id": "job-1",
         "priority_band": 1, "res_device_json": '{"gpu": {"variant": "GB200", "count": 4}}',
         "node_name": "n1", "started_at_ms": int(fixtures._at(-180).timestamp() * 1000),
         "topology_json": f'{{"ds.coreweave.com/nvlink.domain": "{fixtures.RACK_122}"}}'},
        {"root_job_id": "/bob/eval-coordinator", "user_id": fixtures.BOB, "job_id": "job-2",
         "priority_band": 3, "res_device_json": '{"gpu": {"variant": "GB200", "count": 4}}',
         "node_name": "n2", "started_at_ms": int(fixtures._at(-30).timestamp() * 1000),
         "topology_json": f'{{"ds.coreweave.com/nvlink.domain": "{fixtures.RACK_122}"}}'},
    ],
    fixtures.CLUSTER_B: [
        {"root_job_id": "/carol/prod-east", "user_id": fixtures.CAROL, "job_id": "job-5",
         "priority_band": 1, "res_device_json": '{"gpu": {"variant": "H100", "count": 8}}',
         "node_name": "g1", "started_at_ms": int(fixtures._at(-40).timestamp() * 1000),
         "topology_json": "{}"},
    ],
    fixtures.CLUSTER_UNMIRRORED: [
        {"root_job_id": "/dave/train", "user_id": "dave", "job_id": "/dave/train",
         "priority_band": 1, "res_device_json": '{"gpu": {"variant": "H100", "count": 8}}',
         "node_name": "r1", "started_at_ms": int(fixtures._at(-90).timestamp() * 1000),
         "topology_json": "{}"},
    ],
}

fixture_path = tempfile.mkstemp(prefix="occupancy-live-rows-", suffix=".json")[1]
with open(fixture_path, "w") as handle:
    json.dump(LIVE_ROWS, handle)
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

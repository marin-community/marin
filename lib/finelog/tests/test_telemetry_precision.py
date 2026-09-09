# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact scalar readback through the native JSON ingestion path."""

import json
import math

from finelog.client import LogClient
from finelog.embedded import require_embedded_server
from rigging import telemetry


def test_integral_metric_values_preserve_digest_bits_through_native_ingestion(tmp_path):
    server = require_embedded_server()(log_dir=str(tmp_path / "finelog"))
    client = LogClient.connect(server.address)
    values = [3970228113034014.0, -3970228113034014.0, float(2**53), 0.5, -0.0]
    telemetry.shutdown()
    try:
        telemetry.configure(endpoint=server.address + "/v1/telemetry", service="marinskyrl")
        metric = telemetry.histogram("training_metric_value")
        for index, value in enumerate(values):
            metric.record(value, attributes={"metric": "consumed/uid_digest_u52", "step": str(index)})
        assert telemetry.flush(timeout=5.0)
        assert telemetry.runtime_status().lost_records == 0
        rows = client.query(
            "SELECT attributes_json,value FROM \"telemetry_v1.marinskyrl\" WHERE name='training_metric_value'"
        ).to_pylist()
        observed = {int(json.loads(row["attributes_json"])["step"]): row["value"] for row in rows}
        assert len(rows) == len(values)
        assert [observed[index] for index in range(len(values))] == values
        assert math.copysign(1.0, observed[4]) == -1.0
    finally:
        telemetry.shutdown()
        client.close()
        server.stop()

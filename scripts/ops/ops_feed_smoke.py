# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Emit one bounded ops event through the task's regional Iris route."""

from finelog.client import LogClient
from finelog.ops_feed import Severity, emit
from iris.runtime.telemetry import resolve


def main() -> None:
    runtime = resolve()
    if runtime is None:
        raise RuntimeError("run this smoke probe inside an Iris CPU job")
    client = LogClient.connect(runtime.endpoint, resolver=runtime.resolver)
    try:
        event_id = emit(
            "ops-feed-smoke",
            "iris-cpu",
            Severity.INFO,
            "regional-write-probe",
            {"route": "iris-runtime"},
            client=client,
        )
        print(event_id)
    finally:
        client.close()


if __name__ == "__main__":
    main()

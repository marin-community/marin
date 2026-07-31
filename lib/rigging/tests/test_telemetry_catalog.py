# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from rigging import telemetry
from rigging import telemetry_catalog


def test_catalog_sync_check_accepts_python_package_mirror():
    package_catalog = Path(telemetry_catalog.__file__).with_name("telemetry_catalog.v1.json")
    repository_root = Path(__file__).resolve().parents[3]
    finelog_catalog = repository_root / "lib/finelog/rust/telemetry_catalog.v1.json"

    assert package_catalog.read_bytes() == finelog_catalog.read_bytes()
    subprocess.run(
        [sys.executable, "scripts/sync_telemetry_catalog.py", "--check"],
        cwd=repository_root,
        check=True,
    )


def test_contract_example_descriptors_are_cataloged():
    example_meter = telemetry.meter(
        scope="skyrl.inference",
        owner="skyrl",
        default_cadence=timedelta(seconds=15),
    )

    requests = example_meter.counter(
        "requests",
        description="Completed inference requests",
        unit="{request}",
        attributes=(telemetry.AttributeSpec("outcome", ("success", "failure")),),
    )
    request_duration = example_meter.histogram(
        "request_duration",
        description="Inference request latency",
        unit="s",
        attributes=(telemetry.AttributeSpec("outcome", ("success", "failure")),),
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    )
    queue_depth = example_meter.gauge(
        "queue_depth",
        description="Requests waiting for inference",
        unit="{request}",
    )

    assert requests.descriptor is not None
    assert request_duration.descriptor is not None
    assert queue_depth.descriptor is not None
    assert any(event.event_name == "skyrl.worker.ready" for event in telemetry_catalog.load_catalog().events)


def test_metric_declaration_must_match_catalog_exactly():
    handle = telemetry.meter(
        scope="telemetry.runtime",
        owner="rigging",
        default_cadence=timedelta(seconds=10),
    ).counter(
        "emissions",
        description="A descriptor that is not the catalog descriptor",
        unit="{emission}",
        attributes=(
            telemetry.AttributeSpec("signal", ("metric", "event", "log", "artifact")),
        ),
        cardinality_limit=4,
        maturity=telemetry.Maturity.STABLE,
    )

    assert handle.descriptor is None

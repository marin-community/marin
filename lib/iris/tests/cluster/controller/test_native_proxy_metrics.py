# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from iris.cluster.controller.native_proxy_metrics import NativeProxyMetricsCollector


class _Proxy:
    rpc_metrics_json = json.dumps(
        {
            "series": [
                {
                    "service": "iris.cluster.ControllerService",
                    "method": "ListJobs",
                    "upstream": "controller",
                    "requests": 3,
                    "responses": {"200": 2, "500": 1},
                    "in_flight": 1,
                    "latency_buckets": [["0.005", 1], ["+Inf", 3]],
                    "latency_sum_seconds": 0.012,
                }
            ]
        }
    )


def test_native_proxy_metrics_collector_preserves_prometheus_counter_gauge_and_histogram_samples():
    collector = NativeProxyMetricsCollector()
    collector.set_proxy(_Proxy())

    samples = {sample.name: sample for family in collector.collect() for sample in family.samples}

    assert samples["iris_rpc_requests_total"].value == 3
    assert samples["iris_rpc_responses_total"].labels["status"] == "500"
    assert samples["iris_rpc_in_flight"].value == 1
    assert samples["iris_rpc_duration_seconds_bucket"].labels["le"] in {"0.005", "+Inf"}
    assert samples["iris_rpc_duration_seconds_count"].value == 3

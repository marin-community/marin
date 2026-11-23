# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small native-command substitute for capture tests, not a performance benchmark."""

import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
output = Path(args[args.index("--output-artifact-dir") + 1])
duration = int(args[args.index("--benchmark-duration") + 1])
metrics_url = args[args.index("--server-metrics") + 1]
output.mkdir(parents=True)
(output / "profile_export_aiperf.json").write_text(
    json.dumps(
        {
            "metadata": {
                "submission_valid": True,
                "metric_duration_coverage": [
                    {
                        "phase_name": "profiling",
                        "expected_duration_seconds": duration,
                        "required_ratio": 0.9,
                        "ttft_ratio": 1.0,
                        "inter_token_latency_ratio": 1.0,
                    }
                ],
            },
            "was_cancelled": False,
        }
    )
)
(output / "profile_export.jsonl").write_text(
    '{"metadata":{"benchmark_phase":"warmup"},"error":{"message":"fixture warmup failure"}}\n'
    '{"metadata":{"benchmark_phase":"profiling"}}\n'
)
(output / "server_metrics_export.json").write_text(
    json.dumps({"summary": {"endpoints_successful": [metrics_url]}, "metrics": {"vllm:num_requests_running": {}}})
)
raise SystemExit(int(os.environ.get("CAPTURE_FIXTURE_EXIT", "0")))

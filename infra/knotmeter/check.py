# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a running dashboard with authentic collector bundles."""

import argparse
import collections
import gzip
import hashlib
import json
import math
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--bundle", action="append", type=Path, required=True)
    parser.add_argument(
        "--source-runs", type=Path, help="Saved GitHub attempt responses, keyed by run ID; otherwise use gh"
    )
    args = parser.parse_args()
    sources = json.loads(args.source_runs.read_text()) if args.source_runs else {}

    def api(path: str, **query):
        url = args.url.rstrip("/") + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=60) as response:
            body = response.read()
        return json.loads(gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body)

    root = Path(__file__).resolve().parent
    build = api("/knotmeter-build.json")
    assert build["source"] == json.loads((root / "settings.json").read_text())["app_source"]
    assert build["patch_sha256"] == hashlib.sha256((root / "app.patch").read_bytes()).hexdigest()
    profile = hashlib.sha256((root / "build.mjs").read_bytes() + (root / "layout.tsx").read_bytes()).hexdigest()
    assert build["profile_sha256"] == profile
    report = {"build": build, "points": []}
    models = {
        "snowball011": ("Marin-Snowball-67B-A2B-v0.1.1", 8),
        "hero144k": ("Marin-Hero-535B-step144000", 16),
    }
    for directory in args.bundle:
        manifest = json.loads((directory / "agentx-source.json").read_text())
        source = manifest["source"]
        actual = sources.get(str(source["run_id"]))
        if actual is None:
            actual = json.loads(
                subprocess.check_output(
                    [
                        "gh",
                        "api",
                        f"repos/{source['repository']}/actions/runs/{source['run_id']}/attempts/{source['run_attempt']}",
                    ]
                )
            )
        assert actual["head_sha"] == source["head_sha"]
        assert actual["run_attempt"] == source["run_attempt"]
        run_url = (
            f"https://github.com/{source['repository']}/actions/runs/{source['run_id']}/attempts/{source['run_attempt']}"
        )
        for aggregate_path in directory.glob("bmk_*/agg.json"):
            aggregate = json.loads(aggregate_path.read_text())
            model, gpus = models[aggregate["infmax_model_prefix"]]
            throughput = aggregate["request_metrics"]["throughput"]
            output = throughput["per_gpu"]["output_tput_tps"]
            assert aggregate["num_gpus"] == gpus
            assert math.isclose(output, throughput["output"]["tokens_per_second"] / gpus, abs_tol=0.00001)
            rows = api("/api/v1/benchmarks", model=model)
            row = next(r for r in rows if r["run_url"] == run_url and r["conc"] == aggregate["conc"])
            point_id = int(row["id"])
            assert row["num_decode_gpu"] == gpus
            assert row["image"] == aggregate["image"]
            assert row["recipe_fingerprint"] == aggregate["recipe_fingerprint"]
            assert row["metrics"]["output_tput_per_gpu"] == output
            assert row["metrics"]["p90_ttft"] == aggregate["request_metrics"]["latency"]["ttft"]["p90"]
            history = api("/api/v1/benchmarks/history", model=model, benchmarkType="agentic_traces")
            entry = next(r for r in history if r["run_url"] == run_url)
            assert entry["date"] == actual["created_at"][:10]
            siblings = api("/api/v1/benchmark-siblings", id=point_id)
            assert siblings["sku"]["github_run_id"] == source["run_id"]
            sibling = next(s for s in siblings["siblings"] if s["is_current"])
            assert sibling["num_decode_gpu"] == gpus and sibling["has_trace"]
            timeline = api("/api/v1/request-timeline", id=point_id)
            raw_path = (
                directory / aggregate_path.parent.name.removeprefix("bmk_") / "aiperf_artifacts/profile_export.jsonl"
            )
            raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
            assert len(timeline["requests"]) == len(raw)
            expected = collections.defaultdict(list)
            observed = collections.defaultdict(list)
            for record in raw:
                meta = record["metadata"]
                expected[meta["conversation_id"], meta["turn_index"], meta["benchmark_phase"]].append(meta)
            for request in timeline["requests"]:
                observed[request["cid"], request["ti"], request["phase"]].append(request)
            assert expected.keys() == observed.keys()
            maximum_rounding = 0
            for key, records in expected.items():
                requests = observed[key]
                assert len(records) == len(requests)
                pairs = zip(
                    sorted(records, key=lambda r: r["request_start_ns"]),
                    sorted(requests, key=lambda r: r["start"]),
                    strict=True,
                )
                for original, rendered in pairs:
                    for original_field, rendered_field in [("request_start_ns", "start"), ("request_end_ns", "end")]:
                        difference = abs(original[original_field] - (timeline["startNs"] + rendered[rendered_field]))
                        maximum_rounding = max(maximum_rounding, difference)
                        assert difference <= 1024, (key, original_field, difference)
            profiling = sum(r["metadata"]["benchmark_phase"] == "profiling" and not r.get("error") for r in raw)
            point = next(p for p in manifest["points"] if p["manifest"]["concurrency"] == aggregate["conc"])
            assert profiling == point["receipt"]["validation"]["profiling_requests"] == sibling["total_requests"]
            view = api(
                "/api/v1/views/inference",
                model=model,
                sequence="agentic-traces",
                precisions="bf16",
                metric="y_outputTputPerGpu",
                percentile="p90",
                optimal="false",
            )
            plotted = next(p for series in view["series"] for p in series["points"] if p["id"] == point_id)
            assert plotted["y"] == output and plotted["runId"] == source["run_id"]
            report["points"].append(
                {
                    "id": point_id,
                    "model": model,
                    "source": source,
                    "date": entry["date"],
                    "gpus": gpus,
                    "output_tokens_per_second_per_gpu": output,
                    "runtime": row["image"],
                    "recipe": row["recipe_fingerprint"],
                    "timeline_requests": len(raw),
                    "profiling_requests": profiling,
                    "maximum_timestamp_rounding_ns": maximum_rounding,
                }
            )
    assert report["points"], "No benchmark points checked"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

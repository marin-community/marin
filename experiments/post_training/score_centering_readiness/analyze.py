# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize real-worker windows without changing or selecting observations."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("artifact", type=Path)
args = parser.parse_args()
raw = args.artifact.read_bytes()
data = json.loads(raw)
assert data.get("complete") is True or data.get("backend_checks_completed") is True, "Qualification is incomplete"
assert data["candidate"] == "2fb64c1c2b5e8692f996942704383be68a242587"
windows = data["windows"]
modes = ("ordinary", "telemetry", "capture", "sc")
expected = 2 if data["workload"] == "qwen" else 4
assert len(windows) == 4 * expected
assert data["applied_updates"] == (9 if data["workload"] == "qwen" else 18)
assert len({row["tokens"] for row in windows}) == 1
assert len({row["loss_tokens"] for row in windows}) == 1
assert len({tuple(row["sequence_shape"]) for row in windows}) == 1
for row in windows:
    assert row["status"]["policy_update_steps"] == 1
    assert row["status"]["skipped_steps"] == 0
    assert math.isfinite(row["status"]["raw_grad_norm"]) and row["status"]["raw_grad_norm"] > 0
    assert row["generator_applied_updates"] == 0 and row["generator_policy_step"] == 1
    row["training_seconds"] = row["forward_seconds"] + row["update_seconds"]


def devices(value):
    if isinstance(value, dict):
        if "peak_allocated" in value:
            yield value
        else:
            for nested in value.values():
                yield from devices(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from devices(nested)


summary = {
    "workload": data["workload"],
    "candidate": data["candidate"],
    "artifact_sha256": hashlib.sha256(raw).hexdigest(),
    "raw_complete_flag": data.get("complete", False),
    "recovery": data.get("recovery"),
    "applied_updates": data["applied_updates"],
    "prompt_sha256": data["prompt_sha256"],
    "prompt_length_min_max": [min(data["prompt_lengths"]), max(data["prompt_lengths"])],
    "response_hashes": sorted({row["response_ids_sha256"] for row in windows}),
    "tokens_per_window": windows[0]["tokens"],
    "loss_tokens_per_window": windows[0]["loss_tokens"],
    "sequence_shape": windows[0]["sequence_shape"],
    "setup_timings": data["setup_timings"],
    "runtime": data["runtime"],
    "modes": {},
    "comparisons": {},
    "uncertainty": (
        "Paired log-ratio Student-t intervals across measured repetitions, with first repetition excluded per mode. "
        "One allocation, rotating mode order, frozen sampler and sequential learner updates. "
        "Intervals assume independent approximately normal log differences; "
        "no between-allocation uncertainty or learning-benefit claim."
    ),
}
for mode in modes:
    rows = [row for row in windows if row["mode"] == mode and not row["warmup"]]
    out = {"measured_windows": len(rows)}
    for field in (
        "generation_seconds",
        "preparation_seconds",
        "forward_seconds",
        "update_seconds",
        "training_seconds",
        "total_seconds",
    ):
        values = [row[field] for row in rows]
        out[field] = {"mean": statistics.mean(values), "min": min(values), "max": max(values), "values": values}
    out["response_tokens_per_second"] = sum(row["tokens"] for row in rows) / sum(row["total_seconds"] for row in rows)
    out["loss_tokens_per_second"] = sum(row["loss_tokens"] for row in rows) / sum(row["total_seconds"] for row in rows)
    for role in ("learner", "inference"):
        memory = [device for row in rows for device in devices(row[f"{role}_memory"])]
        assert memory
        out[f"{role}_memory"] = {
            "observations": len(memory),
            **{
                field: max(device[field] for device in memory)
                for field in ("peak_allocated", "peak_reserved", "sampled_device_peak_used")
            },
        }
    summary["modes"][mode] = out

for treatment, control in (
    ("telemetry", "ordinary"),
    ("capture", "telemetry"),
    ("sc", "capture"),
    ("sc", "telemetry"),
    ("sc", "ordinary"),
):
    out = {}
    for field in ("total_seconds", "training_seconds", "update_seconds", "generation_seconds"):
        treated = {row["repetition"]: row[field] for row in windows if row["mode"] == treatment and not row["warmup"]}
        baseline = {row["repetition"]: row[field] for row in windows if row["mode"] == control and not row["warmup"]}
        assert treated.keys() == baseline.keys()
        log_diffs = [math.log(treated[key] / baseline[key]) for key in sorted(treated)]
        center = statistics.mean(log_diffs)
        interval = None
        if len(log_diffs) > 1:
            critical = {2: 4.302652729749464, 3: 3.182446305284263}[len(log_diffs) - 1]
            half = critical * statistics.stdev(log_diffs) / math.sqrt(len(log_diffs))
            interval = [100 * math.expm1(center - half), 100 * math.expm1(center + half)]
        out[field] = {
            "paired_percent": 100 * math.expm1(center),
            "paired_t_95_percent": interval,
            "ratio_of_means_percent": 100 * (statistics.mean(treated.values()) / statistics.mean(baseline.values()) - 1),
        }
    summary["comparisons"][f"{treatment}_versus_{control}"] = out

summary["qualification"] = {
    key: value
    for key, value in data["qualification"].items()
    if key
    not in (
        "prefix_replay",
        "temperature_0_7_prefix_replay",
        "stale_prefix_replay",
        "sampled_diagnostics",
        "native_training",
    )
}
for key in ("prefix_replay", "temperature_0_7_prefix_replay", "stale_prefix_replay"):
    summary["qualification"][key] = {k: v for k, v in data["qualification"][key].items() if k != "samples"}
sampled = data["qualification"]["sampled_diagnostics"]
summary["qualification"]["sampled_diagnostics"] = {
    key: value for key, value in sampled["status"].items() if "score_centering/" in key
}
if "native_training" in data["qualification"]:
    native = data["qualification"]["native_training"]
    summary["qualification"]["native_training"] = {k: v for k, v in native.items() if k != "uids"}
target = args.artifact.with_suffix(".summary.json")
target.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
with args.artifact.with_suffix(".windows.csv").open("w") as stream:
    fields = (
        "mode",
        "repetition",
        "warmup",
        "generation_seconds",
        "preparation_seconds",
        "forward_seconds",
        "update_seconds",
        "training_seconds",
        "total_seconds",
        "tokens",
        "loss_tokens",
        "applied_updates",
        "response_ids_sha256",
    )
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(windows)
print(
    json.dumps(
        {
            "summary": str(target),
            "applied_updates": data["applied_updates"],
            "response_hashes": len(summary["response_hashes"]),
            "modes": {
                key: {
                    "total_seconds": value["total_seconds"]["mean"],
                    "training_seconds": value["training_seconds"]["mean"],
                    "tokens_per_second": value["response_tokens_per_second"],
                }
                for key, value in summary["modes"].items()
            },
            "comparisons": summary["comparisons"],
        },
        indent=2,
    )
)

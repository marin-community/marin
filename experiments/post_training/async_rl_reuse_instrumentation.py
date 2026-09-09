# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize the measured reuse ladder from bound history, native and quality receipts.

Input filenames follow the retained E4.1 audit conventions. No remote data is read.
Update means are descriptive; this reducer does not add adjacent contrasts.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def summarize(input_directory: Path, native_directory: Path, output: Path) -> None:
    cache = input_directory
    arms = []

    def sha(x):
        return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    for n in [1, 2, 4, 8, 16]:
        for seed in [17, 29]:
            if n <= 2:
                hf = cache / f"async-v2-e41-n{n}-seed{seed}-selected-history.json"
                rf = cache / f"async-v2-e41-n{n}-seed{seed}-scientific-audit.json"
            elif (n, seed) == (16, 17):
                hf = cache / "e41-n16-retry-n16-seed17-history.json"
                rf = cache / "async-v2-e41-n16-retry-n16-seed17-scientific-audit.json"
            else:
                hf = cache / f"e41-n{n}-seed{seed}-history.json"
                rf = cache / f"async-v2-e41-n{n}-seed{seed}-scientific-audit.json"
            h = json.loads(hf.read_text())
            r = json.loads(rf.read_text())
            rows = h["selected_rows"] if isinstance(h, dict) else h
            source_sha = h["source_canonical_sha256"] if isinstance(h, dict) else sha(h)
            assert source_sha == r["history_sha256"], (n, seed, "history binding")
            merged = {}
            for row in rows:
                step = row.get("global_step", row.get("trainer/global_step"))
                if step is not None:
                    target = merged.setdefault(int(step), {})
                    for key, value in row.items():
                        if key in target and target[key] != value:
                            raise ValueError("Conflicting duplicate history metric " + key)
                        target[key] = value
            cf = native_directory / (
                f"e41-n{n}-seed{seed}-finelog-final.json"
                if (n, seed) != (16, 17)
                else "e41-n16-retry-n16-seed17-finelog-final.json"
            )
            cap = json.loads(cf.read_text())
            assert sha(cap) == r["native_telemetry"]["capture_sha256"], (n, seed, "native capture receipt binding")
            native = {}
            for native_row in cap["results"]["scalars"]:
                key = (native_row["step"], native_row["metric"])
                assert key not in native
                native[key] = native_row["value"]
            series = []
            for step in range(1, 96 // n + 1):
                row = merged[step]
                for i in range(n):
                    prefix = f"policy/by_update/{i}/"
                    x = {"optimizer_update": (step - 1) * n + i + 1, "batch": step, "within_batch_update": i}
                    for key in [
                        "stale/log_ratio_mean",
                        "stale/abs_log_ratio_mean",
                        "stale/ess_fraction",
                        "stale/kl_k3",
                        "stale/selected_tokens",
                        "grad_cosine",
                        "grad_cosine_valid",
                        "raw_grad_norm",
                    ]:
                        v = row[prefix + key]
                        assert isinstance(v, (int, float)) and math.isfinite(v)
                        assert math.isclose(v, native[step, prefix + key], rel_tol=1e-9, abs_tol=1e-12), (
                            n,
                            seed,
                            step,
                            key,
                            "native mismatch",
                        )
                        x[key] = v
                    series.append(x)
            assert len(series) == 96
            stats = {}
            for key in [
                "stale/log_ratio_mean",
                "stale/abs_log_ratio_mean",
                "stale/ess_fraction",
                "stale/kl_k3",
                "raw_grad_norm",
                "grad_cosine",
            ]:
                values = [x[key] for x in series if key != "grad_cosine" or x["grad_cosine_valid"] == 1]
                stats[key] = {
                    "valid_updates": len(values),
                    "mean": statistics.mean(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }
            arms.append(
                {
                    "minibatches": n,
                    "seed": seed,
                    "native_capture_file": str(cf),
                    "native_capture_bytes_sha256": hashlib.sha256(cf.read_bytes()).hexdigest(),
                    "native_scalar_joins": 96 * 8,
                    "history_file": str(hf),
                    "history_bytes_sha256": hashlib.sha256(hf.read_bytes()).hexdigest(),
                    "full_history_canonical_sha256": source_sha,
                    "scientific_receipt_file": str(rf),
                    "scientific_receipt_bytes_sha256": hashlib.sha256(rf.read_bytes()).hexdigest(),
                    "updates": series,
                    "summary": stats,
                    "clipped_entropy": None,
                }
            )
    qf = cache / "async-v2-e41-full-results.json"
    qb = qf.read_bytes()
    readback = json.loads((cache / "async-v2-e41-full-results-readback.json").read_text())
    binding = next(x for x in readback if x["name"] == "results.json")
    assert hashlib.sha256(qb).hexdigest() == binding["sha256"] and len(qb) == binding["bytes"]
    quality = json.loads(qb)["primary_family"]["results"]
    endpoints = {1: quality["N2-N1"]["reference"], **{n: quality[f"N{n}-N1"]["candidate"] for n in [2, 4, 8, 16]}}
    best = max(endpoints, key=endpoints.get)
    drop = max((endpoints[best] - v) * 100 for n, v in endpoints.items() if n > best)
    out = {
        "quality_results_bytes_sha256": binding["sha256"],
        "quality_endpoints": endpoints,
        "schema": "e41_reuse_instrumentation_summary_v1",
        "arms": arms,
        "scope": (
            "Descriptive equal-weight summary over 96 observed optimizer updates per arm; "
            "cosine uses only valid adjacent-gradient updates. Per-update stale statistics retain "
            "their worker token-pooling semantics. No causal-age, adjacent-N inference or "
            "pooled-token ESS claim. Clipped entropy unavailable."
        ),
        "knee_disposition": {
            "planning_mde_pp": 25.336525786,
            "observed_quality_maximum_N": best,
            "largest_point_drop_from_observed_maximum_to_larger_N_pp": drop,
            "crosses_planning_mde": drop > 25.336525786,
            "statement": (
                f"Largest point drop from observed N{best} to a larger measured N is {drop:.8f} pp. "
                "Compare this descriptive drop with the declared planning MDE. "
                "This is not noninferiority or an adjacent contrast test; "
                "all predeclared versus-N1 contrasts remain separately reported."
            ),
        },
    }
    p = output
    p.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")
    print(
        "E41_REUSE_INSTRUMENTATION_SUMMARY_PASS arms=10 updates=960 sha256=" + hashlib.sha256(p.read_bytes()).hexdigest()
    )
    for a in arms:
        print(
            a["minibatches"],
            a["seed"],
            {k: round(v["mean"], 7) if v["mean"] is not None else None for k, v in a["summary"].items()},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input_directory, args.native_directory, args.output)


if __name__ == "__main__":
    main()

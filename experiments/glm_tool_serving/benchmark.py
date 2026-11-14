# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one isolated, receipt-preserving arm of the real conversion benchmark."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from experiments.tool_normalization.driver import run
from experiments.tool_normalization.throughput import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--thinking-token-budget", type=int, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 4, 8, 16, 32), required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.template.read_text())
    generation = "dedicated-h100-" + args.root.name
    capacity = args.root / "capacity.json"
    capacity.write_text(
        json.dumps(
            {
                "generation": generation,
                "conversion_slots": args.workers,
                "science_slots": 0,
                "expires_at_unix": time.time() + 86400,
                "scope": "Dedicated H100 endpoint only; independent of Orion capacity and leases",
            },
            indent=2,
        )
    )
    config["workers"] = args.workers
    config["model"].update(
        {
            "base_url": args.base_url,
            "model": "glm-5.3",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "thinking_token_budget": args.thinking_token_budget,
            "generation": generation,
            "api_key_file": str(args.api_key_file.resolve()),
            "capacity_file": str(capacity.resolve()),
            "ledger": str((args.root / "ledger.sqlite3").resolve()),
            "context_tokens": 262144,
        }
    )
    path = args.root / "config.json"
    path.write_text(json.dumps(config, indent=2))
    (args.root / "input-pin.json").write_text(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "benchmark_only": True,
                "training_export_adoption": False,
            },
            indent=2,
        )
    )
    run(path, args.manifest, args.root / "run")
    report = summarize(args.root / "run")
    (args.root / "throughput.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

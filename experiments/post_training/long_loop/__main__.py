# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect the local long-loop plan or render worker timing events."""

import argparse
from pathlib import Path

from experiments.post_training.long_loop.timing import read_events, report

STAGES = (
    "resolve/preflight",
    "SFT",
    "committed SFT handoff",
    "RL endpoint/mask preflight",
    "async RL",
    "drain/committed RL handoff",
    "eval endpoint preflight",
    "Micro210",
    "report",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("plumbing", "qualification", "full"), default="plumbing")
    parser.add_argument("--version", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--events", type=Path, nargs="+")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.run:
        parser.error("Cluster execution is not wired: pin SFT/handoff and Snowball Micro210 bindings first.")
    if args.events:
        if args.report is None:
            parser.error("--events requires --report")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report(read_events(args.events), synthetic=args.synthetic))
        return
    print(f"Long-loop {args.version}: {args.profile}; PLAN ONLY, no submissions")
    print(" → ".join(STAGES))
    print("GS8; 64 groups/update; staleness 2; max 256 H100; interactive priority")
    print("Plumbing: 15 SFT updates, 3 RL updates at 24,576 context, 21 eval paths")
    print("Unresolved: source/runtime pins, remote stage bindings, allocation-release and submit reconciliation,")
    print("live evidence collectors, full Micro210 integration, native phase timing adapters.")


if __name__ == "__main__":
    main()

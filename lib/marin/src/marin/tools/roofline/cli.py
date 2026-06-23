# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for the roofline dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marin.tools.roofline.report import build_report, render_markdown, write_json_report, write_markdown_report
from marin.tools.roofline.serve import serve_report


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "summarize":
        report = build_report(
            hardware_name=args.hardware,
            model_preset_name=args.model_preset,
            wandb_run=args.wandb_run,
            profile=args.profile,
        )
        if args.output:
            write_json_report(report, Path(args.output))
        if args.output_md:
            write_markdown_report(report, Path(args.output_md))
        if not args.output and not args.output_md:
            print(render_markdown(report), end="")
        return

    report = build_report(
        hardware_name=args.hardware,
        model_preset_name=args.model_preset,
        wandb_run=args.wandb_run,
        profile=args.profile,
    )
    if args.output:
        write_json_report(report, Path(args.output))
    else:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if args.serve:
        serve_report(report, args.serve)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Grug/MoE roofline dashboard report.")
    _add_common_args(parser)
    parser.add_argument("--output", help="Write the machine-readable report JSON to this path.")
    parser.add_argument("--serve", help="Serve a local dashboard at host:port, for example 127.0.0.1:6070.")
    subparsers = parser.add_subparsers(dest="command")
    summarize = subparsers.add_parser("summarize", description="Write a non-serving roofline summary.")
    _add_common_args(summarize)
    summarize.add_argument("--output", help="Write the machine-readable report JSON to this path.")
    summarize.add_argument("--output-md", help="Write a Markdown summary to this path.")
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb-run", help="W&B run id, entity/project/run id, or run URL.")
    parser.add_argument(
        "--profile", help="Local profile summary, xprof table JSON, xprof table directory, or XPlane path."
    )
    parser.add_argument("--hardware", default="coreweave_h100", help="Hardware preset name.")
    parser.add_argument("--model-preset", default="grug_moe_d2560_may", help="Model preset name.")

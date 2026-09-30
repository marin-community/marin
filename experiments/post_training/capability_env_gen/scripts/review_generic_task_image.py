#!/usr/bin/env python3
"""Freeze a generated image request and obtain a fresh GLM5.3 capture decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from capability_pipeline.generic_image_construction import (
    prepare_construction_capture,
    request_needed,
)
from capability_pipeline.image_plan_review import run_review
from capability_pipeline.synthesis import OMPAgent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", type=Path, required=True)
    parser.add_argument("--capture-tools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--omp", default="omp")
    parser.add_argument("--omp-config", type=Path, required=True)
    parser.add_argument("--session-seconds", type=int, default=1800)
    parser.add_argument("--builder-session-id", action="append", default=[])
    args = parser.parse_args(argv)
    if args.session_seconds <= 0 or not args.omp_config.is_file():
        raise ValueError("review session configuration is invalid")
    if args.output.exists():
        raise ValueError("review output already exists; use a fresh review directory")
    if not request_needed(args.item / "workspace")["needed"]:
        print(json.dumps({"state": "not_required"}))
        return 0
    frozen = prepare_construction_capture(args.item, args.capture_tools)
    agent = OMPAgent(args.omp, "glm-orion/glm-5.3", args.session_seconds,
                     max_continuations=0, config=args.omp_config)
    result = run_review(
        item_root=args.item, plan_path=Path(frozen["plan_path"]),
        review_root=args.output, agent=agent,
        builder_session_ids=set(args.builder_session_id),
    )
    print(json.dumps({**result, "frozen_capture": frozen}))
    return 0 if result["state"] == "approve" else 2


if __name__ == "__main__":
    raise SystemExit(main())

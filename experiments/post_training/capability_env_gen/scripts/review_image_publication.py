#!/usr/bin/env python3
"""Validate a credential-free image publication plan without executing it."""

import argparse
import json
from pathlib import Path

from capability_pipeline.image_publication_contract import validate_publication_contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_publication_contract(plan, workspace=args.workspace)
    print(json.dumps({"state": plan["state"], "images": [x["role"] for x in plan["images"]], "review_blockers": len(plan["review_blockers"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

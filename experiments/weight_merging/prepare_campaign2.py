# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy the frozen campaign recipes with explicit parent and output locations."""

import argparse
import json
from pathlib import Path

TEMPLATES = Path(__file__).parent / "campaign2"
HISTORICAL_OUTPUT = "s3://marin-us-east-02a/marin/weight-merging/campaign2-20261009"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--nupa", required=True)
    parser.add_argument("--bfcl", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--recipes", type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads((TEMPLATES / "avg-01.json").read_text())
    replacements = {
        sources["base"]["path"]: args.base.rstrip("/"),
        sources["donors"][0]["path"]: args.nupa.rstrip("/"),
        sources["donors"][1]["path"]: args.bfcl.rstrip("/"),
        HISTORICAL_OUTPUT: args.output_prefix.rstrip("/"),
    }
    args.recipes.mkdir(parents=True, exist_ok=False)
    for path in sorted(TEMPLATES.glob("*.json")):
        payload = path.read_text()
        if "output" not in json.loads(payload):
            continue
        for old, new in replacements.items():
            payload = payload.replace(json.dumps(old)[1:-1], json.dumps(new)[1:-1])
        json.loads(payload)
        (args.recipes / path.name).write_text(payload)


if __name__ == "__main__":
    main()

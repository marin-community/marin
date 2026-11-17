# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download the frozen campaign recipes and copy them with explicit parent and output locations."""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

RECIPES_REPO = "open-athena/Snowball-67B-A2B-Model-Merging-Artifacts-2026.10.09"
RECIPES_REVISION = "1ef199578aeec9f08e142a8dd760d44658264be4"
RECIPES_SUBFOLDER = "campaign2-recipes"
HISTORICAL_OUTPUT = "s3://marin-us-east-02a/marin/weight-merging/campaign2-20261009"
CALIBRATION_DATA = Path(__file__).parent / "campaign2" / "calibration.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--nupa", required=True)
    parser.add_argument("--bfcl", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--recipes-repo", default=RECIPES_REPO)
    parser.add_argument("--recipes-revision", default=RECIPES_REVISION)
    args = parser.parse_args()
    templates = (
        Path(
            snapshot_download(
                repo_id=args.recipes_repo,
                repo_type="dataset",
                revision=args.recipes_revision,
                allow_patterns=[f"{RECIPES_SUBFOLDER}/*.json", f"{RECIPES_SUBFOLDER}/calibration.jsonl"],
            )
        )
        / RECIPES_SUBFOLDER
    )
    if not CALIBRATION_DATA.exists():
        shutil.copyfile(templates / "calibration.jsonl", CALIBRATION_DATA)
        print(f"Wrote missing calibration data to {CALIBRATION_DATA}")
    sources = json.loads((templates / "avg-01.json").read_text())
    replacements = {
        sources["base"]["path"]: args.base.rstrip("/"),
        sources["donors"][0]["path"]: args.nupa.rstrip("/"),
        sources["donors"][1]["path"]: args.bfcl.rstrip("/"),
        HISTORICAL_OUTPUT: args.output_prefix.rstrip("/"),
    }
    args.recipes.mkdir(parents=True, exist_ok=False)
    for path in sorted(templates.glob("*.json")):
        payload = path.read_text()
        if "output" not in json.loads(payload):
            continue
        for old, new in replacements.items():
            payload = payload.replace(json.dumps(old)[1:-1], json.dumps(new)[1:-1])
        json.loads(payload)
        (args.recipes / path.name).write_text(payload)


if __name__ == "__main__":
    main()

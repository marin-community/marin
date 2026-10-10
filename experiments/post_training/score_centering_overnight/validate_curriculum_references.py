# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate native math references before committing GPU time to curriculum trials."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from skyrl_gym.envs.aime.utils import general_math_answer_reward, normalize_curriculum_answer


def validate(row: dict) -> dict:
    reference = normalize_curriculum_answer(row["reference"])
    try:
        verdict = general_math_answer_reward(reference, reference)
        return {**row, "accepted": verdict.reward == 1.0}
    except Exception as error:
        return {**row, "accepted": False, "error_type": type(error).__name__, "error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"sources": [], "validation": "reference acceptance; not response-quality evidence"}
    for source in args.source:
        frame = pd.read_parquet(source)
        rows = []
        for index, row in frame.iterrows():
            if row["env_class"] != "aime":
                continue
            reference = (row["reward_spec"] or row["reward_model"])["ground_truth"]
            rows.append({"source_row": int(index), "domain": row["data_source"], "reference": reference})
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(validate, rows, chunksize=32))
        rejected = [row for row in results if not row["accepted"]]
        entry = {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "math_references": len(rows),
            "accepted": len(rows) - len(rejected),
            "rejected": rejected,
        }
        report["sources"].append(entry)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(source.name, len(rows), "references;", len(rejected), "rejected", flush=True)


if __name__ == "__main__":
    main()

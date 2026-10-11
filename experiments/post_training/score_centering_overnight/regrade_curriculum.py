# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay retained response grades against original, immutable curriculum rows."""

import argparse
import gzip
import hashlib
import json
import math
import subprocess
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml
from grading_text import grading_text
from skyrl_gym.envs.aime.utils import verify
from skyrl_gym.envs.gsm8k.utils import compute_score
from skyrl_gym.envs.reasoning_gym.scoring import normalize_ground_truth, score_response

SOURCES = {}
PRESERVE_FULL_RESPONSE = False


def load_sources(train: str, validation: str, preserve_full_response: bool) -> None:
    global PRESERVE_FULL_RESPONSE
    SOURCES.update(train=pd.read_parquet(train), eval=pd.read_parquet(validation))
    PRESERVE_FULL_RESPONSE = preserve_full_response


def regrade(record: dict) -> dict:
    phase = record["phase"]
    extras = record["trajectory"]["environment_extras"]
    source_row = int(extras["campaign_source_row"])
    row = SOURCES[phase].iloc[source_row]
    assert row["data_source"] == extras["data_source"]
    environment = row["env_class"]
    expected = (row["reward_spec"] or row["reward_model"])["ground_truth"]
    response = grading_text(record["response"]["text"], record["response"]["messages"], PRESERVE_FULL_RESPONSE)
    try:
        if environment == "aime":
            quality = float(verify(response, expected, allow_general_math_reference=True)[0])
        elif environment == "gsm8k":
            quality = float(compute_score(response, expected, method="strict"))
        elif environment == "reasoning_gym":
            quality = score_response(response, normalize_ground_truth(expected))
        else:
            raise ValueError(f"Unsupported curriculum environment: {environment}")
        assert math.isfinite(quality) and 0 <= quality <= 1
    except Exception as error:
        return {
            "record_id": record["record_id"],
            "phase": phase,
            "step": record["global_step"],
            "source_row": source_row,
            "offline_error": type(error).__name__,
            "error": str(error),
        }
    verdict = record["verification_result"]
    native_error = verdict["status"] == "error" or record["disposition"]["exception_type"] is not None
    native = verdict["passed"] if verdict["passed"] is not None else verdict["score"]
    return {
        "record_id": record["record_id"],
        "phase": phase,
        "step": record["global_step"],
        "source_row": source_row,
        "native_error": native_error,
        "quality": quality,
        "native_quality": None if native_error else float(native),
        "mismatch": not native_error and not math.isclose(float(native), quality, rel_tol=0, abs_tol=1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--validation-source", type=Path, required=True)
    parser.add_argument("--runtime-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    audited = json.loads(args.audit.read_text())
    assert audited["audit_passed"] and audited["run_id"] == config["run"]["id"]
    source_hashes = {
        phase: hashlib.sha256(path.read_bytes()).hexdigest()
        for phase, path in (("train", args.train_source), ("eval", args.validation_source))
    }
    if "sc-snowball" in config["run"]["id"]:
        manifest = json.loads((Path(__file__).parent / "results/snowball_data_manifest.json").read_text())
        expected_hashes = {
            "eval" if row["split"] == "validation" else "train": row["source_sha256"] for row in manifest["membership"]
        }
    else:
        manifest = json.loads((Path(__file__).parent / "results/curriculum_data_manifest.json").read_text())
        expected_hashes = {"train": manifest["source_files"]["train"], "eval": manifest["source_files"]["validation"]}
    assert source_hashes == expected_hashes
    preserve_full_response = config["skyrl"]["generator"].get("preserve_full_response_text", False)
    files = [
        "envs/aime/utils.py",
        "envs/gsm8k/utils.py",
        "envs/reasoning_gym/scoring.py",
        "envs/nemotron_ultra/answer_extraction.py",
    ]
    grader_files = {}
    for file in files:
        relative = "skyrl-gym/skyrl_gym/" + file
        frozen = subprocess.check_output(
            ["git", "show", config["runtime"]["launcher_commit"] + ":" + relative], cwd=args.runtime_checkout
        )
        assert (args.runtime_checkout / relative).read_bytes() == frozen
        grader_files[relative] = hashlib.sha256(frozen).hexdigest()
    records = []
    digest = hashlib.sha256()
    record_ids = set()
    for item in sorted(audited["selected_archives"], key=lambda item: item["Key"]):
        relative = "attempts/" + item["Key"].split("/attempts/", 1)[1]
        with zipfile.ZipFile(args.cache / relative) as archive:
            for name in archive.namelist():
                if not name.startswith("records/"):
                    continue
                raw = gzip.decompress(archive.read(name))
                record = json.loads(raw)
                assert record["record_id"] not in record_ids
                record_ids.add(record["record_id"])
                digest.update(raw)
                records.append(record)
    counts = Counter()
    quality = Counter()
    discrepancies = []
    native_errors = []
    with ProcessPoolExecutor(
        max_workers=4,
        initializer=load_sources,
        initargs=(str(args.train_source), str(args.validation_source), preserve_full_response),
    ) as executor:
        for result in executor.map(regrade, records, chunksize=16):
            key = (result["phase"], result["step"])
            counts[key] += 1
            if result.get("offline_error") or result.get("mismatch"):
                discrepancies.append(result)
            else:
                quality[key] += result["quality"]
            if result.get("native_error"):
                native_errors.append(result)
    report = {
        "run_id": config["run"]["id"],
        "source": config["runtime"]["launcher_commit"],
        "preserve_full_response_text": preserve_full_response,
        "basis": (
            "Every retained response replayed against its original source row using the frozen grading contract; "
            "this is not an independent verifier implementation."
        ),
        "grader_files_sha256": grader_files,
        "source_sha256": source_hashes,
        "record_payloads_sha256": digest.hexdigest(),
        "records": len(records),
        "quality": [
            {
                "phase": phase,
                "completed_updates": step,
                "responses": count,
                "replayed_quality": quality[(phase, step)] / count,
            }
            for (phase, step), count in sorted(counts.items())
        ],
        "discrepancies": discrepancies,
        "native_errors_replayed": native_errors,
        "regrading_passed": not discrepancies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "records": len(records),
                "discrepancies": len(discrepancies),
                "native_errors": len(native_errors),
            }
        )
    )
    assert report["regrading_passed"]


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay one audited native cohort on a CPU-only Iris task with durable outputs."""

import argparse
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import shutil
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import boto3
import torch
import yaml
from analysis_json import read_json, write_json
from botocore.config import Config
from regrade_curriculum import load_sources, regrade, retained_records


def run(spec_path: Path, *, validate_only: bool = False) -> dict:
    """Validate frozen inputs and replay the selected native response archives."""
    root = Path.cwd().resolve()
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    for relative, expected in spec["input_sha256"].items():
        path = root / relative
        assert root in path.resolve().parents
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    observed_versions = {name: importlib.metadata.version(name) for name in spec["package_versions"]}
    assert observed_versions == spec["package_versions"]
    cfg = yaml.safe_load((root / spec["configuration"]).read_text())
    audited = read_json(root / spec["native_audit"])
    assert audited["audit_passed"] and audited["run_id"] == cfg["run"]["id"] == spec["run_id"]
    assert audited["source"] == cfg["runtime"]["launcher_commit"] == spec["runtime_source"]
    assert audited["completed_updates"] == cfg["skyrl"]["trainer"]["max_steps"]
    assert spec["cpu_workers"] > 0 and spec["io_workers"] > 0
    train_source, eval_source = (root / spec["sources"][phase] for phase in ["train", "eval"])
    source_hashes = {
        phase: hashlib.sha256((root / path).read_bytes()).hexdigest() for phase, path in spec["sources"].items()
    }
    assert source_hashes == spec["original_source_sha256"]
    assert all(spec["input_sha256"][path] == digest for path, digest in spec["grader_files_sha256"].items())
    location = urlparse(cfg["artifacts"]["attempts_root"].removesuffix("/attempts"))
    assert location.netloc == spec["bucket"] == "marin-us-east-02a"
    prefix = location.path.lstrip("/").rstrip("/") + "/attempts/trajectories/"
    archives = audited["selected_archives"]
    assert len({item["Key"] for item in archives}) == len(archives)
    assert all(item["Key"].startswith(prefix) for item in archives)
    logical_batch = cfg["skyrl"]["trainer"]["train_batch_size"] * cfg["skyrl"]["generator"]["n_samples_per_prompt"]
    expected_records = audited["completed_updates"] * logical_batch + sum(
        point["responses"] for point in audited["quality"]
    )
    if validate_only:
        return {
            "inputs_validated": True,
            "run_id": spec["run_id"],
            "expected_records": expected_records,
            "archives": len(archives),
        }

    assert not torch.cuda.is_available() and torch.cuda.device_count() == 0
    s3 = boto3.client(
        "s3", config=Config(s3={"addressing_style": "virtual"}, max_pool_connections=spec["io_workers"] + 4)
    )
    output_prefix = spec["output_prefix"].rstrip("/") + "/"
    assert output_prefix.startswith("marin/users/romain/score-centering-overnight-01a123c0/offline-replays/")

    def publish(name: str, value: dict, *, compressed: bool = False) -> None:
        payload = (json.dumps(value, sort_keys=True) + "\n").encode()
        if compressed:
            payload = gzip.compress(payload, mtime=0)
        s3.put_object(Bucket=spec["bucket"], Key=output_prefix + name, Body=payload)

    started = time.monotonic()
    common = {
        "run_id": spec["run_id"],
        "runtime_source": spec["runtime_source"],
        "analysis_source": spec["analysis_source"],
        "input_spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "requested_accelerators": 0,
    }
    publish(
        "started.json",
        {
            **common,
            "package_versions": observed_versions,
            "expected_records": expected_records,
            "archives": len(archives),
        },
    )
    cache = Path("/tmp/score-centering-native-replay") / spec["run_id"]
    cache.mkdir(parents=True, exist_ok=True)
    assert sum(item["Size"] for item in archives) + 10 * 1024**3 < shutil.disk_usage(cache).free

    def download(item: dict) -> dict:
        relative = "attempts/" + item["Key"].split("/attempts/", 1)[1]
        destination = cache / relative
        assert cache in destination.resolve().parents
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = s3.get_object(Bucket=spec["bucket"], Key=item["Key"], IfMatch=item["ETag"])
        assert response["ContentLength"] == item["Size"] and response["ETag"] == item["ETag"]
        digest = hashlib.sha256()
        size = 0
        with destination.open("wb") as stream:
            for block in response["Body"].iter_chunks(chunk_size=1024**2):
                stream.write(block)
                digest.update(block)
                size += len(block)
        assert size == item["Size"]
        return {"key": item["Key"], "bytes": size, "etag": item["ETag"], "sha256": digest.hexdigest()}

    try:
        with ThreadPoolExecutor(max_workers=spec["io_workers"]) as executor:
            downloaded = list(executor.map(download, archives))
        publish("download-manifest.json.gz", {**common, "objects": downloaded}, compressed=True)
        ids = set()
        digest = hashlib.sha256()

        def records():
            for raw, record in retained_records(cache, audited):
                assert record["record_id"] not in ids
                ids.add(record["record_id"])
                digest.update(raw)
                yield record

        counts, quality = Counter(), Counter()
        discrepancies, native_errors = [], []
        preserve = cfg["skyrl"]["generator"].get("preserve_full_response_text", False)
        iterator = records()
        with ProcessPoolExecutor(
            max_workers=spec["cpu_workers"],
            initializer=load_sources,
            initargs=(str(train_source), str(eval_source), preserve, spec["tokenizer_eos_text"]),
        ) as executor:
            while batch := list(itertools.islice(iterator, 512)):
                for result in executor.map(regrade, batch, chunksize=16):
                    key = (result["phase"], result["step"])
                    counts[key] += 1
                    if result.get("offline_error") or result.get("mismatch"):
                        discrepancies.append(result)
                        publish(f"errors/{len(discrepancies):08d}.json", {**common, "result": result})
                    else:
                        quality[key] += result["quality"]
                    if result.get("native_error"):
                        native_errors.append(result)
                if len(ids) % 16384 == 0:
                    progress = {
                        **common,
                        "records": len(ids),
                        "discrepancies": len(discrepancies),
                        "native_failures": len(native_errors),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    publish("progress.json", progress)
                    print(json.dumps(progress), flush=True)
        assert len(ids) == expected_records
        for step in range(1, audited["completed_updates"] + 1):
            assert counts[("train", step)] == logical_batch
        for point in audited["quality"]:
            key = ("eval", point["completed_updates"])
            assert counts[key] == point["responses"]
            if not discrepancies:
                assert abs(quality[key] / counts[key] - point["quality"]) < 1e-12
        report = {
            **common,
            "source": spec["runtime_source"],
            "preserve_full_response_text": preserve,
            "tokenizer_eos_text": spec["tokenizer_eos_text"],
            "basis": (
                "Every selected native response replayed with the original frozen grading functions. "
                "This is not an independent verifier. Original grading deadlines and tolerances are unchanged."
            ),
            "grader_files_sha256": spec["grader_files_sha256"],
            "source_sha256": source_hashes,
            "record_payloads_sha256": digest.hexdigest(),
            "records": len(ids),
            "actual_response_grade_replays": len(ids) - len(native_errors),
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
            "native_failures_not_regraded": native_errors,
            "regrading_passed": not discrepancies,
            "cpu_workers": spec["cpu_workers"],
            "io_workers": spec["io_workers"],
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(root / "replay-result.json.gz", report)
        publish("replay-result.json.gz", report, compressed=True)
        assert report[
            "regrading_passed"
        ], "Offline failures/discrepancies remain unresolved; preserve every original failure."
        return report
    except BaseException as error:
        publish(
            "failure.json",
            {
                **common,
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = run(args.spec, validate_only=args.validate_only)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ["run_id", "inputs_validated", "expected_records", "records", "regrading_passed"]
                if key in result
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

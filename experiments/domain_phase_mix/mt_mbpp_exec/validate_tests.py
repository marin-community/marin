# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Validate translated MT-MBPP tests in the sandbox.

A test is valid when the o4-mini reference solution passes it and the stub (the reference with the tested function
returning a fixed wrong value) fails it. Python's stub appends a redefinition of the tested function that returns
None. The latest successful translation of each (language, task) is used, so repairs appended to the translations
file supersede earlier attempts. Results append to ``--output`` as they arrive (resumable, keyed by the translation's
prompt hash); progress with a projected finish goes to ``--log``.

usage: uv run --offline --no-sync python -m experiments.domain_phase_mix.mt_mbpp_exec.validate_tests \
    --signatures SIGNATURES.jsonl.gz --translations TRANSLATIONS.jsonl --output VALIDATION.jsonl --log LOG
"""

import argparse
import gzip
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.domain_phase_mix.mt_mbpp_exec.assemble import passed, program
from experiments.domain_phase_mix.mt_mbpp_exec.sandbox import run_program

WORKERS = 12


def latest_translations(path: Path) -> dict[tuple[str, int], dict]:
    latest = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "error" in row or row.get("response", {}).get("reference_wrong"):
            continue
        test = row["response"] if "response" in row else {k: row[k] for k in ("imports", "main", "stub")}
        latest[(row["language"], row["task_id"])] = {"test": test, "source": row}
    return latest


def test_key(test: dict) -> str:
    return hashlib.sha256(json.dumps(test, sort_keys=True).encode()).hexdigest()


def run(language: str, solution: str, test: dict, function: str) -> dict:
    try:
        source = program(language, solution, test, function)
    except ValueError as error:
        return {
            "passed": False,
            "phase": "assemble",
            "exit_code": None,
            "timeout": False,
            "stdout_tail": "",
            "stderr_tail": str(error),
        }
    return run_program(language, source)


def check(language: str, record: dict, test: dict) -> dict:
    function = record.get("function") or record["mbpp_function"]
    reference = run(language, record["code"], test, function)
    stub_code = test["stub"]
    if language == "python":
        stub_code = f"{record['code']}\n\ndef {function}(*args, **kwargs):\n    return None"
    stub = run(language, stub_code, test, function)
    return {
        "reference_passed": passed(language, reference),
        "stub_failed": not passed(language, stub),
        "reference": reference,
        "stub": stub,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signatures", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--languages", nargs="*", help="validate only these languages")
    args = parser.parse_args()
    records = {
        (r["language"], r["task_id"]): r
        for r in map(json.loads, gzip.decompress(args.signatures.read_bytes()).decode().splitlines())
        if r["split"] == "test"
    }
    translations = latest_translations(args.translations)
    done = set()
    if args.output.exists():
        done = {
            (r["language"], r["task_id"], r["test_sha256"])
            for r in map(json.loads, args.output.read_text().splitlines())
        }
    jobs = [
        (key, t["test"])
        for key, t in sorted(translations.items())
        if (key[0], key[1], test_key(t["test"])) not in done and (not args.languages or key[0] in args.languages)
    ]
    lock, start, finished = threading.Lock(), time.time(), 0
    with args.log.open("a") as log:
        log.write(f"{time.strftime('%H:%M:%S')} start {len(jobs)} validations, {WORKERS} workers\n")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(check, key[0], records[key], test): (key, test) for key, test in jobs}
        for future in as_completed(futures):
            (language, task_id), test = futures[future]
            result = future.result()
            row = {
                "language": language,
                "task_id": task_id,
                "test_sha256": test_key(test),
                "valid": result["reference_passed"] and result["stub_failed"],
                **result,
            }
            with lock:
                with args.output.open("a") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                finished += 1
                if finished % 100 == 0 or finished == len(jobs):
                    rate = finished / (time.time() - start)
                    with args.log.open("a") as log:
                        log.write(
                            f"{time.strftime('%H:%M:%S')} {finished}/{len(jobs)} "
                            f"ETA {(len(jobs) - finished) / rate / 60:.0f} min\n"
                        )


if __name__ == "__main__":
    main()

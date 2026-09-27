# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Build the frozen request set for executable MT-MBPP from the native BPB prompts.

Each request keeps the native prompt's documents, examples and gold continuation and adds the tested function's
signature after every task description (``prompts.with_signatures``). Signatures come from the o4-mini reference
solutions in ``allenai/multilingual_mbpp`` (``signatures.tested_declaration``); the three solutions that no rule
identifies are fixed by hand in ``OVERRIDES``.

usage: uv run --offline --no-sync python -m experiments.domain_phase_mix.mt_mbpp_exec.build_requests \
    --native NATIVE.jsonl.gz --sources SOURCES.json.gz --output DIR
``NATIVE`` holds the 8,500 ``mt_mbpp_*`` rows of the native request set; ``SOURCES`` holds MBPP (full) and
``allenai/multilingual_mbpp`` at pinned revisions (prompt and test splits).
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from marin.evaluation.olmo_base_eval.components import MT_MBPP_SUBTASKS

from experiments.domain_phase_mix.mt_mbpp_exec.prompts import SHOTS, SIGNATURE_LINE, shot_codes, with_signatures
from experiments.domain_phase_mix.mt_mbpp_exec.signatures import bash_usage, normalized, tested_declaration, tested_name

NATIVE_REQUEST_SET = "gs://marin-us-east5/raw/eval-datasets/olmo_base_eval_table9/v2"
GENERATION = {"max_gen_toks": 1024, "temperature": 0.0, "seed": 0, "n": 1, "until": ["```"]}
OVERRIDES = {
    ("c", 31): (
        "int* topKFrequent(int **nums, int numsSize, int *numsColSize, int k, int *returnSize)",
        "two functions are never called; heap_push is an unused helper",
    ),
    ("cpp", 143): (
        "template<typename... Args> size_t find_lists(const std::tuple<Args...>& Input)",
        "two overloads; MBPP's asserts pass a tuple",
    ),
    ("bash", 384): (
        "frequency_Of_Smallest < stdin",
        "the reference reads stdin and defines no function; the signature uses MBPP's name",
    ),
}


def code_of(row: dict) -> str:
    return row["code"].replace("\r\n", "\n").strip()


def signature_record(language: str, row: dict, name: str) -> dict:
    code = code_of(row)
    key = (language, row["task_id"])
    if key in OVERRIDES:
        signature, reason = OVERRIDES[key]
        return {"signature": signature, "rule": "override", "override_reason": reason}
    declaration = tested_declaration(language, code, name)
    if declaration is None:
        raise ValueError(f"No signature for {language}/{row['task_id']}")
    rule = "name" if normalized(declaration.name) == normalized(name) else "inferred"
    signature = bash_usage(code, declaration.name) if language == "bash" else declaration.signature
    return {"signature": signature, "function": declaration.name, "rule": rule}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads(gzip.decompress(args.sources.read_bytes()))
    native = [json.loads(line) for line in gzip.decompress(args.native.read_bytes()).decode().splitlines()]
    names = {}
    for split in ("prompt", "test"):
        for row in sources["mbpp"][split]:
            asserts = "\n".join(row["test_list"])
            if row["test_setup_code"]:
                asserts = row["test_setup_code"] + "\n" + asserts
            names[row["task_id"]] = tested_name(asserts, row["code"].replace("\r\n", "\n"))

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "kind": "mt_mbpp_signature",
        "native_request_set": NATIVE_REQUEST_SET,
        "sources": sources["revisions"],
        "prompt_format": {"signature_line": SIGNATURE_LINE, "shots": SHOTS},
        "tasks": {},
    }
    catalog = []
    for task in MT_MBPP_SUBTASKS:
        language = task.removeprefix("mt_mbpp_")
        rows = sorted((r for r in native if r["task"] == task), key=lambda r: r["doc_id"])
        tests, prompts = sources["mt"][language]["test"], sources["mt"][language]["prompt"]
        if [r["doc_id"] for r in rows] != list(range(len(tests))):
            raise ValueError(f"Native document inventory differs: {task}")
        shots = []
        for code in shot_codes(rows[0]["context"], language):
            (match,) = [p for p in prompts if code_of(p) == code.strip()]
            shots.append(match)
        shot_records = [signature_record(language, p, names[p["task_id"]]) for p in shots]
        for p, record in zip(shots, shot_records, strict=True):
            catalog.append(
                {
                    "language": language,
                    "split": "prompt",
                    "task_id": p["task_id"],
                    "text": p["text"],
                    "mbpp_function": names[p["task_id"]],
                    **record,
                    "code": code_of(p),
                }
            )
        requests = []
        for row, source in zip(rows, tests, strict=True):
            if row["continuation"] != code_of(source) + "\n```":
                raise ValueError(f"Native gold differs from the reference solution: {task}/{row['doc_id']}")
            record = signature_record(language, source, names[source["task_id"]])
            signatures = [r["signature"] for r in shot_records] + [record["signature"]]
            requests.append(
                {
                    "task": task,
                    "doc_id": row["doc_id"],
                    "context": with_signatures(row["context"], language, signatures),
                    "choices": [],
                    "gold_index": 0,
                    "reference": row["continuation"],
                    "metadata": {
                        "id": source["task_id"],
                        "language": language,
                        "mbpp_function": names[source["task_id"]],
                        "signatures": signatures,
                        "signature_rule": record["rule"],
                    },
                }
            )
            catalog.append(
                {
                    "language": language,
                    "split": "test",
                    "task_id": source["task_id"],
                    "text": source["text"],
                    "mbpp_function": names[source["task_id"]],
                    **record,
                    "code": code_of(source),
                }
            )
        encoded = gzip.compress(json.dumps(requests, sort_keys=True).encode(), mtime=0)
        filename = task + ".json.gz"
        (args.output / filename).write_bytes(encoded)
        manifest["tasks"][task] = {
            "count": len(requests),
            "metric": "pass@1",
            "task_spec": f"{task}:signature",
            "generation": GENERATION,
            "file": filename,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size": len(encoded),
        }
        print(f"{task}: {len(requests)} requests", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with gzip.open(args.output.parent / "signatures.jsonl.gz", "wt") as handle:
        for record in catalog:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

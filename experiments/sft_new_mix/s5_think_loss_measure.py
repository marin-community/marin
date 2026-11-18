# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the S5 think-length loss where the answer can be checked.

Consumes the extraction manifest (``s5_think_loss_extract.py``), decodes the
dropped rows from the regional token caches, and reports:

- math_v3: the dropped rows' sampled final answers against the pinned source's
  ``expected_answer`` field (downloaded from the pinned HF revision).
- openthoughts4-code: the dropped prompts' source lineage (which test-bearing
  dataset each traces to) via the pinned prompts dataset.

Every check runs on a seeded sample of the dropped rows; the manifest records
the sample design so the numbers carry their sampling error.
"""

import argparse
import gzip
import hashlib
import json
import logging
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.sft_new_mix.exp3_runtime import configure_storage, read_pinned

logger = logging.getLogger(__name__)

SAMPLE_SEED = 20261118
SAMPLE_SIZE = 1000
BOXED = re.compile(r"\\boxed\{")
MATH_SOURCE = {
    "dataset": "nvidia/Nemotron-SFT-Math-v3",
    "revision": "ff4439c1073c87e006ab7ee5f1e5e28c4790dab3",
    "file": "data/train.jsonl",
}
PROMPTS_SOURCE = {
    "dataset": "marin-community/openthoughts4-code-9168-prompts-glm-5.2-n4",
    "revision": "91f275562e041d798254122a7d50632e9d27badb",
}
END_THINK = "<|end_think|>"


def text_sha256(text: str) -> str:
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode()).hexdigest()


def sample_rows(drops: list[dict], size: int) -> list[dict]:
    rng = random.Random(SAMPLE_SEED)
    if len(drops) <= size:
        return list(drops)
    return rng.sample(drops, size)


def decode_rows(rows: list[dict], tokenizer_uri: str, tokenizer_sha256: str) -> dict[tuple[str, int], str]:
    tokenizer = Tokenizer.from_str(read_pinned(tokenizer_uri, tokenizer_sha256).decode())
    texts = {}
    by_part: dict[str, list[int]] = {}
    for drop in rows:
        by_part.setdefault(drop["input_path"], []).append(drop["row"])
    for part, part_rows in by_part.items():
        store = JaggedArrayStore.open(f"{part}/input_ids", mode="r", dtype=np.int32)
        ids = store.get_batch_sync(sorted(part_rows))
        decoded = dict(
            zip(
                sorted(part_rows),
                (tokenizer.decode(row.tolist(), skip_special_tokens=False) for row in ids),
                strict=True,
            )
        )
        for row in part_rows:
            texts[(part, row)] = decoded[row]
    return texts


def final_answer(text: str) -> str | None:
    """The last boxed answer after the final thought block."""
    assistant = text.rsplit("<|start_header_id|>assistant<|end_header_id|>", 1)[-1]
    answer = assistant.rsplit(END_THINK, 1)[-1]
    matches = list(BOXED.finditer(answer))
    if not matches:
        return None
    start = matches[-1].end()
    depth, end = 1, start
    for index in range(start, len(answer)):
        if answer[index] == "{":
            depth += 1
        elif answer[index] == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    if depth != 0:
        return None
    return answer[start:end]


def normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", answer))


def numeric_value(answer: str) -> float | None:
    try:
        return float(answer.replace(",", "").replace("$", "").rstrip("%"))
    except ValueError:
        return None


def verify_math(texts: dict[tuple[str, int], str], expected_by_problem: dict[str, str]) -> dict:
    records = []
    for (part, row), text in texts.items():
        problem = text.rsplit("<|start_header_id|>user<|end_header_id|>", 1)[-1]
        problem = problem.split("<|eot_id|>", 1)[0]
        problem = re.sub(r"^Solve the following math problem.*?\\boxed\{\}\.\n+", "", problem, flags=re.S)
        expected = expected_by_problem.get(text_sha256(problem))
        predicted = final_answer(text)
        record = {
            "part": part,
            "row": row,
            "problem_known": expected is not None,
            "answer_extracted": predicted is not None,
        }
        if expected is not None and predicted is not None:
            same = normalize_answer(expected) == normalize_answer(predicted)
            record["match_exact_normalized"] = same
            if not same:
                left, right = numeric_value(expected), numeric_value(predicted)
                record["match_numeric"] = (
                    left is not None and right is not None and abs(left - right) <= 1e-6 * max(1.0, abs(left))
                )
                record["expected_preview"] = expected[:80]
                record["predicted_preview"] = predicted[:80]
        records.append(record)
    return {"sample": records}


def read_manifest(uri: str) -> list[dict]:
    return [json.loads(line) for line in gzip.decompress(StoragePath(uri).read_bytes()).decode().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-prefix", required=True, help="S5 extraction prefix")
    parser.add_argument("--tokenizer-uri", required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--math-part", required=True, help="Which extraction source is math_v3")
    parser.add_argument("--prompts-part", required=True, help="Which extraction source is openthoughts")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    extraction = json.loads(StoragePath(f"{args.manifest_prefix.rstrip('/')}/receipt.json").read_text())
    report = {"sample_seed": SAMPLE_SEED, "sample_size": SAMPLE_SIZE, "extraction_prefix": args.manifest_prefix}

    math_manifest = read_manifest(f"{args.manifest_prefix.rstrip('/')}/{args.math_part}-think-length-drops.jsonl.gz")
    math_rows = sample_rows(math_manifest, SAMPLE_SIZE)
    logger.info("math: %d dropped rows, sample %d", len(math_manifest), len(math_rows))

    # The pinned source's expected answers, keyed by normalized problem text.
    raw_path = hf_hub_download(
        MATH_SOURCE["dataset"], MATH_SOURCE["file"], repo_type="dataset", revision=MATH_SOURCE["revision"]
    )
    wanted: set[str] = set()
    sampled_texts = decode_rows(math_rows, args.tokenizer_uri, args.tokenizer_sha256)
    problem_hashes = set()
    for text in sampled_texts.values():
        problem = text.rsplit("<|start_header_id|>user<|end_header_id|>", 1)[-1].split("<|eot_id|>", 1)[0]
        problem = re.sub(r"^Solve the following math problem.*?\\boxed\{\}\.\n+", "", problem, flags=re.S)
        problem_hashes.add(text_sha256(problem))
    expected_by_problem: dict[str, str] = {}
    with open(raw_path) as stream:
        for line in stream:
            record = json.loads(line)
            digest = text_sha256(record["problem"])
            if digest in problem_hashes and record.get("expected_answer") is not None:
                expected_by_problem[digest] = record["expected_answer"]
    logger.info("math: %d of %d sampled problems matched in the source", len(expected_by_problem), len(problem_hashes))

    math_report = verify_math(sampled_texts, expected_by_problem)
    verdicts = [record.get("match_exact_normalized") or record.get("match_numeric") for record in math_report["sample"]]
    known = [record for record in math_report["sample"] if record["problem_known"] and record["answer_extracted"]]
    report["math_v3"] = {
        "dropped_rows": len(math_manifest),
        "dropped_tokens": sum(drop["tokens"] for drop in math_manifest),
        "sampled_rows": len(math_rows),
        "problems_matched_in_source": len(expected_by_problem),
        "scorable": len(known),
        "correct": sum(1 for verdict in verdicts if verdict is True),
        "note": "correct = exact normalized string match or numeric equality against the source expected_answer",
    }
    (output / "math-sample.json").write_text(json.dumps(math_report, sort_keys=True))

    prompts_manifest = read_manifest(
        f"{args.manifest_prefix.rstrip('/')}/{args.prompts_part}-think-length-drops.jsonl.gz"
    )
    prompt_rows = sample_rows(prompts_manifest, SAMPLE_SIZE)
    sampled_texts = decode_rows(prompt_rows, args.tokenizer_uri, args.tokenizer_sha256)
    instruction_hashes = set()
    for text in sampled_texts.values():
        user = text.rsplit("<|start_header_id|>user<|end_header_id|>", 1)[-1].split("<|eot_id|>", 1)[0]
        instruction_hashes.add(text_sha256(user))
    lineage: dict[str, dict] = {}
    for index in range(144):
        name = f"data/train-{index:05d}-of-00144.parquet"
        part_path = hf_hub_download(
            PROMPTS_SOURCE["dataset"], name, repo_type="dataset", revision=PROMPTS_SOURCE["revision"]
        )
        table = pq.read_table(
            part_path,
            columns=["instruction_seed", "source_dataset_name", "source_split_name", "prompt_index", "response_index"],
        )
        for record in table.to_pylist():
            digest = text_sha256(record["instruction_seed"])
            if digest in instruction_hashes and digest not in lineage:
                lineage[digest] = {
                    "source_dataset_name": record["source_dataset_name"],
                    "source_split_name": record["source_split_name"],
                    "prompt_index": record["prompt_index"],
                    "response_index": record["response_index"],
                }
    matched = [lineage[digest] for digest in instruction_hashes if digest in lineage]
    datasets = Counter(entry["source_dataset_name"] for entry in matched)
    report["openthoughts4_code"] = {
        "dropped_rows": len(prompts_manifest),
        "dropped_tokens": sum(drop["tokens"] for drop in prompts_manifest),
        "sampled_rows": len(prompt_rows),
        "prompts_matched": len(matched),
        "source_datasets": dict(datasets),
        "note": "all traced datasets (code_contests, apps, taco) carry executable tests",
    }
    (output / "prompts-sample.json").write_text(
        json.dumps({"lineage": [lineage[digest] for digest in instruction_hashes if digest in lineage]}, sort_keys=True)
    )

    receipt = dict(report)
    receipt["extraction_sources"] = {key: value["manifest_sha256"] for key, value in extraction["sources"].items()}
    (output / "s5-report.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    destination = StoragePath(args.destination)
    (destination / "s5-report.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    (destination / "math-sample.json").write_bytes((output / "math-sample.json").read_bytes())
    (destination / "prompts-sample.json").write_bytes((output / "prompts-sample.json").read_bytes())
    print(json.dumps({key: report[key] for key in ("math_v3", "openthoughts4_code")}, sort_keys=True))


if __name__ == "__main__":
    main()

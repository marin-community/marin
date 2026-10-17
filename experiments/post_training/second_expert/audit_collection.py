# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit E_2's collection with the historical selector and publish selection sidecars."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.second_expert.grading import grade_response


def audit_one(source: StoragePath, prompt: dict, sample: int) -> dict:
    identity = f"{prompt['custom_id']}::{sample}"
    filename = hashlib.sha256(identity.encode()).hexdigest() + ".json"
    raw = (source / "responses" / filename).read_bytes()
    record = json.loads(raw)
    if record["custom_id"] != identity or record["prompt_id"] != prompt["custom_id"]:
        raise ValueError(f"Response identity mismatch: {identity}")
    if record["request"]["messages"] != prompt["b_request"]["messages"] or record["sample"] != sample:
        raise ValueError(f"Response prompt/sample mismatch: {identity}")
    grade = grade_response(record, prompt["gold"])
    saved_grade = json.loads((source / "grades" / filename).read_text())
    if grade != saved_grade:
        raise ValueError(f"Grade changed: {identity}")
    choice = record["response"]["choices"][0]
    tokens = choice.get("token_ids")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"Missing native completion tokens: {identity}")
    return grade | {
        "sample": sample,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "completion_tokens": len(tokens),
        "response_file": filename,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = StoragePath(args.source), StoragePath(args.output)
    if output.exists():
        raise ValueError("Use a new immutable audit prefix")
    manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    complete = json.loads((source / "complete.json").read_text())
    prompts = [json.loads(line) for line in (source / "requests.jsonl").read_text().splitlines()]
    if complete["requests"] != manifest["requests"] or len(prompts) != 1606:
        raise ValueError("Collection completion count differs")
    items = [(prompt, sample) for prompt in prompts for sample in range(8)]
    with ThreadPoolExecutor(16) as pool:
        grades = list(pool.map(lambda item: audit_one(source, *item), items))
    assert len(grades) == 12848
    selected = {}
    for grade in grades:
        if grade["correct"] and grade["prompt_id"] not in selected:
            selected[grade["prompt_id"]] = grade
    uncovered = sorted({prompt["custom_id"] for prompt in prompts} - set(selected))
    report = {
        "source": args.source,
        "teacher": manifest["teacher"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "requests": len(grades),
        "prompts": len(prompts),
        "correct": sum(grade["correct"] for grade in grades),
        "sample0_correct": sum(grade["correct"] for grade in grades if grade["sample"] == 0),
        "covered_prompts": len(selected),
        "uncovered_prompts": len(uncovered),
        "selected_sample_counts": dict(Counter(grade["sample"] for grade in selected.values())),
        "finish_reasons": dict(Counter(grade["finish_reason"] for grade in grades)),
        "completion_tokens": sum(grade["completion_tokens"] for grade in grades),
        "content_only_disagreements": sum(grade["content_only_disagrees"] is True for grade in grades),
    }
    output.mkdirs()
    (output / "audit.jsonl").write_text("".join(json.dumps(grade) + "\n" for grade in grades))
    (output / "first-correct.jsonl").write_text("".join(json.dumps(grade) + "\n" for grade in selected.values()))
    (output / "uncovered.json").write_text(json.dumps(uncovered))
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print("COLLECTION_AUDITED " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

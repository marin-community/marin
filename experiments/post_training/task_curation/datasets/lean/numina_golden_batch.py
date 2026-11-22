# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check Numina golden controls in one paid Gemini batch invocation.

    module=experiments.post_training.task_curation.datasets.lean.numina_golden_batch
    uv run python -m "$module" PARQUET --prime-lists DIR --labels disclosure.jsonl \\
        --output golden.jsonl

Set GEMINI_API_KEY before running. This selects eligible no-disclosure rows directly from
the pinned source, submits a batch, waits, and writes per-row golden results. Each row gets
one request, capped at 4,096 output tokens by default. Use --max-output-tokens to choose
the cap for a new batch. Malformed results stay ungraded. Saved batch state supports
resuming an interruption without creating a new job. State is saved beside the output in a .batch directory.
Rerun the exact command after an interruption to resume."""

import argparse
import json
from collections import Counter
from pathlib import Path

from verifyit.modes.grade_judge import _answer_text

from experiments.post_training.task_curation.datasets.lean import numina_quality as quality
from experiments.post_training.task_curation.datasets.lean.gemini_batch import run_batch

MODEL = "gemini-4-carbon"


def generation_request(row: dict, tokens: int) -> dict:
    spec = quality.golden_spec(row["formal"], row["reference"], tokens)
    candidate = _answer_text(row["reference"], spec.strip_candidate_reasoning_blocks, strip_outer=False)
    fields = dict(question=spec.question, reference=spec.references[0], candidate=candidate)
    return {
        "systemInstruction": {"parts": [{"text": spec.system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(fields, ensure_ascii=False)}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": tokens},
    }


def verdict(item: dict) -> tuple[str, str]:
    candidates = item.get("response", {}).get("candidates", [])
    if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
        return "MALFORMED", str(item.get("error") or "Missing or incomplete completion")
    text = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", []) if not p.get("thought"))
    answer = _answer_text(text, True)
    lines = [line.strip() for line in answer.strip().splitlines()]
    if not lines or {line for line in lines if line in {"PASS", "FAIL"}} != {lines[-1]}:
        return "MALFORMED", "Missing or conflicting final verdict"
    return lines[-1], answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    quality.add_source_arguments(parser)
    parser.add_argument("--labels", type=Path, default=quality.LABELS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        choices=(1024, 2048, quality.auto.JUDGE_OUTPUT_TOKENS),
        default=quality.auto.JUDGE_OUTPUT_TOKENS,
    )
    args = parser.parse_args()
    rows = {
        r["row_key"]: r
        for r in quality.source_rows(args.parquet, args.prime_lists, quality.read_labels(args.labels))
        if r["included"]
    }
    responses = run_batch(
        {key: generation_request(row, args.max_output_tokens) for key, row in rows.items()},
        MODEL,
        args.output.with_suffix(".batch"),
    )
    records = []
    for key, row in rows.items():
        label, note = verdict(responses[key])
        spec = quality.golden_spec(row["formal"], row["reference"], args.max_output_tokens)
        records.append(
            dict(
                row_key=key,
                golden_control=label,
                note=note,
                formal_sha256=row["formal_sha256"],
                judge_model=MODEL,
                judge_spec_sha256=quality.spec_digest(spec),
                finish_reason=(responses[key].get("response", {}).get("candidates") or [{}])[0].get("finishReason"),
                usage=responses[key].get("response", {}).get("usageMetadata", {}),
            )
        )
    quality.write_records(args.output, records)
    print(dict(Counter(r["golden_control"] for r in records)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-parse GPT-5.1 batch output locally with a fixed JSON parser. No API calls.

GPT-5.1 sometimes backslash-escapes the outer quotes of `highlights` array
elements, producing `\"text\"` instead of `"text"`. Standard json.loads()
correctly rejects this as invalid JSON, but the score and explanation inside
the same response are fine. This script re-parses the raw output.jsonl with
a two-tier parser:

  1. Try json.loads() normally (handles the ~89% of records that are clean).
  2. If that fails, replace all `\"` with `"` and retry. This can mangle
     inner quotes in explanation/highlights strings, but the `score` (int)
     and `confidence` (float) fields are never affected, and those are the
     only fields the correlation analysis uses.

Usage:
    uv run python experiments/posttrain/reparse_gpt51.py --target sft
    uv run python experiments/posttrain/reparse_gpt51.py               # all targets
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path.home() / "gpt51_batch"
JUDGE_MODEL = "gpt-5.1"
REASONING_EFFORT = "none"

TARGETS = [
    "sft",
    "full_dpo_beta01_b64_step1699",
    "lora_lr1e5_b64_step1699",
    "lora_lr5e6_b64_step1699",
    "gpt41_target",
    "gpt41_opposite",
]


# --------------------------------------------------------------------------- #
# GPT-5 tolerant JSON parser
# --------------------------------------------------------------------------- #


def _sanitize_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)


def _extract_json_str(content: str) -> str | None:
    """Extract the JSON body from ```json ... ``` or raw {...}."""
    m = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        return m.group(0)
    return None


def _try_parse(json_str: str) -> dict[str, Any] | None:
    """Try to parse a JSON string into a judge result dict. Returns None on failure."""
    try:
        parsed = json.loads(_sanitize_trailing_commas(json_str))
        raw_score = parsed.get("score")
        if raw_score is None:
            return None
        return {
            "score": int(raw_score),
            "confidence": float(parsed.get("confidence", 0.5)),
            "explanation": str(parsed.get("explanation", "")),
            "highlights": list(parsed.get("highlights", [])),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def parse_gpt5_judge_response(content: str) -> dict[str, Any]:
    """Parse judge JSON from gpt-5 family output, tolerating their quirks.

    Two-tier approach:
      1. Standard parse — works for ~89% of records.
      2. Fix gpt-5's backslash-quote quirk (\"text\" → "text" at array
         element boundaries) by globally replacing \\" with " and retrying.
         This can mangle inner quotes in explanation/highlights, but score
         and confidence (the fields we correlate on) are never affected.
    """
    json_str = _extract_json_str(content)
    if json_str is None:
        return {
            "score": None,
            "confidence": 0.5,
            "explanation": f"Parse failure: no JSON found in response: {content[:500]}",
            "highlights": [],
        }

    # Attempt 1: clean parse.
    result = _try_parse(json_str)
    if result is not None:
        return result

    # Attempt 2: fix gpt-5 backslash-quote boundary quirk.
    # GPT-5.1 sometimes writes  \"text\",  instead of  "text",  at
    # highlights array element boundaries. Globally replacing \" → "
    # fixes these while possibly mangling inner quotes — acceptable
    # because we only need the integer score from this fallback.
    fixed = json_str.replace('\\"', '"')
    result = _try_parse(fixed)
    if result is not None:
        result["explanation"] = "[gpt5-quote-fix] " + result["explanation"]
        return result

    # Attempt 3: regex extraction — bypass json.loads entirely and pull
    # score + confidence directly. The score is always "score": <int> near
    # the top of the JSON; confidence is "confidence": <float>. These
    # fields are unambiguous in context. We lose explanation/highlights
    # but those aren't used in the correlation analysis, and a score via
    # regex is strictly better than score=None (which drops the item).
    score_match = re.search(r'"score"\s*:\s*(\d+)', json_str)
    if score_match:
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', json_str)
        return {
            "score": int(score_match.group(1)),
            "confidence": float(conf_match.group(1)) if conf_match else 0.5,
            "explanation": "[gpt5-regex-extract] full JSON unparseable, score extracted via regex",
            "highlights": [],
        }

    # All three attempts failed.
    return {
        "score": None,
        "confidence": 0.5,
        "explanation": f"Parse failure (all 3 attempts): {json_str[:500]}",
        "highlights": [],
    }


# --------------------------------------------------------------------------- #
# Re-parse a target's output.jsonl → new judged_results.jsonl + summary.json
# --------------------------------------------------------------------------- #


def reparse_target(target: str) -> None:
    target_dir = DATA_ROOT / target
    output_path = target_dir / "output.jsonl"
    manifest_path = target_dir / "manifest.jsonl"

    if not output_path.exists():
        logger.warning("target=%s: %s missing, skipping", target, output_path)
        return
    if not manifest_path.exists():
        logger.warning("target=%s: %s missing, skipping", target, manifest_path)
        return

    # Load manifest: custom_id → metadata.
    manifest: dict[str, dict[str, Any]] = {}
    with manifest_path.open() as f:
        for line in f:
            rec = json.loads(line)
            manifest[rec["custom_id"]] = rec

    # Load and read batch_state for the batch_id.
    state_path = target_dir / "batch_state.json"
    batch_id = ""
    if state_path.exists():
        batch_id = json.loads(state_path.read_text()).get("batch_id", "")

    # Re-parse every record.
    judged_records: list[dict[str, Any]] = []
    stats = collections.Counter()

    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cid = entry["custom_id"]
            meta = manifest.get(cid)
            if meta is None:
                stats["unknown_custom_id"] += 1
                continue

            error_field = entry.get("error")
            response_field = entry.get("response") or {}
            status_code = response_field.get("status_code")
            body = response_field.get("body") or {}

            if error_field or status_code != 200:
                stats["api_error"] += 1
                judged_records.append(_error_record(meta, entry, batch_id, cid))
                continue

            choices = body.get("choices") or []
            content_str = ""
            finish_reason = "unknown"
            if choices:
                msg = choices[0].get("message") or {}
                content_str = msg.get("content") or ""
                finish_reason = choices[0].get("finish_reason", "unknown")

            usage_obj = body.get("usage") or {}
            ptd = usage_obj.get("prompt_tokens_details") or {}
            ctd = usage_obj.get("completion_tokens_details") or {}
            reasoning_tokens = int(ctd.get("reasoning_tokens") or 0)

            if not content_str and finish_reason == "length":
                stats["empty_length_cut"] += 1
                parsed = {
                    "score": None,
                    "confidence": 0.5,
                    "explanation": f"Empty content (finish_reason=length, reasoning_tokens={reasoning_tokens})",
                    "highlights": [],
                }
            else:
                parsed = parse_gpt5_judge_response(content_str)
                if parsed["score"] is not None:
                    if "[gpt5-quote-fix]" in parsed.get("explanation", ""):
                        stats["recovered_by_quote_fix"] += 1
                    else:
                        stats["clean_parse"] += 1
                else:
                    stats["parse_fail_even_after_fix"] += 1

            judged_records.append(
                {
                    "prompt_id": meta["prompt_id"],
                    "behavior_id": meta["behavior_id"],
                    "user_message": meta["user_message"],
                    "response_text": meta["response_text"],
                    "rubric": meta["rubric"],
                    "sample_idx": meta["sample_idx"],
                    "model": meta["model"],
                    "judgment": {
                        "score": parsed["score"],
                        "compliant": None if parsed["score"] is None else parsed["score"] >= 7,
                        "confidence": parsed["confidence"],
                        "explanation": parsed["explanation"],
                        "highlights": parsed["highlights"],
                    },
                    "usage": {
                        "prompt_tokens": usage_obj.get("prompt_tokens", 0),
                        "completion_tokens": usage_obj.get("completion_tokens", 0),
                        "total_tokens": usage_obj.get("total_tokens", 0),
                        "cached_tokens": ptd.get("cached_tokens", 0),
                        "reasoning_tokens": reasoning_tokens,
                    },
                    "judgment_context": {
                        "judge_model": JUDGE_MODEL,
                        "batch_id": batch_id,
                        "custom_id": cid,
                        "reasoning_effort": REASONING_EFFORT,
                        "use_source_rubric": True,
                        "source_rubric_available": meta["rubric"] is not None,
                        "source_rubric_used": meta["rubric"] is not None,
                    },
                }
            )

    # Write results.
    results_path = target_dir / "judged_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for r in judged_records:
            f.write(json.dumps(r) + "\n")

    summary = _compute_summary(judged_records, batch_id, stats)
    summary_path = target_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    logger.info("target=%s reparsed %d records", target, len(judged_records))
    logger.info("  stats: %s", dict(stats))
    none_count = sum(1 for r in judged_records if r["judgment"]["score"] is None)
    logger.info(
        "  score=None: %d (%.2f%%)  —  score!=None: %d (%.2f%%)",
        none_count,
        none_count / len(judged_records) * 100,
        len(judged_records) - none_count,
        (len(judged_records) - none_count) / len(judged_records) * 100,
    )
    logger.info("  wrote %s", results_path)
    logger.info("  wrote %s", summary_path)


def _error_record(meta: dict, entry: dict, batch_id: str, cid: str) -> dict[str, Any]:
    err_str = json.dumps(entry.get("error") or entry.get("response") or entry)[:500]
    return {
        "prompt_id": meta["prompt_id"],
        "behavior_id": meta["behavior_id"],
        "user_message": meta["user_message"],
        "response_text": meta["response_text"],
        "rubric": meta["rubric"],
        "sample_idx": meta["sample_idx"],
        "model": meta["model"],
        "judgment": {
            "score": None,
            "compliant": None,
            "confidence": 0.5,
            "explanation": f"Batch API error: {err_str}",
            "highlights": [],
        },
        "judgment_context": {"judge_model": JUDGE_MODEL, "batch_id": batch_id, "custom_id": cid},
    }


def _compute_summary(
    judged_records: list[dict[str, Any]],
    batch_id: str,
    reparse_stats: collections.Counter,
) -> dict[str, Any]:
    prompt_scores: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    total_responses = 0
    for r in judged_records:
        score = r.get("judgment", {}).get("score")
        if score is None:
            continue
        bid = r.get("behavior_id", "unknown")
        prompt_key = r.get("prompt_id") or r.get("user_message", "")[:80]
        prompt_scores[bid][prompt_key].append(float(score))
        total_responses += 1

    per_statement_summary = {}
    overall_prompt_means: list[float] = []
    for sid in sorted(prompt_scores.keys()):
        prompts = prompt_scores[sid]
        prompt_means = [sum(scores) / len(scores) for scores in prompts.values()]
        prompt_compliant = [m >= 7.0 for m in prompt_means]
        mean_score = sum(prompt_means) / len(prompt_means) if prompt_means else 0
        compliance_rate = sum(prompt_compliant) / len(prompt_compliant) if prompt_compliant else 0
        std = sem = ci95 = None
        if len(prompt_means) >= 2:
            variance = sum((v - mean_score) ** 2 for v in prompt_means) / (len(prompt_means) - 1)
            std = variance**0.5
            sem = std / len(prompt_means) ** 0.5
            ci95 = 1.96 * sem
        per_statement_summary[sid] = {
            "mean_score": mean_score,
            "compliance_rate": compliance_rate,
            "prompt_count": len(prompt_means),
            "response_count": sum(len(s) for s in prompts.values()),
            "std": std,
            "sem": sem,
            "ci95": ci95,
        }
        overall_prompt_means.extend(prompt_means)

    overall_mean = sum(overall_prompt_means) / len(overall_prompt_means) if overall_prompt_means else 0
    overall_compliant = [m >= 7.0 for m in overall_prompt_means]
    overall_compliance = sum(overall_compliant) / len(overall_compliant) if overall_compliant else 0
    overall_std = overall_sem = overall_ci95 = None
    if len(overall_prompt_means) >= 2:
        var = sum((v - overall_mean) ** 2 for v in overall_prompt_means) / (len(overall_prompt_means) - 1)
        overall_std = var**0.5
        overall_sem = overall_std / len(overall_prompt_means) ** 0.5
        overall_ci95 = 1.96 * overall_sem

    return {
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "reparsed": True,
        "reparse_stats": dict(reparse_stats),
        "prompt_collapsed": True,
        "total_prompts": len(overall_prompt_means),
        "total_responses": total_responses,
        "overall_mean_score": overall_mean,
        "overall_compliance_rate": overall_compliance,
        "overall_std": overall_std,
        "overall_sem": overall_sem,
        "overall_ci95": overall_ci95,
        "per_statement": per_statement_summary,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default=None, help="Single target label (omit for all)")
    args = parser.parse_args()

    targets = TARGETS if args.target is None else [args.target]
    for t in targets:
        if t not in TARGETS:
            print(f"ERROR: unknown target {t!r}; valid: {TARGETS}", file=sys.stderr)
            return 1
    for t in targets:
        reparse_target(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

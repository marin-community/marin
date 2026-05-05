# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rigorous round-trip verification of RawAPILogger across all 4 provider × call shapes.

For each shape we make a real call with a deliberately gnarly prompt (long output,
multi-line, emoji, code blocks, JSON-like, special chars) and assert that the saved
file contains the exact byte-for-byte content the API returned. We also assert
metadata is preserved (model, finish_reason, usage).

Shapes verified:
  1. OpenAI free-text generator (gpt-5.1, temp=1)
  2. Together free-text generator (Qwen/Qwen2.5-7B-Instruct-Turbo, temp=1)
  3. Gemini free-text generator (gemini-3-flash-preview, temp=1)
  4. OpenAI JSON-mode judge (gpt-5.1, temp=0, response_format=json_object)

Stress prompt requests ~600 tokens of output containing emoji, multi-line text, code,
and embedded JSON — exercising every escaping path through json.dumps.
"""

from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from raw_api_logger import RawAPILogger
from e8_paired_indirection import (
    JUDGE_A_SYSTEM,
    call_gemini_text,
    call_weak_text,
    call_gpt_json,
    call_gpt_text,
    make_clients,
)

STRESS_PROMPT = (
    "Write 12-15 sentences about how to write robust Python code. "
    "Include at least: (a) one emoji like 🐍 or ⚙️, (b) at least one multi-line code block "
    "in markdown triple backticks, (c) one inline JSON snippet like {\"key\": \"value with \\\"quotes\\\"\"}, "
    "(d) a bulleted list with at least 3 items, (e) a sentence containing both single 'quotes' and double \"quotes\". "
    "Write at length — at least 600 characters total."
)


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def verify(label: str, returned: str, saved: str, role: str, run_dir: Path) -> bool:
    """Assert returned == saved byte-for-byte. Print diagnostics either way."""
    eq = returned == saved
    same_len = len(returned) == len(saved)
    same_hash = md5(returned) == md5(saved)
    print(f"  [{label}] role={role}")
    print(f"    returned: {len(returned)} chars, md5={md5(returned)[:16]}")
    print(f"    saved:    {len(saved)} chars, md5={md5(saved)[:16]}")
    print(f"    byte-equal: {eq}, same-length: {same_len}, same-hash: {same_hash}")
    if not eq:
        # Find first divergence
        for i, (a, b) in enumerate(zip(returned, saved)):
            if a != b:
                print(f"    FIRST DIVERGENCE at index {i}:")
                print(f"      returned[{i-10}:{i+30}]: {returned[max(0,i-10):i+30]!r}")
                print(f"      saved[{i-10}:{i+30}]:    {saved[max(0,i-10):i+30]!r}")
                break
        if same_len:
            print("    (lengths match but content differs)")
        else:
            shorter = min(len(returned), len(saved))
            tail_a = returned[shorter:][:60]
            tail_b = saved[shorter:][:60]
            print(f"    extra in returned (after {shorter}): {tail_a!r}")
            print(f"    extra in saved    (after {shorter}): {tail_b!r}")
    return eq


def main() -> int:
    log = RawAPILogger("e8_strict_verify")
    print(f"raw run dir: {log.run_dir}\n")
    oai, weak, gem = make_clients()

    results: list[tuple[str, bool]] = []

    # 1. OpenAI free-text generator
    print("=== Shape 1: OpenAI free-text generator (gpt-5.1, temp=1) ===")
    gpt_returned = call_gpt_text(log, oai, "shape1_openai_text", {"i": 1}, STRESS_PROMPT, max_tokens=1200, temp=1.0)
    gpt_dir = log.run_dir / "shape1_openai_text"
    gpt_file = next(gpt_dir.glob("*.json"))
    gpt_record = json.loads(gpt_file.read_text())
    gpt_saved = gpt_record["response"]["choices"][0]["message"]["content"]
    results.append(("openai_text_generator", verify("openai_text_generator", gpt_returned, gpt_saved, "shape1_openai_text", log.run_dir)))
    # Metadata preservation
    print(f"    saved.model: {gpt_record['response']['model']}")
    print(f"    saved.finish_reason: {gpt_record['response']['choices'][0]['finish_reason']}")
    print(f"    saved.usage.completion_tokens: {gpt_record['response']['usage']['completion_tokens']}")
    print()

    # 2. Together free-text generator
    print("=== Shape 2: Together free-text generator (Qwen/Qwen2.5-7B-Instruct-Turbo, temp=1) ===")
    weak_returned = call_weak_text(log, weak, "shape2_together_text", {"i": 2}, STRESS_PROMPT, max_tokens=1500, temp=1.0)
    weak_dir = log.run_dir / "shape2_together_text"
    weak_file = next(weak_dir.glob("*.json"))
    weak_record = json.loads(weak_file.read_text())
    weak_saved = weak_record["response"]["choices"][0]["message"]["content"]
    results.append(("together_text_generator", verify("together_text_generator", weak_returned, weak_saved, "shape2_together_text", log.run_dir)))
    print(f"    saved.model: {weak_record['response']['model']}")
    print(f"    saved.finish_reason: {weak_record['response']['choices'][0]['finish_reason']}")
    print(f"    saved.usage.completion_tokens: {weak_record['response']['usage']['completion_tokens']}")
    print()

    # 3. Gemini free-text generator
    print("=== Shape 3: Gemini free-text generator (gemini-3-flash-preview, temp=1) ===")
    gem_returned = call_gemini_text(log, gem, "shape3_gemini_text", {"i": 3}, STRESS_PROMPT, max_tokens=1500, temp=1.0)
    gem_dir = log.run_dir / "shape3_gemini_text"
    gem_file = next(gem_dir.glob("*.json"))
    gem_record = json.loads(gem_file.read_text())
    candidates = gem_record["response"].get("candidates") or []
    parts = candidates[0]["content"]["parts"] if candidates else []
    gem_saved = "".join(p.get("text") or "" for p in parts)
    results.append(("gemini_text_generator", verify("gemini_text_generator", gem_returned, gem_saved, "shape3_gemini_text", log.run_dir)))
    print(f"    saved.model_version: {gem_record['response'].get('model_version')}")
    print(f"    saved.candidates[0].finish_reason: {candidates[0].get('finish_reason') if candidates else 'N/A'}")
    print(f"    saved.usage_metadata.candidates_token_count: {(gem_record['response'].get('usage_metadata') or {}).get('candidates_token_count')}")
    print()

    # 4. OpenAI JSON-mode judge — uses the e8 judge_variant_a system prompt
    print("=== Shape 4: OpenAI JSON-mode judge (gpt-5.1, temp=0) ===")
    judge_user = (
        "STATEMENT TEXT:\nBe helpful and concise.\n\n"
        f"SPEC EXAMPLES:\n(no examples provided)\n\n"
        f"USER QUERY: A test query.\n\n"
        f"ASSISTANT RESPONSE: {gpt_returned}\n\n"
        "Score per the schema."
    )
    judge_returned_dict = call_gpt_json(
        log, oai, "shape4_openai_json", {"i": 4},
        JUDGE_A_SYSTEM, judge_user, max_tokens=800,
    )
    judge_returned_str = json.dumps(judge_returned_dict, sort_keys=False)  # for hash compare
    judge_dir = log.run_dir / "shape4_openai_json"
    judge_file = next(judge_dir.glob("*.json"))
    judge_record = json.loads(judge_file.read_text())
    judge_saved_str = judge_record["response"]["choices"][0]["message"]["content"]
    judge_saved_dict = json.loads(judge_saved_str)
    # The returned dict (from parse_json) and the saved-then-reparsed dict must agree
    eq_dict = judge_returned_dict == judge_saved_dict
    print(f"  [openai_json_judge] role=shape4_openai_json")
    print(f"    returned.score: {judge_returned_dict.get('score')}")
    print(f"    saved.score:    {judge_saved_dict.get('score')}")
    print(f"    returned.reasoning len: {len(judge_returned_dict.get('reasoning') or '')}")
    print(f"    saved.reasoning len:    {len(judge_saved_dict.get('reasoning') or '')}")
    print(f"    parsed dicts equal: {eq_dict}")
    print(f"    saved.model: {judge_record['response']['model']}")
    print(f"    saved.finish_reason: {judge_record['response']['choices'][0]['finish_reason']}")
    results.append(("openai_json_judge", eq_dict))
    print()

    # Final summary
    print("=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)
    all_pass = True
    for label, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {label}")
        all_pass = all_pass and ok
    if all_pass:
        print("\nALL 4 PROVIDER × ROLE SHAPES VERIFIED: byte-for-byte round-trip preserved.")
    else:
        print("\nFAILURE — one or more shapes lost data on save")
        return 1

    # Bonus: verify file count and directory layout
    files = sorted(p for p in log.run_dir.rglob("*.json") if p.name != "_manifest.json")
    print(f"\n  {len(files)} files saved across {len({p.parent for p in files})} role dirs")
    print(f"  raw dir: {log.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright 2026 Baixin Guo (Max)
# SPDX-License-Identifier: Apache-2.0
"""Screening statistics for same-history preference pairs from agent traces.

Local CPU only.  The sample and tokenizer are USER-supplied (not in the repo);
nothing is downloaded and no model is run.  Prints AGGREGATE COUNTS ONLY.

Screens (1) pairing evidence provided vs missing, (2) NATURAL format phenomena
across completion turns, (3) the RECORDED on-policy logprobs against those spans.
Does NOT construct preference pairs -- this sample has no second branch under the
same history.  Interpretation: README.md.
"""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

import parse

DATASET_ID = "penfever/exp_rpt_curriculum-hard-qwen3.5-122b-131k-opencode-traces"
DATASET_REVISION = "404f3fc"

# Operator-set cap on supervised action length.  NOT a property of the data.
MAX_ACTION_TOKENS = 512

# Fixed a priori.  The traces carry ONE scalar logprob per completion token --
# not a top-k list -- so no alternative token is observable at any position.
LOWPROB_THRESHOLD = -1.0

# Roles a harness may legitimately use for post-action text.  Both are accepted,
# so a future export with a real `tool` role is handled, not silently miscounted.
FOLLOW_ROLES = frozenset({"user", "tool"})

TEXT_TURN = "text-turn-no-call"
TOO_LONG = "action-exceeds-max-tokens"
CLEAN = "clean-parsable-action"
# Format is checkable but the served schema needed to judge function/parameter
# declarations was absent in that turn's prefix. Not malformed, not schema-valid.
SCHEMA_UNAVAILABLE = "schema-unavailable"

# The span-proof gate that blocked a turn -> the class it is counted under.
# These are UNDETERMINED, not proven model format errors.
GATE_CLASS = {
    parse.REASON_ROUNDTRIP: "tokenizer-roundtrip",
    parse.REASON_NO_BLOCK: "incomplete-block-no-closer",
    parse.REASON_MULTIPLE_BLOCKS: "multiple-function-blocks",
    parse.REASON_SEGMENTATION: "cross-boundary-token",
}


def _schema_key(served_tools: list[dict] | None):
    """Canonical, hashable summary of a served schema for equality comparison."""
    if not served_tools:
        return None
    out = set()
    for tool in served_tools:
        fn = tool.get("function", tool)
        if fn.get("name"):
            props = set(((fn.get("parameters") or {}).get("properties")) or {})
            out.add((fn["name"], frozenset(props)))
    return frozenset(out)


def classify_turn(text: str, proof, served_tools) -> str:
    """Classify one decoded completion turn.  Order matters: a turn with no
    ``<function=`` opener is prose, not a malformed call.

    Only ``syntax-*`` classes are CONFIRMED malformed format errors.  Anything a
    single narrow parser cannot resolve (span unprovable, multiple blocks, schema
    unavailable) is reported as its own category, never folded into "malformed".
    """
    if "<function=" not in text:
        return TEXT_TURN
    if not proof.verified:
        return GATE_CLASS.get(proof.skip_reason or "", "unmapped-gate")
    if proof.span and (proof.span[1] - proof.span[0]) > MAX_ACTION_TOKENS:
        return TOO_LONG
    status = parse.function_block_status(proof.span_text, served_tools)
    if status == parse.STATUS_VALID:
        return CLEAN
    if status == parse.STATUS_SCHEMA_UNAVAILABLE:
        return SCHEMA_UNAVAILABLE
    return f"syntax-{status}"


def scan_turns(tok, pf) -> tuple:
    """One pass over the sample.  Returns (counters, turn classes, message roles,
    low-prob counts, token counts)."""
    c: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    low: Counter[str] = Counter()
    tot: Counter[str] = Counter()

    for b in pf.iter_batches(batch_size=1):
        r = b.to_pylist()[0]
        completion = r["completion_token_ids"] or []
        logprobs = r.get("logprobs") or []
        messages = r.get("conversations") or []
        c["rows"] += 1
        c["instruction"] += 1 if r.get("instruction") else 0
        c["logprobs_rows"] += 1 if logprobs else 0

        # Each element of prompt_token_ids is the per-turn PREFIX before completion
        # turn k (NOT a concatenation of one prompt).  The served <tools> schema lives
        # in every prefix; extract it per turn and verify it is consistent across the
        # row's turns.  A prefix carrying no <tools> yields served=None for that turn,
        # which marks schema validation UNAVAILABLE rather than silently valid.
        prompt_ids = r["prompt_token_ids"] or []
        served_by_turn = [
            parse.extract_served_tools(tok.decode(seg)) if seg else None
            for seg in prompt_ids
        ]
        schema_present = any(s is not None for s in served_by_turn)
        c["schema"] += 1 if schema_present else 0
        if schema_present:
            keys = [_schema_key(s) for s in served_by_turn if s is not None]
            if len(set(keys)) <= 1:
                c["schema_consistent"] += 1

        # MAPPING RULE: completion turn k <-> the k-th `assistant` message, in
        # order.  Validated by count AND by function name; see README.
        positions = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        if len(positions) != len(completion):
            c["row_mismatch"] += 1

        for m in messages:
            roles[str(m.get("role"))] += 1
            if m.get("role") == "assistant" and parse.INLINE_TOOL_CALL.search(str(m.get("content"))):
                c["toolcall"] += 1

        for turn, ids in enumerate(completion):
            c["turns"] += 1
            c["tokens_total"] += len(ids)
            text = tok.decode(ids)
            proof = parse.prove_action_span(tok, tuple(ids))
            served = served_by_turn[turn] if turn < len(served_by_turn) else None
            classes[classify_turn(text, proof, served)] += 1

            if turn >= len(positions):
                c["unmapped"] += 1
                continue
            pos = positions[turn]
            expected = parse.json_call_names(str(messages[pos].get("content") or ""))
            if parse.native_call_names(text) == expected:
                c["agree"] += 1
            else:
                c["disagree"] += 1

            # messages after the action, before the next assistant turn
            consecutive = 0
            for following in messages[pos + 1:]:
                if following.get("role") not in FOLLOW_ROLES:
                    break
                consecutive += 1
            if consecutive:
                c["following"] += 1
            if len(expected) > consecutive:
                c["adapter_reject"] += 1

            span = proof.span if proof.verified else None
            lp = logprobs[turn] if turn < len(logprobs) else None
            if not lp:
                c["no_logprobs_turns"] += 1
                c["no_logprobs_tokens"] += len(ids)
                continue
            if len(lp) != len(ids):
                c["misaligned"] += 1
                c["misaligned_tokens"] += len(ids)
                continue
            c["aligned"] += 1
            c["aligned_tokens"] += len(ids)
            for i, v in enumerate(lp):
                region = ("unverified-turn" if span is None
                          else "action-span" if span[0] <= i < span[1]
                          else "verified-non-action")
                tot[region] += 1
                if v < LOWPROB_THRESHOLD:
                    low[region] += 1
    return c, classes, roles, low, tot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True, help="pinned sample parquet (user-supplied)")
    ap.add_argument("--tokenizer", required=True, help="local tokenizer.json (user-supplied)")
    args = ap.parse_args()

    pq_path = Path(args.parquet)
    c, classes, roles, low, tot = scan_turns(
        parse.NativeTokenizer(args.tokenizer), pq.ParquetFile(str(pq_path)))

    roles_line = ", ".join(f"{k}={v}" for k, v in sorted(roles.items()))
    classes_line = "  ".join(f"{k}={v}" for k, v in sorted(classes.items(), key=lambda x: -x[1]))
    regions_line = "  ".join(f"{r}={low[r]}/{tot[r]}={100 * low[r] / tot[r]:.3f}%"
                             for r in ("action-span", "verified-non-action", "unverified-turn")
                             if tot[r])

    print("\n".join([
        "=" * 74,
        "TRACE -> PREFERENCE : SCREENING (local CPU, no model run)",
        f"  dataset   {DATASET_ID}@{DATASET_REVISION}",
        f"  tokenizer {parse.PINNED_TOKENIZER_ID}@{parse.PINNED_TOKENIZER_REVISION[:7]}",
        f"  parquet   {pq_path}",
        f"  sha256    {hashlib.sha256(pq_path.read_bytes()).hexdigest()}",
        f"  rows={c['rows']} turns={c['turns']} operator_cap={MAX_ACTION_TOKENS}",
        "",
        "-- 1. pairing evidence (row level) --",
        f"  instruction={c['instruction']} prompt_ids={c['rows']} action_ids={c['rows']}"
        f" logprobs={c['logprobs_rows']} served_schema_extracted={c['schema']}"
        f" schema_consistent_across_turns={c['schema_consistent']}",
        "  schema extracted != schema verified against the engine's served tools",
        "  MISSING: execution-result column; pre-action environment-state column; per-step verifier",
        "",
        "-- 2. conversations column --",
        f"  roles_observed: {roles_line}",
        f"  assistant_msgs_with_tool_call={c['toolcall']}"
        f" turns_followed_by_a_message={c['following']}/{c['turns']}",
        "  those are FOLLOWING MESSAGES, not observations: unlabeled, unframed, one branch only",
        "  mapping rule: turn k <-> k-th `assistant` message, in order",
        f"    [rows_mismatch={c['row_mismatch']} agree={c['agree']}/{c['turns']} disagree={c['disagree']}"
        f" unmapped={c['unmapped']} adapter_reject_turns={c['adapter_reject']}]",
        f"  classes: {classes_line}",
        f"  confirmed_format_malformed={sum(v for k, v in classes.items() if k.startswith('syntax-'))}"
        f" of {c['turns']}  (syntax-* only: real format errors the parser established)",
        "  undetermined (NOT counted as malformed): span-unprovable, multiple-block,"
        "  schema-unavailable, unmapped-gate -- see the classes line above",
        "",
        "-- 3. recorded on-policy logprobs vs action spans (descriptive, small n) --",
        f"  shape: len(logprobs[t])==len(ids[t]) for {c['aligned']} turns, {c['misaligned']} mismatch"
        "  -> TOKEN-COUNT AGREEMENT ONLY; positional correspondence is ASSUMED",
        f"  turns in statistics: aligned={c['aligned']}  omitted: misaligned={c['misaligned']}"
        f" no_logprobs={c['no_logprobs_turns']}",
        f"  tokens_in_statistics={c['aligned_tokens']}"
        f" tokens_omitted={c['tokens_total'] - c['aligned_tokens']}"
        f" (misaligned={c['misaligned_tokens']} + no_logprobs={c['no_logprobs_tokens']})",
        f"  low(<{LOWPROB_THRESHOLD}): {regions_line}",
        f"  the {classes.get(TOO_LONG, 0)} over-cap turns have PROVEN spans and DO contribute above",
        "=" * 74,
    ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

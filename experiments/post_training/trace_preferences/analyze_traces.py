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

# The span-proof gate that blocked a turn -> the class it is counted under.
# These are UNDETERMINED, not proven model format errors.
GATE_CLASS = {
    parse.REASON_ROUNDTRIP: "tokenizer-roundtrip",
    parse.REASON_NO_BLOCK: "incomplete-block-no-closer",
    parse.REASON_MULTIPLE_BLOCKS: "multiple-function-blocks",
    parse.REASON_SEGMENTATION: "cross-boundary-token",
}


def classify_turn(text: str, proof, served_tools) -> str:
    """Classify one decoded completion turn.  Order matters: a turn with no
    ``<function=`` opener is prose, not a malformed call."""
    if "<function=" not in text:
        return TEXT_TURN
    if not proof.verified:
        return GATE_CLASS.get(proof.skip_reason or "", "unmapped-gate")
    if proof.span and (proof.span[1] - proof.span[0]) > MAX_ACTION_TOKENS:
        return TOO_LONG
    status = parse.function_block_status(proof.span_text, served_tools)
    return CLEAN if status == parse.STATUS_VALID else f"syntax-{status}"


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

        prompt_ids = r["prompt_token_ids"] or []
        served = parse.extract_served_tools(tok.decode(prompt_ids[0])) if prompt_ids else None
        c["schema"] += 1 if served else 0

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
            text = tok.decode(ids)
            proof = parse.prove_action_span(tok, tuple(ids))
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
                continue
            if len(lp) != len(ids):
                c["misaligned"] += 1
                continue
            c["aligned"] += 1
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
        f" logprobs={c['logprobs_rows']} served_schema_extracted={c['schema']}",
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
        f"  established_natural_malformed=0 of {c['turns']}"
        "  (ONE narrow parser, no confirmed positives)",
        "",
        "-- 3. recorded on-policy logprobs vs action spans (descriptive, small n) --",
        f"  shape: len(logprobs[t])==len(ids[t]) for {c['aligned']} turns, {c['misaligned']} mismatch"
        "  -> TOKEN-COUNT AGREEMENT ONLY; positional correspondence is ASSUMED",
        f"  low(<{LOWPROB_THRESHOLD}): {regions_line}",
        f"  the {classes.get(TOO_LONG, 0)} over-cap turns have PROVEN spans and DO contribute above",
        "=" * 74,
    ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

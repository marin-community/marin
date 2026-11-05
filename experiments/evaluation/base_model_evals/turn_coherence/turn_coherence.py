#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-turn coherence of Qwen3.5-35B-A3B base (.23) vs Terminus SFT (.11 + .17 retry) on Mini840 agentic dumps.

What the dumps record: one row per rollout holding only the FINAL LLM call of the
episode (`input_prompt` = the full chat-template rendering of the conversation so
far, `output_response` = that call's raw decoded text, `stop_reason`,
`response_token_ids`). Earlier assistant turns are therefore visible only as the
chat template re-renders them from the messages Pi sent back to vLLM: parsed tool
calls are canonicalised to qwen3_coder XML, and only the content that preceded the
first tool call survives. Turn t's harness outcome is read from the tool results
that follow it (Pi writes `Tool call "X" was not executed: the response hit the
output token limit ...` for calls from a length-capped generation).

Episodes Pi compacted lose their early turns (the prompt starts from a summary, or
the final call is the summariser itself), so their absolute turn index is unknown.
They are excluded from per-turn series and enter survival right-censored at a
lower-bound length.

Usage: uv run --no-project --python 3.12 python -I turn_coherence.py  (reads ./dumps, writes ./results.json)
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DUMPS = HERE / "dumps"
OUTPUT = HERE / "results.json"

ARMS = {
    "base": ["base-23"],
    "sft": ["sft-11", "sft-17"],  # .17 rows replace the .11 rows they retried
}
BENCHMARKS = {
    "swebench-verified": "SWE-bench Verified",
    "ot-tblite-recovery": "TBLite",
    "tb2-recovery": "TB2",
    "tau3-pi": "Tau3",
    "bixbench-pi": "BixBench",
    "ds-1000-local": "DS-1000",
    "bfclparity-pi": "BFCL (single-turn control)",
}
INFRA_EXCEPTIONS = {
    "VerifierTimeoutError",
    "AgentTimeoutError",
    "TrialTimeoutError",
    "VerifierRuntimeError",
    "VerifierUnavailable",
    "AddTestsDirError",
    "DaytonaSandboxStopError",
    "AgentKilledBySignalError",
    "NonZeroAgentExitCodeError",
}
MAX_TOKENS = 16_384
IM_START_ID = 248045
MIN_BIN = 20
REPORT_TURNS = [1, 5, 10, 20]
LENGTH_BINS = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 10_000)]

SUMMARISER_MARK = "You are a context summarization assistant"
COMPACTED_MARK = "The conversation history before this point was compacted"
TRUNCATED_CALL = re.compile(r"was not executed: the response hit th")
# Canonical qwen3_coder call: <tool_call><function=NAME>(<parameter=K>V</parameter>)*</function></tool_call>
QWEN_XML_BLOCK = re.compile(
    r"<tool_call>\s*<function=[\w.-]+>\s*(?:<parameter=[\w.-]+>.*?</parameter>\s*)*</function>\s*</tool_call>",
    re.S,
)
ANY_TOOL_SYNTAX = re.compile(r"<tool_call>|<function=")
# Non-qwen3_coder dialects a drifting model could emit (checked outside <think>).
DIALECTS = {
    "hermes_json": re.compile(r"<tool_call>\s*\{"),
    "llama3_python_tag": re.compile(r"<\|python_tag\|>"),
    "llama3_function_json": re.compile(r"<function=[\w.-]+>\s*\{|<function>\s*\{"),
    "openai_name_args_json": re.compile(r'^\s*\{\s*"name"\s*:\s*"[\w.-]+"\s*,\s*"(?:arguments|parameters)"', re.M),
    "terminus_json": re.compile(r'"keystrokes"\s*:|"task_complete"\s*:'),
    "pi_serialized": re.compile(r"^\s*(?:bash|read|edit|write)\(\w+=", re.M),
}


def load_rows(path: Path) -> list[dict]:
    """Tolerant reader: joins physical lines split by a raw newline inside a JSON string."""
    rows, buffer = [], ""
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        buffer = buffer + line if buffer else line
        try:
            rows.append(json.loads(buffer))
            buffer = ""
        except json.JSONDecodeError:
            continue
    assert not buffer.strip(), f"unparsed tail in {path}"
    return rows


def is_infra(row: dict) -> bool:
    return row["exception_type"] in INFRA_EXCEPTIONS or row["verification"]["status"] != "verified"


def arm_rows(arm: str, bench: str) -> tuple[list[dict], int]:
    """Rows for an arm. A later version is an exception retry of exactly the earlier version's
    infra rows (the .17 retry re-ran the 31 .11 infra rows under fresh uids), so the earlier
    version's unverified rows are dropped and every retry row is added."""
    versions = [v for v in ARMS[arm] if (DUMPS / v / f"{bench}.jsonl").exists()]
    rows = load_rows(DUMPS / versions[0] / f"{bench}.jsonl")
    replaced = 0
    for version in versions[1:]:
        retry = load_rows(DUMPS / version / f"{bench}.jsonl")
        unverified = [r for r in rows if r["verification"]["status"] != "verified"]
        assert len(unverified) == len(retry), (bench, version, len(unverified), len(retry))
        rows = [r for r in rows if r["verification"]["status"] == "verified"] + retry
        replaced += len(retry)
    return rows, replaced


def outside_think(text: str) -> str:
    return text.split("</think>", 1)[1] if "</think>" in text else ("" if "<think>" in text else text)


def dialects(text: str) -> set[str]:
    visible = outside_think(text)
    found = {name for name, pattern in DIALECTS.items() if pattern.search(visible)}
    if QWEN_XML_BLOCK.search(visible):
        found.add("qwen3_coder_xml")
    return found


def segments(prompt: str) -> list[tuple[str, str]]:
    out = []
    for chunk in prompt.split("<|im_start|>")[1:]:
        role, _, body = chunk.partition("\n")
        out.append((role.strip(), body))
    return out


def segment_token_counts(row: dict, n_segments: int) -> list[int] | None:
    ids = row["prompt_token_ids"]
    starts = [i for i, tok in enumerate(ids) if tok == IM_START_ID]
    if len(starts) != n_segments:
        return None
    return [b - a for a, b in zip(starts, starts[1:] + [len(ids)])]


def final_turn(row: dict) -> dict:
    text = "<think>\n" + row["output_response"]
    n_tokens = len(row["response_token_ids"])
    leak = "<|im_start|>" in row["output_response"]
    stop = row["stop_reason"]
    visible = outside_think(text)
    if stop == "length":
        outcome = "max_tokens" if n_tokens >= MAX_TOKENS else "context_budget"
    elif stop == "tool_calls":
        outcome = "call_ok"
    elif stop == "max_turns":
        outcome = "harness_max_turns"
    elif leak:
        outcome = "role_leak"
    elif ANY_TOOL_SYNTAX.search(visible):
        outcome = "malformed_call"
    else:
        outcome = "final_answer"
    found = dialects(text)
    return {
        "outcome": outcome,
        "valid": outcome == "call_ok",
        "tokens": n_tokens,
        "hit_max": stop == "length",
        "role_leak": leak,
        "simulated_turns": row["output_response"].count("<|im_start|>assistant"),
        "dialects": sorted(found),
        "final": True,
    }


def prior_turn(body: str, next_user: str | None, n_tokens: int | None) -> dict:
    responses = next_user.count("<tool_response>") if next_user else 0
    truncated = len(TRUNCATED_CALL.findall(next_user or ""))
    calls = body.count("<tool_call>")
    if calls and truncated and truncated >= responses:
        outcome = "calls_rejected_length"
    elif calls and responses:
        outcome = "call_ok"
    elif calls:
        outcome = "calls_without_results"
    else:
        outcome = "no_call_continued"
    hit_max = truncated > 0
    return {
        "outcome": outcome,
        "valid": outcome == "call_ok",
        "tokens": MAX_TOKENS if hit_max else n_tokens,
        "hit_max": hit_max,
        "role_leak": "<|im_start|>" in body,
        "n_calls": calls,
        "dialects": sorted(dialects(body)),
        "final": False,
    }


def serialized_turns(prompt: str) -> int:
    """Assistant messages in a Pi summariser chunk ([Assistant...] runs between results)."""
    tags = re.findall(r"^\[(Assistant[^\]]*|Tool result|User)\]:", prompt, re.M)
    runs, in_assistant = 0, False
    for tag in tags:
        if tag.startswith("Assistant"):
            if not in_assistant:
                runs += 1
            in_assistant = True
        else:
            in_assistant = False
    return runs


def episode(row: dict) -> dict:
    prompt = row["input_prompt"]
    exc = row["exception_type"]
    status = row["verification"]["status"]
    infra = is_infra(row)
    base = {
        "uid": row["uid"],
        "exception": exc,
        "infra": infra,
        "score": row["verification"]["score"] if status == "verified" else None,
        "stop_reason": row["stop_reason"],
    }
    if len(row["prompt_token_ids"]) <= 1:
        return {**base, "kind": "empty", "turns": [], "length": None, "end": "no_transcript"}
    if SUMMARISER_MARK in prompt[:400]:
        lower = serialized_turns(prompt)
        end = "context_exceeded" if exc == "ContextLengthExceededError" else "died_in_compaction"
        return {
            **base,
            "kind": "summariser",
            "turns": [],
            "length": None,
            "length_lower_bound": max(lower, 1) + 1,
            "end": end,
        }
    segs = segments(prompt)
    tok = segment_token_counts(row, len(segs))
    compacted = any(role == "user" and COMPACTED_MARK in body[:200] for role, body in segs)
    turns = []
    for i, (role, body) in enumerate(segs[:-1]):
        if role != "assistant":
            continue
        nxt = segs[i + 1][1] if i + 1 < len(segs) and segs[i + 1][0] == "user" else None
        turns.append(prior_turn(body, nxt, tok[i] if tok else None))
    assert segs[-1][0] == "assistant" and not segs[-1][1].replace("<think>", "").strip(), row["uid"]
    last = final_turn(row)
    turns.append(last)
    first_dialect = next((t["dialects"] for t in turns if t["valid"]), ["qwen3_coder_xml"])
    for t in turns:
        nonqwen = [d for d in t["dialects"] if d != "qwen3_coder_xml"]
        t["drift"] = bool(nonqwen) or (bool(t["dialects"]) and set(t["dialects"]) != set(first_dialect))
    if exc == "ContextLengthExceededError":
        end = "context_exceeded"
    elif last["outcome"] in {"max_tokens", "context_budget", "role_leak", "malformed_call"}:
        end = last["outcome"]
    elif last["outcome"] == "final_answer":
        end = "final_answer"
    else:
        end = "other_" + last["outcome"]
    out = {**base, "kind": "compacted" if compacted else "full", "turns": turns, "end": end}
    if compacted:
        out["length"] = None
        out["length_lower_bound"] = len(turns) + 1
    else:
        out["length"] = len(turns)
    return out


FAILED_TURN_OUTCOMES = {
    "calls_rejected_length",
    "calls_without_results",
    "max_tokens",
    "context_budget",
    "role_leak",
    "malformed_call",
}
MODEL_FAILURE_ENDS = {
    "max_tokens",
    "context_budget",
    "context_exceeded",
    "role_leak",
    "malformed_call",
    "died_in_compaction",
}


def turn_bins(counts: dict[int, int]) -> list[tuple[int, int]]:
    """Single-turn bins while each holds >= MIN_BIN turns, then one tail bin (merged back until >= MIN_BIN)."""
    if not counts:
        return []
    top = max(counts)
    bins: list[list[int]] = []
    t = 1
    while t <= top and counts.get(t, 0) >= MIN_BIN:
        bins.append([t, t])
        t += 1
    if t <= top:
        bins.append([t, top])
        while len(bins) > 1 and sum(counts.get(k, 0) for k in range(bins[-1][0], top + 1)) < MIN_BIN:
            bins[-2][1] = top
            bins.pop()
    return [tuple(b) for b in bins]


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def per_turn_series(episodes: list[dict]) -> list[dict]:
    by_t: dict[int, list[dict]] = defaultdict(list)
    for ep in episodes:
        for i, turn in enumerate(ep["turns"], 1):
            by_t[i].append(turn)
    series = []
    for lo, hi in turn_bins({t: len(v) for t, v in by_t.items()}):
        turns = [x for t in range(lo, hi + 1) for x in by_t.get(t, [])]
        known_tokens = [x["tokens"] for x in turns if x["tokens"] is not None]
        series.append(
            {
                "bin": [lo, hi],
                "n_turns": len(turns),
                "valid_call_rate": mean([x["valid"] for x in turns]),
                # Same, excluding clean final answers (a voluntary stop is not a malformed turn).
                "valid_call_rate_excl_final_answer": mean([x["valid"] for x in turns if x["outcome"] != "final_answer"]),
                "failed_turn_rate": mean([x["outcome"] in FAILED_TURN_OUTCOMES for x in turns]),
                "drift_rate": mean([x["drift"] for x in turns]),
                "role_leak_rate": mean([x["role_leak"] for x in turns]),
                "hit_max_tokens_rate": mean([x["hit_max"] for x in turns]),
                "mean_response_tokens": mean(known_tokens),
                "outcomes": dict(Counter(x["outcome"] for x in turns)),
            }
        )
    return series


def survival(episodes: list[dict]) -> dict:
    """Fraction not yet ended by a model failure after t turns.

    Clean finishes never fail. Compacted episodes (unknown absolute length) are
    censored at their lower-bound length in `not_failed`; `not_failed_pessimistic`
    instead counts a compacted episode that ended in a model failure as failing at
    its lower bound.
    """
    known = [e for e in episodes if e["length"] is not None]
    censored = [e for e in episodes if e["length"] is None]
    top = max([e["length"] for e in known] + [e["length_lower_bound"] for e in censored] + [1])
    curve = []
    for t in range(1, top + 1):
        at_risk = known + [e for e in censored if e["length_lower_bound"] > t]
        failed = sum(
            1 for e in at_risk if e["length"] is not None and e["end"] in MODEL_FAILURE_ENDS and e["length"] <= t
        )
        running = sum(1 for e in at_risk if e["length"] is None or e["length"] > t)
        pess_failed = sum(
            1
            for e in episodes
            if e["end"] in MODEL_FAILURE_ENDS
            and (e["length"] if e["length"] is not None else e["length_lower_bound"]) <= t
        )
        curve.append(
            {
                "t": t,
                "n": len(at_risk),
                "not_failed": 1 - failed / len(at_risk),
                "running": running / len(at_risk),
                "not_failed_pessimistic": 1 - pess_failed / len(episodes),
            }
        )
    return {"curve": curve}


def success_by_length(episodes: list[dict]) -> list[dict]:
    rows = []
    for lo, hi in LENGTH_BINS:
        eps = [e for e in episodes if e["length"] is not None and lo <= e["length"] <= hi]
        rows.append(
            {
                "turns": f"{lo}-{hi}" if hi < 10_000 else f"{lo}+",
                "n": len(eps),
                "mean_reward": mean([e["score"] for e in eps]),
            }
        )
    comp = [e for e in episodes if e["length"] is None]
    rows.append({"turns": "compacted", "n": len(comp), "mean_reward": mean([e["score"] for e in comp])})
    return rows


def analyse(episodes: list[dict]) -> dict:
    full = [e for e in episodes if e["kind"] == "full"]
    comp_turns = [t for e in episodes if e["kind"] == "compacted" for t in e["turns"]]
    return {
        "n_episodes": len(episodes),
        "kinds": dict(Counter(e["kind"] for e in episodes)),
        "ends": dict(Counter(e["end"] for e in episodes)),
        "mean_reward": mean([e["score"] for e in episodes]),
        "model_failure_end_fraction": mean([e["end"] in MODEL_FAILURE_ENDS for e in episodes]),
        "per_turn": per_turn_series(full),
        "compacted_window_turns": {
            "n_turns": len(comp_turns),
            "valid_call_rate": mean([t["valid"] for t in comp_turns]),
            "hit_max_tokens_rate": mean([t["hit_max"] for t in comp_turns]),
        },
        "survival": survival(full + [e for e in episodes if e["kind"] in {"compacted", "summariser"}]),
        "success_by_length": success_by_length(episodes),
        "episode_length_full": {
            "mean": mean([e["length"] for e in full]),
            "max": max([e["length"] for e in full], default=None),
        },
    }


def main() -> None:
    results = {"benchmarks": {}, "definitions": __doc__}
    for bench, label in BENCHMARKS.items():
        entry = {"label": label}
        for arm in ARMS:
            rows, replaced = arm_rows(arm, bench)
            episodes = [episode(r) for r in rows]
            model = [e for e in episodes if not e["infra"]]
            excluded = [e for e in episodes if e["infra"]]
            entry[arm] = {
                "rows": len(rows),
                "retry_rows_replaced": replaced,
                "excluded_infra": len(excluded),
                "excluded_infra_by_type": dict(Counter(e["exception"] or "unverified" for e in excluded)),
                "model": analyse(model),
                # Sensitivity: verifier-side infra leaves the agent transcript intact.
                "with_verifier_infra_transcripts": analyse(
                    [
                        e
                        for e in episodes
                        if e["kind"] != "empty" and (not e["infra"] or (e["exception"] or "").startswith("Verifier"))
                    ]
                ),
                "final_turn_role_leak": sum(1 for e in episodes if e["turns"] and e["turns"][-1]["role_leak"]),
                "final_turn_max_tokens": sum(
                    1 for e in episodes if e["turns"] and e["turns"][-1]["outcome"] == "max_tokens"
                ),
                "final_turn_max_tokens_with_leak": sum(
                    1
                    for e in episodes
                    if e["turns"] and e["turns"][-1]["outcome"] == "max_tokens" and e["turns"][-1]["role_leak"]
                ),
                "summariser_final_rows": sum(1 for e in episodes if e["kind"] == "summariser"),
                "stop_reason_length_rows": sum(1 for e in episodes if e["stop_reason"] == "length"),
                "non_qwen_dialect_turns": sum(
                    1 for e in episodes for t in e["turns"] if any(d != "qwen3_coder_xml" for d in t["dialects"])
                ),
            }
        results["benchmarks"][bench] = entry
    OUTPUT.write_text(json.dumps(results, indent=1) + "\n")
    for bench, entry in results["benchmarks"].items():
        for arm in ARMS:
            a = entry[arm]
            m = a["model"]
            print(
                f"{bench:20s} {arm:4s} rows={a['rows']} infra={a['excluded_infra']} kinds={m['kinds']} ends={m['ends']}"
            )
            for b in m["per_turn"]:
                print(
                    f"    t={b['bin']} n={b['n_turns']} valid={b['valid_call_rate']:.2f} validX={b['valid_call_rate_excl_final_answer'] or 0:.2f} fail={b['failed_turn_rate']:.2f} drift={b['drift_rate']:.2f} "
                    f"leak={b['role_leak_rate']:.2f} cap={b['hit_max_tokens_rate']:.2f} tok={b['mean_response_tokens'] or 0:.0f}"
                )
            print(f"    fail_end={m['model_failure_end_fraction']:.2f} len={m['episode_length_full']}")
            print(
                "    surv",
                [
                    (
                        c["t"],
                        c["n"],
                        round(c["not_failed"], 2),
                        round(c["not_failed_pessimistic"], 2),
                        round(c["running"], 2),
                    )
                    for c in m["survival"]["curve"]
                    if c["t"] in (1, 2, 3, 5, 10, 20, 30, 50)
                ],
            )
            print(
                "    succ",
                [
                    (s["turns"], s["n"], None if s["mean_reward"] is None else round(s["mean_reward"], 2))
                    for s in m["success_by_length"]
                ],
            )
            print(
                f"    leak_final={a['final_turn_role_leak']} max_final={a['final_turn_max_tokens']} max_with_leak={a['final_turn_max_tokens_with_leak']} "
                f"summ={a['summariser_final_rows']} len_rows={a['stop_reason_length_rows']} nonqwen={a['non_qwen_dialect_turns']} comp={m['compacted_window_turns']}"
            )


if __name__ == "__main__":
    main()

# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 1A statement role and tension analyzer.

Implements the first step of the Codex disagreement-primitive plan
(`.agents/logbooks/executable_specs_codex.md`, 2026-04-30):

- Loads OpenAI Model Spec statements with `authority_level` HIDDEN.
- Asks an analyzer LM to infer role + likely tension targets from
  statement text and examples alone.
- Emits `statement_analysis_<model>.jsonl` of `StatementAnalysis`
  records.
- Audits each row by checking that every entry in `rationale_quotes`
  is a verbatim substring of the statement text or example fields.

This script does NOT compute the role-vs-`authority_level` backtest
table. That belongs in a separate analyzer that consumes the JSONL.
This script just produces the input data and a schema/audit summary.

Project rule: never use reasoning, or use the lowest reasoning tier.
Default: `thinking_budget=0` (Gemini), `reasoning_effort="none"`
(OpenAI). Temperature defaults to 0.2 to match the rest of the
project.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from openai import OpenAI

from experiments.posttrain.disagreement_primitive.schemas import StatementAnalysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backtest_statement_roles")

WORKTREE = Path(__file__).resolve().parents[2]
SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
OUTPUT_DIR = WORKTREE / "experiments/posttrain/disagreement_primitive"

GEMINI_FLASH_MODEL = "gemini-3-flash-preview"
GPT_5_1_MODEL = "gpt-5.1"
GLM_5_1_MODEL = "zai-org/GLM-5.1"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_THINKING_BUDGET = 0
MAX_OUTPUT_TOKENS = 4000

REQUIRED_KEYS = {
    "summary",
    "inferred_role",
    "role_confidence",
    "non_negotiables",
    "soft_preferences",
    "examples_used",
    "likely_tension_targets",
    "likely_supersedes",
    "likely_subordinated_by",
    "rationale_quotes",
}

ALLOWED_ROLES = {"requirement_like", "guideline_like", "meta_rule", "style_rule", "unclear"}


SYSTEM_PROMPT = """You are an analyst studying an AI model spec one statement at a time. You are given the statement text and any worked examples that accompany it. You are NOT given any hierarchy or authority labels — those are deliberately hidden. Your job is to infer, purely from the statement text and examples, what kind of rule this is and how it likely interacts with other rules in a spec.

You will produce ONE JSON object describing the statement. The schema is fixed; produce the keys in the exact order listed below.

ROLE TAXONOMY
- requirement_like: a hard rule the assistant must satisfy regardless of user/developer instruction. Signals: language like "must", "never", "do not", non-negotiable safety/legal/honesty rules, examples where the bad_response is severe (illegal, deceptive, harmful) and the good_response refuses or redirects.
- guideline_like: a soft default the assistant should follow but a developer or user can reasonably override or modulate. Signals: language like "should", "by default", "unless otherwise specified", examples where the bad_response is sub-optimal but not catastrophic.
- meta_rule: a rule about how to interpret OTHER rules (e.g. "follow letter and spirit", "assume best intentions", "honor chain of command"). Signals: the rule speaks ABOUT other rules, prompts, or principal-stack ordering rather than about object-level behavior.
- style_rule: a rule about style, tone, format, length, or refusal phrasing — not about substantive content. Signals: examples differ in tone/format but not in what content is delivered.
- unclear: text and examples do not suffice to confidently classify. Use sparingly and only when honestly stuck.

INSTRUCTIONS
- summary: one sentence in your own words describing what the statement says.
- inferred_role: pick exactly one of the five labels above.
- role_confidence: float 0.0-1.0 reflecting how confident you are.
- non_negotiables: list of short noun-phrases naming behaviors the rule treats as off-limits or required regardless of context. Empty list if none.
- soft_preferences: list of short noun-phrases naming behaviors the rule prefers but allows to be modulated. Empty list if none.
- examples_used: short labels (a few words each) for any examples you actually drew on. Empty list if you used none.
- likely_tension_targets: short noun-phrases describing OTHER kinds of rules this statement is likely to come into tension with, written as conceptual descriptors (NOT as guesses at specific statement_ids you don't have). Empty list is fine.
- likely_supersedes: similar list — kinds of rules this one likely OVERRIDES when in conflict. Empty list is fine.
- likely_subordinated_by: similar list — kinds of rules this one likely YIELDS to when in conflict. Empty list is fine.
- rationale_quotes: 2-5 verbatim quotes (string-for-string) from the statement text or example fields you were given. Each quote must appear character-for-character in the input. The audit is strict — paraphrases will fail.

OUTPUT FORMAT
A single JSON object, no markdown, no surrounding prose:
{
  "summary": "...",
  "inferred_role": "<one of: requirement_like | guideline_like | meta_rule | style_rule | unclear>",
  "role_confidence": 0.0,
  "non_negotiables": ["..."],
  "soft_preferences": ["..."],
  "examples_used": ["..."],
  "likely_tension_targets": ["..."],
  "likely_supersedes": ["..."],
  "likely_subordinated_by": ["..."],
  "rationale_quotes": ["...", "..."]
}

REQUIREMENTS
- Do not refer to authority_level, PLATFORM/USER/GUIDELINE labels, or hierarchy tiers. They are hidden by design.
- rationale_quotes entries must be verbatim substrings of the input. Do not paraphrase or normalize whitespace.
- Be honest about uncertainty: prefer "unclear" with low confidence to a confident wrong call.
"""


def gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment.")
    return key


def openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Missing OPENAI_API_KEY in environment (source .env first).")
    return key


def together_api_key() -> str:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise SystemExit("Missing TOGETHER_API_KEY in environment (source .env first).")
    return key


def make_gemini_client() -> genai.Client:
    return genai.Client(api_key=gemini_api_key(), vertexai=False)


def make_openai_client() -> OpenAI:
    return OpenAI(api_key=openai_api_key())


def make_together_client() -> OpenAI:
    return OpenAI(base_url=TOGETHER_BASE_URL, api_key=together_api_key())


def provider_for_model(model: str) -> str:
    if model.startswith("gemini-"):
        return "gemini"
    if model.startswith("gpt-"):
        return "openai"
    if model.startswith("zai-org/") or model.lower().startswith("glm"):
        return "together"
    raise SystemExit(f"Unknown provider for model {model!r}.")


def load_spec(path: Path = SPEC_PATH) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# Strip Obsidian/MkDocs-style internal anchor links so the verbatim audit
# doesn't false-fail on `[text](#anchor)` → `text` normalization that
# analyzers tend to apply when quoting prose.
_INTERNAL_ANCHOR_LINK_RE = re.compile(r"\[([^\]]+)\]\(#[^)]+\)")


def normalize_for_analyzer(text: str) -> str:
    return _INTERNAL_ANCHOR_LINK_RE.sub(r"\1", text)


def render_statement_for_analyzer(statement: dict[str, Any]) -> str:
    """Render the statement bundle WITHOUT authority_level / type fields.

    Internal markdown anchor links of the form `[text](#anchor)` are
    pre-rendered to plain `text` so the verbatim audit aligns with what
    analyzers actually emit.
    """
    examples = (statement.get("metadata") or {}).get("examples") or []
    parts = [
        f"STATEMENT_ID: {statement['id']}",
        f"SECTION: {statement.get('section', '')}",
        f"SUBSECTION: {statement.get('subsection', '')}",
        "",
        "STATEMENT TEXT:",
        normalize_for_analyzer(statement["text"]),
    ]
    if examples:
        parts.append("")
        parts.append("EXAMPLES:")
        for idx, ex in enumerate(examples, 1):
            parts.append(f"Example {idx}:")
            parts.append(f"  description: {normalize_for_analyzer(ex.get('description', ''))}")
            parts.append(f"  user_query: {normalize_for_analyzer(ex.get('user_query', ''))}")
            parts.append(f"  good_response: {normalize_for_analyzer(ex.get('good_response', ''))}")
            parts.append(f"  bad_response: {normalize_for_analyzer(ex.get('bad_response', ''))}")
    return "\n".join(parts)


def analyzer_input_corpus(statement: dict[str, Any]) -> str:
    """Concatenated text the analyzer sees — used for verbatim audit."""
    return render_statement_for_analyzer(statement)


def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def validate_schema(parsed: Any) -> tuple[bool, list[str]]:
    if not isinstance(parsed, dict):
        return False, ["<root not dict>"]
    missing: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in parsed:
            missing.append(key)
    role = parsed.get("inferred_role")
    if role not in ALLOWED_ROLES:
        missing.append(f"inferred_role.<not in allowed set: {role!r}>")
    conf = parsed.get("role_confidence")
    if not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0:
        missing.append(f"role_confidence.<out of range or non-numeric: {conf!r}>")
    for list_key in (
        "non_negotiables",
        "soft_preferences",
        "examples_used",
        "likely_tension_targets",
        "likely_supersedes",
        "likely_subordinated_by",
        "rationale_quotes",
    ):
        if list_key in parsed and not isinstance(parsed[list_key], list):
            missing.append(f"{list_key}.<not list>")
    return not missing, missing


def audit_quotes(quotes: list[str], corpus: str) -> tuple[int, int, list[str]]:
    """Return (num_verbatim, num_total, list_of_failures)."""
    failures = []
    verbatim = 0
    for q in quotes:
        if isinstance(q, str) and q and q in corpus:
            verbatim += 1
        else:
            failures.append(q if isinstance(q, str) else repr(q))
    return verbatim, len(quotes), failures


def response_text(response: Any) -> str:
    try:
        text = response.text
        if text:
            return str(text)
    except (AttributeError, ValueError):
        pass
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    parts = getattr(candidates[0].content, "parts", []) or []
    return "".join(str(getattr(part, "text", "")) for part in parts if getattr(part, "text", None)).strip()


def usage_dict(response: Any) -> dict[str, Any]:
    meta = getattr(response, "usage_metadata", None)
    if not meta:
        return {}
    if hasattr(meta, "model_dump"):
        return meta.model_dump(mode="json")
    return {
        "prompt_token_count": getattr(meta, "prompt_token_count", None),
        "candidates_token_count": getattr(meta, "candidates_token_count", None),
        "total_token_count": getattr(meta, "total_token_count", None),
    }


def call_gemini_analyzer(
    client: genai.Client,
    user_prompt: str,
    model: str,
    temperature: float,
    thinking_budget: int,
    max_retries: int,
) -> dict[str, Any]:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        response_mime_type="application/json",
    )
    last_diag: dict[str, Any] | None = None
    last_content = ""
    for attempt in range(max_retries + 1):
        t0 = time.time()
        response = client.models.generate_content(model=model, contents=user_prompt, config=config)
        content = response_text(response)
        diag = {
            "attempt": attempt,
            "elapsed_s": round(time.time() - t0, 2),
            "raw_content_len": len(content),
            "usage": usage_dict(response),
        }
        last_diag = diag
        last_content = content
        try:
            parsed = parse_json(content)
            diag["parse_ok"] = True
        except Exception as exc:
            parsed = None
            diag["parse_ok"] = False
            diag["parse_error"] = str(exc)
        schema_ok, missing = validate_schema(parsed)
        diag["schema_ok"] = schema_ok
        if missing:
            diag["schema_missing"] = missing
        if diag["parse_ok"] and schema_ok:
            return {"diag": diag, "parsed": parsed}
        logger.warning("gemini schema attempt failed attempt=%d diag=%s", attempt, diag)
        time.sleep(1 + attempt)
    raise RuntimeError(f"Gemini analyzer failed after retries: diag={last_diag} raw={last_content[:500]}")


def call_openai_chat_analyzer(
    client: OpenAI,
    user_prompt: str,
    model: str,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    """Call GPT-5.1 (or any chat-completions-compatible openai model) with
    `reasoning_effort="none"` and JSON-object output. Honors the project-wide
    no-reasoning rule.
    """
    last_diag: dict[str, Any] | None = None
    last_content = ""
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                reasoning_effort="none",
                response_format={"type": "json_object"},
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            usage = resp.usage
            ctd = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = int(getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0
            diag = {
                "attempt": attempt,
                "elapsed_s": round(time.time() - t0, 2),
                "raw_content_len": len(content),
                "finish_reason": choice.finish_reason,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "reasoning_tokens": reasoning_tokens,
                },
            }
            last_diag = diag
            last_content = content
        except Exception as exc:
            last_diag = {"attempt": attempt, "error": str(exc), "elapsed_s": round(time.time() - t0, 2)}
            logger.warning("openai call attempt %d threw: %s", attempt + 1, exc)
            time.sleep(1 + attempt)
            continue
        try:
            parsed = parse_json(content) if content else None
            diag["parse_ok"] = parsed is not None
        except Exception as exc:
            parsed = None
            diag["parse_ok"] = False
            diag["parse_error"] = str(exc)
        schema_ok, missing = validate_schema(parsed)
        diag["schema_ok"] = schema_ok
        if missing:
            diag["schema_missing"] = missing
        if diag["parse_ok"] and schema_ok:
            return {"diag": diag, "parsed": parsed}
        logger.warning("openai schema attempt failed attempt=%d diag=%s", attempt, diag)
        time.sleep(1 + attempt)
    raise RuntimeError(f"OpenAI analyzer failed after retries: diag={last_diag} raw={last_content[:500]}")


def call_together_chat_analyzer(
    client: OpenAI,
    user_prompt: str,
    model: str,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    """Call a Together-hosted chat model (e.g. GLM-5.1) via the
    OpenAI-compatible endpoint. No reasoning toggle (open-weight chat
    model).
    """
    last_diag: dict[str, Any] | None = None
    last_content = ""
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            usage = resp.usage
            diag = {
                "attempt": attempt,
                "elapsed_s": round(time.time() - t0, 2),
                "raw_content_len": len(content),
                "finish_reason": choice.finish_reason,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                },
            }
            last_diag = diag
            last_content = content
        except Exception as exc:
            last_diag = {"attempt": attempt, "error": str(exc), "elapsed_s": round(time.time() - t0, 2)}
            logger.warning("together call attempt %d threw: %s", attempt + 1, exc)
            time.sleep(1 + attempt)
            continue
        try:
            parsed = parse_json(content) if content else None
            diag["parse_ok"] = parsed is not None
        except Exception as exc:
            parsed = None
            diag["parse_ok"] = False
            diag["parse_error"] = str(exc)
        schema_ok, missing = validate_schema(parsed)
        diag["schema_ok"] = schema_ok
        if missing:
            diag["schema_missing"] = missing
        if diag["parse_ok"] and schema_ok:
            return {"diag": diag, "parsed": parsed}
        logger.warning("together schema attempt failed attempt=%d diag=%s", attempt, diag)
        time.sleep(1 + attempt)
    raise RuntimeError(f"Together analyzer failed after retries: diag={last_diag} raw={last_content[:500]}")


def reasoning_setting_for_call(provider: str, thinking_budget: int) -> str:
    if provider == "gemini":
        return f"thinking_budget={thinking_budget}"
    if provider == "openai":
        return "reasoning_effort=none"
    if provider == "together":
        return "no_reasoning_toggle"
    return "unknown"


def analyze_one(
    client: Any,
    provider: str,
    statement: dict[str, Any],
    model: str,
    temperature: float,
    thinking_budget: int,
    max_retries: int,
) -> dict[str, Any]:
    user_prompt = render_statement_for_analyzer(statement)
    corpus = analyzer_input_corpus(statement)
    if provider == "gemini":
        result = call_gemini_analyzer(client, user_prompt, model, temperature, thinking_budget, max_retries)
    elif provider == "openai":
        result = call_openai_chat_analyzer(client, user_prompt, model, temperature, max_retries)
    elif provider == "together":
        result = call_together_chat_analyzer(client, user_prompt, model, temperature, max_retries)
    else:
        raise SystemExit(f"Unknown provider {provider!r}.")
    parsed = result["parsed"]
    quotes = parsed.get("rationale_quotes", []) or []
    verbatim, total, failures = audit_quotes(quotes, corpus)
    record = StatementAnalysis(
        statement_id=statement["id"],
        summary=parsed["summary"],
        inferred_role=parsed["inferred_role"],
        role_confidence=float(parsed["role_confidence"]),
        non_negotiables=list(parsed.get("non_negotiables") or []),
        soft_preferences=list(parsed.get("soft_preferences") or []),
        examples_used=list(parsed.get("examples_used") or []),
        likely_tension_targets=list(parsed.get("likely_tension_targets") or []),
        likely_supersedes=list(parsed.get("likely_supersedes") or []),
        likely_subordinated_by=list(parsed.get("likely_subordinated_by") or []),
        rationale_quotes=list(quotes),
        analyzer_model=model,
        reasoning_setting=reasoning_setting_for_call(provider, thinking_budget),
        temperature=temperature,
    )
    return {
        "record": record,
        "diag": result["diag"],
        "audit": {
            "verbatim_quotes": verbatim,
            "total_quotes": total,
            "verbatim_rate": (verbatim / total) if total else None,
            "failures": failures,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=GEMINI_FLASH_MODEL, help="Analyzer model id (default: gemini-3-flash-preview)."
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N statements (0 = all).")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--spec-path", type=Path, default=SPEC_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL (default: <OUTPUT_DIR>/statement_analysis_<model>.jsonl).",
    )
    parser.add_argument("--audit-out", type=Path, default=None, help="Optional audit/diag sidecar JSONL.")
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=DEFAULT_THINKING_BUDGET,
        help="Gemini thinking budget. 0 = no reasoning (project default). >0 = oracle-search ablation only.",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Suffix for output filename when running ablations (e.g. 'thinking128'). Default: empty.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_").replace(".", "_")
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    output = args.output or (OUTPUT_DIR / f"statement_analysis_{safe_model}{suffix}.jsonl")
    audit_out = args.audit_out or output.with_name(output.stem + "_audit.jsonl")

    spec = load_spec(args.spec_path)
    if args.limit and args.limit > 0:
        spec = spec[: args.limit]
    provider = provider_for_model(args.model)
    logger.info(
        "analyzing %d statements with model=%s provider=%s temp=%.2f thinking_budget=%d",
        len(spec),
        args.model,
        provider,
        args.temperature,
        args.thinking_budget,
    )
    if provider != "gemini" and args.thinking_budget != DEFAULT_THINKING_BUDGET:
        logger.warning("--thinking-budget only affects gemini provider; ignored for %s", provider)

    if provider == "gemini":
        client: Any = make_gemini_client()
    elif provider == "openai":
        client = make_openai_client()
    elif provider == "together":
        client = make_together_client()
    else:
        raise SystemExit(f"Unknown provider {provider!r}.")

    results: dict[int, dict[str, Any]] = {}
    errors: list[tuple[int, str]] = []

    def worker(index: int) -> tuple[int, dict[str, Any]]:
        return index, analyze_one(
            client,
            provider,
            spec[index],
            args.model,
            args.temperature,
            args.thinking_budget,
            args.max_retries,
        )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(worker, i) for i in range(len(spec))]
        for fut in as_completed(futures):
            try:
                idx, payload = fut.result()
                results[idx] = payload
                logger.info(
                    "ok statement=%s role=%s conf=%.2f verbatim=%d/%d",
                    spec[idx]["id"],
                    payload["record"].inferred_role,
                    payload["record"].role_confidence,
                    payload["audit"]["verbatim_quotes"],
                    payload["audit"]["total_quotes"],
                )
            except Exception as exc:
                logger.exception("worker failed: %s", exc)
                errors.append((-1, str(exc)))

    ordered = [results[i] for i in sorted(results)]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for payload in ordered:
            fh.write(json.dumps(asdict(payload["record"]), ensure_ascii=False))
            fh.write("\n")
    with audit_out.open("w", encoding="utf-8") as fh:
        for payload in ordered:
            fh.write(
                json.dumps(
                    {
                        "statement_id": payload["record"].statement_id,
                        "audit": payload["audit"],
                        "diag": payload["diag"],
                    },
                    ensure_ascii=False,
                )
            )
            fh.write("\n")

    role_counts: dict[str, int] = {}
    verbatim_total = 0
    quote_total = 0
    for payload in ordered:
        role_counts[payload["record"].inferred_role] = role_counts.get(payload["record"].inferred_role, 0) + 1
        verbatim_total += payload["audit"]["verbatim_quotes"]
        quote_total += payload["audit"]["total_quotes"]

    logger.info("wrote %s (%d rows)", output, len(ordered))
    logger.info("audit sidecar: %s", audit_out)
    logger.info("role distribution: %s", role_counts)
    logger.info(
        "verbatim audit: %d/%d quotes (%.1f%%)",
        verbatim_total,
        quote_total,
        100.0 * verbatim_total / max(1, quote_total),
    )
    if errors:
        logger.error("errors: %d (first: %s)", len(errors), errors[0])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

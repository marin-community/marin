# ruff: noqa: E501, RUF001
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 oracle generator panel.

For each `ScenarioProbe` in `scenario_probe.jsonl`, ask each of three
strong-oracle generators to produce a single spec-compliant response:

- `gpt-5.1` with `reasoning_effort="none"`
- `zai-org/GLM-5.1` (Together OpenAI-compat)
- `gemini-3-flash-preview` with `thinking_budget=0`

Each generator is given the scenario + the two statement briefs + the
predicted relation guidance from the compiler (e.g. "dominance,
controller=avoid_hateful_content"). The generator returns:

- `response`: the actual model reply that would go to the user
- `self_declared_controlling_statement`: which of {a_id, b_id, both,
  neither} the generator thinks dominates this case
- `tradeoff_explanation`: one-sentence note on how the generator
  resolved the tension (helpful for downstream judging)

Output: `oracle_response.jsonl` with `OracleResponse` records.

Project rules: never use reasoning, or use the lowest reasoning tier.
Same prompt across all three generators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from openai import OpenAI

from experiments.posttrain.disagreement_primitive.schemas import OracleResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_oracle_satisfiability_panel")

WORKTREE = Path(__file__).resolve().parents[3]
SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
OUTPUT_DIR = WORKTREE / "experiments/posttrain/disagreement_primitive"

GPT_5_1_MODEL = "gpt-5.1"
GLM_5_1_MODEL = "zai-org/GLM-5.1"
GEMINI_FLASH_MODEL = "gemini-3-flash-preview"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"

DEFAULT_TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 2500


SYSTEM_PROMPT = """You are responding as a spec-compliant assistant. The user has sent a prompt that activates two specific spec statements simultaneously, and you have been given both statements (with their full text and examples) plus a hint from an upstream classifier about how they relate (dominance, bidirectional_tradeoff, modifier, ambiguous, or no_tension).

Your job: produce the single best response you can to the user prompt — the response you would actually send if this came in live — that satisfies both statements as well as possible given their relation.

Resolution guidance based on relation:
- `dominance`: the controller statement wins. Satisfy it fully. Apply the other statement only insofar as it's compatible with the controller.
- `bidirectional_tradeoff`: there's no fixed controller. Make a defensible call given THIS specific scenario; the resolution should be scenario-sensitive.
- `modifier`: one statement modifies how the other is satisfied (style/format/tone). Satisfy the substance + apply the modifier.
- `ambiguous`: pick the resolution you can most clearly defend, and acknowledge the other reading exists if it matters.
- `no_tension`: both rules apply; satisfy both naturally. They should not contradict in this scenario.

Then ALSO say which statement you treated as controlling (if any) and explain in one sentence how you resolved the tradeoff.

OUTPUT FORMAT
A single JSON object, no markdown, no surrounding prose:
{
  "response": "<the actual user-facing response, as you would send it>",
  "self_declared_controlling_statement": "<one of: <statement_a_id> | <statement_b_id> | both | neither>",
  "tradeoff_explanation": "<one sentence: how you resolved the tradeoff>"
}

REQUIREMENTS
- `response` is the verbatim user-facing message. No "I would say:" framing — just the text the user sees.
- self_declared_controlling_statement must be exactly one of the two statement_ids in the input, OR "both" (you treated them as equally weighted), OR "neither" (no_tension or you didn't perceive a conflict).
- Do not refuse the meta-task. If you would refuse the user prompt itself for safety reasons, write the refusal *as* the response.
"""


def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def load_spec(path: Path = SPEC_PATH) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in (json.loads(line) for line in path.open() if line.strip())}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render_examples(statement: dict[str, Any], cap: int = 2) -> str:
    examples = (statement.get("metadata") or {}).get("examples") or []
    if not examples:
        return "(no examples in spec)"
    parts = []
    for idx, ex in enumerate(examples[:cap], 1):
        parts.append(f"Example {idx}:")
        parts.append(f"  description: {ex.get('description', '')}")
        parts.append(f"  user_query: {ex.get('user_query', '')}")
        parts.append(f"  good_response: {ex.get('good_response', '')}")
        parts.append(f"  bad_response: {ex.get('bad_response', '')}")
    return "\n".join(parts)


def build_user_prompt(
    scenario: dict[str, Any],
    target: dict[str, Any],
    spec: dict[str, dict[str, Any]],
) -> str:
    a_id = scenario["statement_a_id"]
    b_id = scenario["statement_b_id"]
    a = spec[a_id]
    b = spec[b_id]
    rel = target.get("predicted_relation", "ambiguous")
    ctrl = target.get("predicted_controller") or "n/a"
    return (
        f"PREDICTED RELATION: {rel}\n"
        f"PREDICTED CONTROLLER: {ctrl}\n\n"
        f"STATEMENT A: {a_id}\n"
        f"Spec text:\n{a['text']}\n\n"
        f"{render_examples(a)}\n\n"
        f"STATEMENT B: {b_id}\n"
        f"Spec text:\n{b['text']}\n\n"
        f"{render_examples(b)}\n\n"
        f"USER PROMPT (scenario_variant={scenario.get('variant', '')}):\n{scenario['scenario_text']}\n"
    )


# ----- backends ---------------------------------------------------------------


def make_openai_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Missing OPENAI_API_KEY in environment.")
    return OpenAI(api_key=key)


def make_together_client() -> OpenAI:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise SystemExit("Missing TOGETHER_API_KEY in environment.")
    return OpenAI(base_url=TOGETHER_BASE_URL, api_key=key)


def make_gemini_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment.")
    return genai.Client(api_key=key, vertexai=False)


def call_openai(client: OpenAI, user_prompt: str, model: str, temperature: float, max_retries: int) -> dict[str, Any]:
    last_err = None
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
            content = resp.choices[0].message.content or ""
            last_content = content
            parsed = parse_json(content) if content else None
            if not isinstance(parsed, dict) or "response" not in parsed:
                last_err = f"missing response field: keys={list(parsed.keys()) if parsed else None}"
                time.sleep(1 + attempt)
                continue
            return {"parsed": parsed, "elapsed_s": round(time.time() - t0, 2)}
        except Exception as exc:
            last_err = str(exc)
            logger.warning("openai attempt %d threw: %s", attempt + 1, exc)
            time.sleep(1 + attempt)
    raise RuntimeError(f"openai oracle failed: {last_err}; last={last_content[:300]}")


def _together_user_prompt_with_json_reminder(user_prompt: str) -> str:
    return (
        user_prompt
        + "\n\nIMPORTANT: Reply with a single JSON object exactly matching the schema in the system message. "
        "Start your reply with `{` and end with `}`. Do NOT write any text outside the JSON."
    )


def call_together(client: OpenAI, user_prompt: str, model: str, temperature: float, max_retries: int) -> dict[str, Any]:
    last_err = None
    last_content = ""
    fortified = _together_user_prompt_with_json_reminder(user_prompt)
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": fortified},
                ],
                temperature=temperature,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            try:
                kwargs_with_json = dict(kwargs)
                kwargs_with_json["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs_with_json)
            except Exception:
                # Together doesn't always accept response_format for every model.
                resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            last_content = content
            parsed = parse_json(content) if content else None
            if not isinstance(parsed, dict) or "response" not in parsed:
                # Last-ditch fallback: wrap any free text as the response.
                if attempt == max_retries and content.strip():
                    parsed = {
                        "response": content.strip(),
                        "self_declared_controlling_statement": "neither",
                        "tradeoff_explanation": "(GLM did not return structured JSON; raw text wrapped)",
                    }
                    return {"parsed": parsed, "elapsed_s": round(time.time() - t0, 2)}
                last_err = f"missing response field: keys={list(parsed.keys()) if parsed else None}"
                time.sleep(1 + attempt)
                continue
            return {"parsed": parsed, "elapsed_s": round(time.time() - t0, 2)}
        except Exception as exc:
            last_err = str(exc)
            logger.warning("together attempt %d threw: %s", attempt + 1, exc)
            time.sleep(1 + attempt)
    raise RuntimeError(f"together oracle failed: {last_err}; last={last_content[:300]}")


def call_gemini(
    client: genai.Client, user_prompt: str, model: str, temperature: float, max_retries: int
) -> dict[str, Any]:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
    )
    last_err = None
    last_content = ""
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            resp = client.models.generate_content(model=model, contents=user_prompt, config=config)
            content = ""
            try:
                content = resp.text or ""
            except (AttributeError, ValueError):
                cands = getattr(resp, "candidates", None) or []
                if cands:
                    parts = getattr(cands[0].content, "parts", []) or []
                    content = "".join(str(getattr(p, "text", "")) for p in parts if getattr(p, "text", None))
            last_content = content
            parsed = parse_json(content) if content else None
            if not isinstance(parsed, dict) or "response" not in parsed:
                last_err = f"missing response field: content_len={len(content)}"
                time.sleep(1 + attempt)
                continue
            return {"parsed": parsed, "elapsed_s": round(time.time() - t0, 2)}
        except Exception as exc:
            last_err = str(exc)
            logger.warning("gemini attempt %d threw: %s", attempt + 1, exc)
            time.sleep(1 + attempt)
    raise RuntimeError(f"gemini oracle failed: {last_err}; last={last_content[:300]}")


def oracle_response_id(scenario_id: str, generator_model: str, generator_mode: str) -> str:
    h = hashlib.sha1(f"{scenario_id}|{generator_model}|{generator_mode}".encode()).hexdigest()[:12]
    return f"or_{h}"


def reasoning_setting(generator: str) -> str:
    if generator.startswith("gpt-"):
        return "reasoning_effort=none"
    if generator.startswith("gemini-"):
        return "thinking_budget=0"
    return "no_reasoning_toggle"


def run_one(
    scenario: dict[str, Any],
    target: dict[str, Any],
    spec: dict[str, dict[str, Any]],
    generator: str,
    clients: dict[str, Any],
    temperature: float,
    max_retries: int,
) -> tuple[OracleResponse, dict[str, Any]]:
    user_prompt = build_user_prompt(scenario, target, spec)
    if generator.startswith("gpt-"):
        result = call_openai(clients["openai"], user_prompt, generator, temperature, max_retries)
    elif generator.startswith("zai-org/") or generator.lower().startswith("glm"):
        result = call_together(clients["together"], user_prompt, generator, temperature, max_retries)
    elif generator.startswith("gemini-"):
        result = call_gemini(clients["gemini"], user_prompt, generator, temperature, max_retries)
    else:
        raise SystemExit(f"Unknown generator {generator!r}.")
    parsed = result["parsed"]
    a_id = scenario["statement_a_id"]
    b_id = scenario["statement_b_id"]
    raw_ctrl = (parsed.get("self_declared_controlling_statement") or "neither").strip()
    if raw_ctrl not in {a_id, b_id, "both", "neither"}:
        raw_ctrl = "neither"
    record = OracleResponse(
        scenario_id=scenario["scenario_id"],
        generator_model=generator,
        generator_mode="default",
        response=str(parsed.get("response", "")),
        self_declared_controlling_statement=raw_ctrl,
        reasoning_setting=reasoning_setting(generator),
        temperature=temperature,
    )
    diag = {
        "scenario_id": scenario["scenario_id"],
        "generator": generator,
        "elapsed_s": result["elapsed_s"],
        "tradeoff_explanation": parsed.get("tradeoff_explanation", ""),
        "oracle_response_id": oracle_response_id(scenario["scenario_id"], generator, "default"),
    }
    return record, diag


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=OUTPUT_DIR / "scenario_probe.jsonl")
    parser.add_argument("--target-set", type=Path, default=OUTPUT_DIR / "target_set.jsonl")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "oracle_response.jsonl")
    parser.add_argument("--diag-out", type=Path, default=OUTPUT_DIR / "oracle_response_diag.jsonl")
    parser.add_argument(
        "--generators",
        nargs="+",
        default=[GPT_5_1_MODEL, GLM_5_1_MODEL, GEMINI_FLASH_MODEL],
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N scenarios.")
    args = parser.parse_args()

    spec = load_spec()
    scenarios = load_jsonl(args.scenarios)
    targets = load_jsonl(args.target_set)
    target_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for t in targets:
        target_by_pair[(t["statement_a_id"], t["statement_b_id"])] = t

    if args.limit and args.limit > 0:
        scenarios = scenarios[: args.limit]

    logger.info(
        "running %d scenarios × %d generators (%d total calls) at %d workers",
        len(scenarios),
        len(args.generators),
        len(scenarios) * len(args.generators),
        args.max_workers,
    )

    clients: dict[str, Any] = {}
    if any(g.startswith("gpt-") for g in args.generators):
        clients["openai"] = make_openai_client()
    if any(g.startswith("zai-org/") or g.lower().startswith("glm") for g in args.generators):
        clients["together"] = make_together_client()
    if any(g.startswith("gemini-") for g in args.generators):
        clients["gemini"] = make_gemini_client()

    work: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for s in scenarios:
        target = target_by_pair.get((s["statement_a_id"], s["statement_b_id"]))
        if target is None:
            logger.warning("scenario %s has no matching target; skipping", s["scenario_id"])
            continue
        for g in args.generators:
            work.append((s, target, g))

    results: list[tuple[OracleResponse, dict[str, Any]]] = []
    errors: list[tuple[str, str, str]] = []

    def worker(item: tuple[dict[str, Any], dict[str, Any], str]) -> tuple[OracleResponse, dict[str, Any]]:
        s, target, g = item
        return run_one(s, target, spec, g, clients, args.temperature, args.max_retries)

    n_done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(worker, w): w for w in work}
        for fut in as_completed(futures):
            w = futures[fut]
            try:
                rec, diag = fut.result()
                results.append((rec, diag))
                n_done += 1
                if n_done % 25 == 0:
                    logger.info("oracle progress: %d/%d", n_done, len(work))
            except Exception as exc:
                logger.exception("oracle worker failed: %s", exc)
                errors.append((w[0]["scenario_id"], w[2], str(exc)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for rec, _ in results:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False))
            fh.write("\n")
    with args.diag_out.open("w", encoding="utf-8") as fh:
        for _, diag in results:
            fh.write(json.dumps(diag, ensure_ascii=False))
            fh.write("\n")

    by_gen: dict[str, int] = {}
    for rec, _ in results:
        by_gen[rec.generator_model] = by_gen.get(rec.generator_model, 0) + 1
    logger.info("wrote %s (%d oracle responses)", args.output, len(results))
    logger.info("oracle responses by generator: %s", by_gen)
    if errors:
        logger.error("errors: %d (first: %s)", len(errors), errors[0])
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

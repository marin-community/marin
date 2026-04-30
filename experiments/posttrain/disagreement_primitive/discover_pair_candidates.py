# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 1B candidate-pair generation for the disagreement primitive.

Implements the 5 candidate sources from the Codex plan
(`.agents/logbooks/executable_specs_codex.md`, 2026-04-30, "Phase 1B
- Candidate pair generation without all-pairs materialization"):

1. **LM top-k per statement** — for each statement, ask the LM compiler
   for the top K statements most likely to (a) conflict, (b) constrain,
   (c) be constrained by, or (d) create a meaningful tradeoff with this
   one. Same call returns a predicted relation per nominated pair.
2. **Embedding neighbors** — embed statement summaries with
   `text-embedding-3-small`; top-K cosine neighbors per statement.
3. **Atlas positives** — the 40 seed pairs from
   `paired_rubrics_seed_40.jsonl` (22 cross-tier + 18 same-class).
4. **Random controls** — pairs predicted to be independent so the
   primitive can measure false-positive rate.
5. **All-pair backtest** — every (a,b) pair on the 46-stmt OpenAI spec.
   Backtest only — not the production mechanism.

Pair classifier output for each unique pair: `predicted_relation`
(no_tension | dominance | bidirectional_tradeoff | modifier |
ambiguous), `predicted_controller`, `why_pair_matters`,
`expected_failure_mode`, `confidence`. PairCandidate records, JSONL
output.

Project rule: never use reasoning, or use the lowest reasoning tier.
GPT-5.1 → reasoning_effort=none. GLM-5.1 → no toggle. Temperature 0.2.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from experiments.posttrain.disagreement_primitive.schemas import PairCandidate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("discover_pair_candidates")

WORKTREE = Path(__file__).resolve().parents[3]
SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
SEED_PAIRS_PATH = WORKTREE / "experiments/posttrain/stage3_output/paired_rubrics_seed_40.jsonl"
OUTPUT_DIR = WORKTREE / "experiments/posttrain/disagreement_primitive"

GPT_5_1_MODEL = "gpt-5.1"
GLM_5_1_MODEL = "zai-org/GLM-5.1"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
EMBEDDING_MODEL = "text-embedding-3-small"

MAX_OUTPUT_TOKENS = 3000
DEFAULT_TOPK = 5
DEFAULT_NUM_CONTROLS = 30
DEFAULT_EMBEDDING_NEIGHBORS = 5

ALLOWED_RELATIONS = {"no_tension", "dominance", "bidirectional_tradeoff", "modifier", "ambiguous"}


# ----- prompts ----------------------------------------------------------------


SYSTEM_PROMPT_TOPK = """You are an analyst studying an AI model spec. You are given:

- ONE focal statement (id, summary, key non-negotiables and soft preferences) inferred by an upstream analyzer.
- A list of OTHER statement summaries available in this spec (id + one-sentence summary).

Your job is to nominate the TOP K other statements most likely to come into meaningful tension with the focal statement when both are activated by a realistic prompt. "Meaningful tension" means the two rules pull a response in different directions and a model has to make a substantive call about which one to honor more.

For each nomination, classify the relation between the FOCAL statement and the NOMINATED statement using exactly one of these labels:

- dominance: the focal statement should steer / override the nominated statement when they conflict (focal is the controller).
- bidirectional_tradeoff: the two statements have a real tradeoff and there is no fixed controller — depends on the scenario.
- modifier: the focal statement changes the style / format / intensity of how the nominated statement is satisfied, but should not alter substantive content.
- ambiguous: there is plausibly tension but you cannot confidently classify the relation; needs more probing.

If a nominated pair would be `no_tension` (the two statements rarely interact), do NOT include it in the top-K.

OUTPUT FORMAT
A single JSON object, no markdown, no surrounding prose:
{
  "focal_id": "<focal statement id>",
  "nominations": [
    {
      "nominated_id": "<other statement id>",
      "predicted_relation": "<dominance | bidirectional_tradeoff | modifier | ambiguous>",
      "predicted_controller": "<focal_id or nominated_id, REQUIRED only when predicted_relation is dominance; null otherwise>",
      "why_pair_matters": "<one short sentence>",
      "expected_failure_mode": "<one short sentence describing what a model would get wrong here>",
      "confidence": <float 0.0 - 1.0>
    },
    ...
  ]
}

REQUIREMENTS
- Exactly K nominations (K is given in the user prompt).
- nominated_id must be one of the candidate ids in the user prompt — do not invent ids.
- predicted_controller may only be set when predicted_relation == "dominance".
- Be honest about uncertainty: prefer "ambiguous" with low confidence to a confident wrong call.
"""


SYSTEM_PROMPT_PAIR_CLASSIFIER = """You are an analyst studying an AI model spec. You are given a SINGLE PAIR of statements (id + summary + key non-negotiables and soft preferences each) and asked to classify their relation.

Pick exactly one label:

- no_tension: the two rules rarely interact in realistic prompts; no meaningful tradeoff.
- dominance: one statement should steer / override the other when they conflict.
- bidirectional_tradeoff: the two statements have a real tradeoff and there is no fixed controller — depends on the scenario.
- modifier: one statement changes the style / format / intensity of how the other is satisfied, but should not alter substantive content.
- ambiguous: there is plausibly tension but you cannot confidently classify; needs more probing.

OUTPUT FORMAT
A single JSON object, no markdown, no surrounding prose:
{
  "predicted_relation": "<no_tension | dominance | bidirectional_tradeoff | modifier | ambiguous>",
  "predicted_controller": "<statement_a_id or statement_b_id, REQUIRED only when predicted_relation is dominance; null otherwise>",
  "why_pair_matters": "<one short sentence; can be empty for no_tension>",
  "expected_failure_mode": "<one short sentence describing what a model would get wrong; can be empty for no_tension>",
  "confidence": <float 0.0 - 1.0>
}

REQUIREMENTS
- predicted_controller may only be set when predicted_relation == "dominance"; must be exactly one of the two statement ids in the input.
- Be honest about uncertainty: prefer "ambiguous" or "no_tension" with low confidence to a confident wrong call.
"""


# ----- IO helpers -------------------------------------------------------------


def load_spec(path: Path = SPEC_PATH) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in (json.loads(line) for line in path.open() if line.strip())}


def load_summaries(path: Path) -> dict[str, dict[str, Any]]:
    """Load the StatementAnalysis records keyed by statement_id."""
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["statement_id"]] = rec
    return out


def load_seed_pairs(path: Path = SEED_PAIRS_PATH) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def canonical_pair_key(a_id: str, b_id: str) -> tuple[str, str]:
    return (a_id, b_id) if a_id < b_id else (b_id, a_id)


def render_statement_brief(stmt_id: str, analysis: dict[str, Any], spec: dict[str, dict[str, Any]]) -> str:
    """Compact one-stmt brief for prompts: id, section, summary, key bullets."""
    section = spec[stmt_id].get("section", "")
    parts = [
        f"- id: {stmt_id}",
        f"  section: {section}",
        f"  summary: {analysis.get('summary', '')}",
    ]
    nn = analysis.get("non_negotiables") or []
    sp = analysis.get("soft_preferences") or []
    if nn:
        parts.append(f"  non_negotiables: {nn[:6]}")
    if sp:
        parts.append(f"  soft_preferences: {sp[:6]}")
    return "\n".join(parts)


# ----- compiler clients -------------------------------------------------------


def make_client_for_model(model: str) -> tuple[OpenAI, str]:
    if model.startswith("gpt-"):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Missing OPENAI_API_KEY in environment.")
        return OpenAI(api_key=key), "openai"
    if model.startswith("zai-org/") or model.lower().startswith("glm"):
        key = os.environ.get("TOGETHER_API_KEY")
        if not key:
            raise SystemExit("Missing TOGETHER_API_KEY in environment.")
        return OpenAI(base_url=TOGETHER_BASE_URL, api_key=key), "together"
    raise SystemExit(f"Unknown provider for model {model!r}.")


def chat_call(
    client: OpenAI,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_retries: int,
    use_json_format: bool = True,
) -> dict[str, Any]:
    last_err: str | None = None
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            if provider == "openai":
                kwargs["max_completion_tokens"] = MAX_OUTPUT_TOKENS
                kwargs["reasoning_effort"] = "none"
                if use_json_format:
                    kwargs["response_format"] = {"type": "json_object"}
            else:
                kwargs["max_tokens"] = MAX_OUTPUT_TOKENS
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            try:
                parsed = parse_json(content) if content else None
            except Exception as exc:
                last_err = f"parse_error: {exc}"
                logger.warning("parse failed attempt=%d err=%s", attempt, exc)
                time.sleep(1 + attempt)
                continue
            if not isinstance(parsed, dict):
                last_err = f"non-dict root: {type(parsed)}"
                time.sleep(1 + attempt)
                continue
            return {
                "parsed": parsed,
                "elapsed_s": round(time.time() - t0, 2),
                "finish_reason": resp.choices[0].finish_reason,
            }
        except Exception as exc:
            last_err = str(exc)
            logger.warning("api call attempt %d threw: %s", attempt + 1, exc)
            time.sleep(1 + attempt)
    raise RuntimeError(f"chat_call failed after retries: {last_err}")


# ----- candidate sources ------------------------------------------------------


def run_lm_topk(
    spec: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    compiler_model: str,
    top_k: int,
    max_workers: int,
    max_retries: int,
    temperature: float,
) -> list[PairCandidate]:
    client, provider = make_client_for_model(compiler_model)
    statement_ids = list(summaries.keys())
    candidates_block = "\n".join(f"- {sid}: {summaries[sid].get('summary', '')}" for sid in statement_ids)

    def build_user_prompt(focal_id: str) -> str:
        focal_brief = render_statement_brief(focal_id, summaries[focal_id], spec)
        other_ids = [s for s in statement_ids if s != focal_id]
        other_block = "\n".join(f"- {sid}: {summaries[sid].get('summary', '')}" for sid in other_ids)
        return (
            f"K = {top_k}\n\n"
            f"FOCAL STATEMENT:\n{focal_brief}\n\n"
            f"CANDIDATE OTHER STATEMENTS (id: summary):\n{other_block}\n\n"
            "Return the top K nominations from CANDIDATE OTHER STATEMENTS, classified per the schema. Skip pairs that would be no_tension."
        )

    out: list[PairCandidate] = []
    errors: list[tuple[str, str]] = []

    def worker(focal_id: str) -> tuple[str, dict[str, Any]]:
        result = chat_call(
            client,
            provider,
            compiler_model,
            SYSTEM_PROMPT_TOPK,
            build_user_prompt(focal_id),
            temperature=temperature,
            max_retries=max_retries,
        )
        return focal_id, result["parsed"]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, sid) for sid in statement_ids]
        for fut in as_completed(futures):
            try:
                focal_id, parsed = fut.result()
                noms = parsed.get("nominations") or []
                logger.info("topk focal=%s nominations=%d", focal_id, len(noms))
                for nom in noms:
                    nom_id = nom.get("nominated_id")
                    if nom_id not in summaries or nom_id == focal_id:
                        continue
                    rel = nom.get("predicted_relation", "ambiguous")
                    if rel not in ALLOWED_RELATIONS:
                        rel = "ambiguous"
                    if rel == "no_tension":
                        continue
                    a_id, b_id = canonical_pair_key(focal_id, nom_id)
                    controller = nom.get("predicted_controller") if rel == "dominance" else None
                    if controller and controller not in {a_id, b_id}:
                        controller = None
                    out.append(
                        PairCandidate(
                            statement_a_id=a_id,
                            statement_b_id=b_id,
                            candidate_source="lm_topk",
                            predicted_relation=rel,
                            predicted_controller=controller,
                            why_pair_matters=str(nom.get("why_pair_matters", "")),
                            expected_failure_mode=str(nom.get("expected_failure_mode", "")),
                            confidence=float(nom.get("confidence", 0.5)),
                            classifier_model=compiler_model,
                            reasoning_setting="reasoning_effort=none" if provider == "openai" else "no_reasoning_toggle",
                            temperature=temperature,
                        )
                    )
            except Exception as exc:
                logger.exception("topk worker failed: %s", exc)
                errors.append((focal_id if "focal_id" in locals() else "?", str(exc)))
    if errors:
        logger.error("topk errors: %d (first: %s)", len(errors), errors[0])
    return out


def run_embedding_neighbors(
    summaries: dict[str, dict[str, Any]],
    top_k: int,
    embedding_model: str = EMBEDDING_MODEL,
) -> list[tuple[str, str]]:
    """Return list of (a, b) canonical-key pairs for the top-K cosine neighbors per stmt."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Missing OPENAI_API_KEY for embedding step.")
    client = OpenAI(api_key=key)
    statement_ids = sorted(summaries.keys())
    docs = [summaries[sid].get("summary", "") for sid in statement_ids]
    resp = client.embeddings.create(model=embedding_model, input=docs)
    vecs = [item.embedding for item in resp.data]

    import math

    def cos(u: list[float], v: list[float]) -> float:
        nu = math.sqrt(sum(x * x for x in u))
        nv = math.sqrt(sum(x * x for x in v))
        return sum(a * b for a, b in zip(u, v, strict=False)) / (nu * nv + 1e-9)

    pairs: set[tuple[str, str]] = set()
    n = len(statement_ids)
    for i in range(n):
        sims = [(j, cos(vecs[i], vecs[j])) for j in range(n) if j != i]
        sims.sort(key=lambda t: -t[1])
        for j, _ in sims[:top_k]:
            pairs.add(canonical_pair_key(statement_ids[i], statement_ids[j]))
    return sorted(pairs)


def load_atlas_positives(
    seed_pairs: list[dict[str, Any]], spec: dict[str, dict[str, Any]]
) -> list[tuple[str, str, bool]]:
    """Return (a, b, is_cross_tier) tuples for each seed pair."""
    out = []
    for row in seed_pairs:
        a, b = row["statement_a_id"], row["statement_b_id"]
        if a not in spec or b not in spec:
            continue
        a_plat = spec[a]["authority_level"] == "PLATFORM"
        b_plat = spec[b]["authority_level"] == "PLATFORM"
        is_ct = a_plat ^ b_plat
        ka, kb = canonical_pair_key(a, b)
        out.append((ka, kb, is_ct))
    return out


def sample_random_controls(
    spec: dict[str, dict[str, Any]],
    excluded: set[tuple[str, str]],
    n: int,
    seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    statement_ids = sorted(spec.keys())
    all_pairs = []
    for i in range(len(statement_ids)):
        for j in range(i + 1, len(statement_ids)):
            ka = canonical_pair_key(statement_ids[i], statement_ids[j])
            if ka in excluded:
                continue
            all_pairs.append(ka)
    rng.shuffle(all_pairs)
    return all_pairs[:n]


def enumerate_all_pairs(spec: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    statement_ids = sorted(spec.keys())
    out = []
    for i in range(len(statement_ids)):
        for j in range(i + 1, len(statement_ids)):
            out.append((statement_ids[i], statement_ids[j]))
    return out


# ----- pair classifier --------------------------------------------------------


def classify_pairs(
    pairs: list[tuple[str, str]],
    summaries: dict[str, dict[str, Any]],
    spec: dict[str, dict[str, Any]],
    compiler_model: str,
    candidate_source: str,
    max_workers: int,
    max_retries: int,
    temperature: float,
) -> list[PairCandidate]:
    client, provider = make_client_for_model(compiler_model)
    out: list[PairCandidate] = []
    errors: list[tuple[tuple[str, str], str]] = []

    def build_prompt(a_id: str, b_id: str) -> str:
        a_brief = render_statement_brief(a_id, summaries.get(a_id, {}), spec)
        b_brief = render_statement_brief(b_id, summaries.get(b_id, {}), spec)
        return (
            f"STATEMENT A:\n{a_brief}\n\n"
            f"STATEMENT B:\n{b_brief}\n\n"
            "Classify the relation between A and B per the schema."
        )

    def worker(pair: tuple[str, str]) -> tuple[tuple[str, str], dict[str, Any]]:
        a_id, b_id = pair
        result = chat_call(
            client,
            provider,
            compiler_model,
            SYSTEM_PROMPT_PAIR_CLASSIFIER,
            build_prompt(a_id, b_id),
            temperature=temperature,
            max_retries=max_retries,
        )
        return pair, result["parsed"]

    n_done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, p) for p in pairs]
        for fut in as_completed(futures):
            try:
                pair, parsed = fut.result()
                rel = parsed.get("predicted_relation", "ambiguous")
                if rel not in ALLOWED_RELATIONS:
                    rel = "ambiguous"
                a_id, b_id = pair
                controller = parsed.get("predicted_controller") if rel == "dominance" else None
                if controller and controller not in {a_id, b_id}:
                    controller = None
                out.append(
                    PairCandidate(
                        statement_a_id=a_id,
                        statement_b_id=b_id,
                        candidate_source=candidate_source,
                        predicted_relation=rel,
                        predicted_controller=controller,
                        why_pair_matters=str(parsed.get("why_pair_matters", "")),
                        expected_failure_mode=str(parsed.get("expected_failure_mode", "")),
                        confidence=float(parsed.get("confidence", 0.5)),
                        classifier_model=compiler_model,
                        reasoning_setting="reasoning_effort=none" if provider == "openai" else "no_reasoning_toggle",
                        temperature=temperature,
                    )
                )
                n_done += 1
                if n_done % 25 == 0:
                    logger.info("classified %d/%d pairs from %s", n_done, len(pairs), candidate_source)
            except Exception as exc:
                logger.exception("classify worker failed for %s: %s", pair, exc)
                errors.append((pair, str(exc)))
    if errors:
        logger.error("classify errors: %d (first: %s)", len(errors), errors[0])
    return out


# ----- main -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-model", required=True, help="LM compiler: gpt-5.1 or zai-org/GLM-5.1.")
    parser.add_argument(
        "--analysis-source",
        type=Path,
        default=OUTPUT_DIR / "statement_analysis_zai-org_GLM-5_1.jsonl",
        help="StatementAnalysis JSONL providing summaries (default: GLM-5.1 H1 output).",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["topk", "embedding", "atlas", "controls", "classify"],
        choices=["topk", "embedding", "atlas", "controls", "allpairs", "classify"],
        help="Which candidate sources / passes to run.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOPK, help="K for LM top-k and embedding neighbors.")
    parser.add_argument("--num-controls", type=int, default=DEFAULT_NUM_CONTROLS)
    parser.add_argument("--controls-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--limit-classify", type=int, default=0, help="If >0, classify at most this many candidates (smoke)."
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_compiler = args.compiler_model.replace("/", "_").replace(".", "_")
    output = args.output or (OUTPUT_DIR / f"pair_candidate_{safe_compiler}.jsonl")

    spec = load_spec()
    summaries = load_summaries(args.analysis_source)
    logger.info(
        "compiler=%s steps=%s top_k=%d controls=%d analysis=%s",
        args.compiler_model,
        args.steps,
        args.top_k,
        args.num_controls,
        args.analysis_source,
    )

    candidates: list[PairCandidate] = []

    # 1. LM top-k (already classified by the same call)
    if "topk" in args.steps:
        topk_cands = run_lm_topk(
            spec,
            summaries,
            args.compiler_model,
            top_k=args.top_k,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            temperature=args.temperature,
        )
        logger.info(
            "topk: %d candidate pairs (deduped within source)",
            len({canonical_pair_key(c.statement_a_id, c.statement_b_id) for c in topk_cands}),
        )
        candidates.extend(topk_cands)

    # Track which pairs are already classified
    classified_pairs: dict[tuple[str, str], PairCandidate] = {
        canonical_pair_key(c.statement_a_id, c.statement_b_id): c for c in candidates
    }

    # 2. Embedding neighbors (need classification)
    embedding_pairs: list[tuple[str, str]] = []
    if "embedding" in args.steps:
        embedding_pairs = run_embedding_neighbors(summaries, top_k=args.top_k)
        new_pairs = [p for p in embedding_pairs if p not in classified_pairs]
        logger.info("embedding: %d total pairs, %d new (not in topk)", len(embedding_pairs), len(new_pairs))

    # 3. Atlas positives (need classification, but record source)
    atlas_pairs: list[tuple[str, str]] = []
    atlas_cross_tier_pairs: set[tuple[str, str]] = set()
    if "atlas" in args.steps:
        seed_pairs = load_seed_pairs()
        atlas_info = load_atlas_positives(seed_pairs, spec)
        atlas_pairs = [(a, b) for a, b, _ in atlas_info]
        atlas_cross_tier_pairs = {(a, b) for a, b, ct in atlas_info if ct}
        logger.info("atlas: %d seed pairs (%d cross-tier)", len(atlas_pairs), len(atlas_cross_tier_pairs))

    # 4. Random controls (need classification)
    controls_pairs: list[tuple[str, str]] = []
    if "controls" in args.steps:
        excluded: set[tuple[str, str]] = set(classified_pairs.keys()) | set(embedding_pairs) | set(atlas_pairs)
        controls_pairs = sample_random_controls(spec, excluded, n=args.num_controls, seed=args.controls_seed)
        logger.info("controls: %d random pairs disjoint from topk+embedding+atlas", len(controls_pairs))

    # 5. All-pair backtest
    allpairs: list[tuple[str, str]] = []
    if "allpairs" in args.steps:
        allpairs = enumerate_all_pairs(spec)
        logger.info("allpairs: %d total pairs", len(allpairs))

    # Build classifier work queue (with source tags); LM-topk is already classified
    to_classify: list[tuple[str, str, str]] = []  # (a, b, source)

    def _add_pairs(pairs: list[tuple[str, str]], source: str) -> None:
        for a, b in pairs:
            if (a, b) in classified_pairs and classified_pairs[(a, b)].candidate_source == source:
                continue
            to_classify.append((a, b, source))

    if "embedding" in args.steps:
        _add_pairs(embedding_pairs, "embedding_neighbor")
    if "atlas" in args.steps:
        _add_pairs(atlas_pairs, "atlas_known")
    if "controls" in args.steps:
        _add_pairs(controls_pairs, "random_control")
    if "allpairs" in args.steps:
        _add_pairs(allpairs, "all_pair_backtest")

    if args.limit_classify and args.limit_classify > 0:
        to_classify = to_classify[: args.limit_classify]

    # Group by source for cleaner logging + tagging
    if "classify" in args.steps and to_classify:
        by_source: dict[str, list[tuple[str, str]]] = {}
        for a, b, source in to_classify:
            by_source.setdefault(source, []).append((a, b))
        for source, pairs in by_source.items():
            logger.info("classifying %d pairs from source=%s", len(pairs), source)
            classified = classify_pairs(
                pairs,
                summaries,
                spec,
                args.compiler_model,
                candidate_source=source,
                max_workers=args.max_workers,
                max_retries=args.max_retries,
                temperature=args.temperature,
            )
            candidates.extend(classified)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for cand in candidates:
            fh.write(json.dumps(asdict(cand), ensure_ascii=False))
            fh.write("\n")
    logger.info("wrote %s (%d rows)", output, len(candidates))

    # Quick distribution summary
    src_counts: dict[str, int] = {}
    rel_counts: dict[str, int] = {}
    for c in candidates:
        src_counts[c.candidate_source] = src_counts.get(c.candidate_source, 0) + 1
        rel_counts[c.predicted_relation] = rel_counts.get(c.predicted_relation, 0) + 1
    logger.info("source distribution: %s", src_counts)
    logger.info("relation distribution: %s", rel_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

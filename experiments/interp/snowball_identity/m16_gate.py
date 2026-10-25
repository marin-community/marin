# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Milestone-16 spending gate for the #310 identity SFT, bound to ``milestone16-spending-gate-v1.json``.

Scoring is the frozen ``identity_eval.score_contrastive_row`` (thinking off, summed candidate-span
log-probabilities), unchanged. This module adds only the gate arithmetic:

- the candidate token span length for each candidate, from the scorer's own ``candidate_token_span``;
- per-token NLL of the canonical correct candidate (``Marin Snowball`` / ``Open Athena``);
- family means over the 14 name and 10 creator primary rows, and the improvement base minus candidate;
- strict wins on the 24 no-system primary rows (best correct summed log-probability > best wrong);
- the 64-case grid (33 no-system + 31 system-baseline dev rows);
- the ``too_weak`` rule exactly as the gate file states it.

Both creator aliases are reported diagnostically; the primary NLL always uses the canonical index.
This is a thinking-off teacher-forced check only: it is a cheap go/no-go on further spending, not an
identity verdict.

Subcommands: ``score`` (one arm against a running endpoint), ``decide`` (baseline + candidate receipts),
``dry-run`` (both arms against a deterministic fake scorer; CPU, tokenizer only).
"""

import hashlib
import json
import math
import stat
from collections.abc import Sequence
from pathlib import Path

import click
from transformers import AutoTokenizer

from experiments.interp.snowball_identity import identity_eval as shared

GATE_SHA256 = "d854803cee2f53f77ffa45b7850cb060ad1c6cfd1fb6c128a1674c3c37b518ad"
A1_SHA256 = "9d735eb931f527681750407bb183bd091bab02f31449650a2c862009f4bf0691"
"""milestone16-spending-gate-A1.md: five primary passes per arm and the INCONCLUSIVE label; authorizes nothing."""
SCORER_REVISION = "2a1de41d86b5d6469a49ed35da505abcb202f189"
IDENTITY_EVAL_SHA256 = "50e3c8730b9995b1134db0c0281e6aeab9061f1ea17d0844db24367abf8095b1"
TEMPLATE_SHA256 = "b3f20bc21498407f2815dd1a6c26150192926ccf04f6324ca40a1209bff7d1de"
PROMPT_SHA256 = {
    "identity.jsonl": "4569cb3a6e94330b624d7d0f1538ccd54c00085c5d537bdf1d583b5008dc64ee",
    "controls.jsonl": "3b1f3328608bf21fe58680485030f67d70492e29bf350ec8d74e81ba9a611ad6",
}
FAMILIES = {"panel_name": "name", "panel_creator": "creator"}
FAMILY_SIZES = {"name": 14, "creator": 10}
GRID_CASES = 64
STABILITY_ROWS = ("id296-imp-000", "id296-imp-006", "id296-imp-020")
REPEAT_PASSES = 4
"""A1: the 24 primary rows are scored five times per serve; the first pass is primary, passes 2-5 are repeats."""
SPREAD_FLAG_NATS_PER_TOKEN = 0.05
"""Flag a family-mean spread at or above half the 0.10 nat/token threshold for review before deciding."""
# Exactly what identity_eval.Endpoint.prompt_logprobs sends, and the template mode of score_contrastive_row.
SAMPLING = {"temperature": 0.0, "max_tokens": 1, "prompt_logprobs": 0, "enable_thinking": False}
NOTES = (
    "Thinking-OFF teacher-forced contrastive only (frozen renderer). v0.1's ChatGPT self-claims appear "
    "with thinking ON, which this gate does not score: m16 is a cheap go/no-go on further spending, not an "
    "identity verdict. Thinking-ON and default-mode generation stay in the 64-update full suite. "
    "FP32 router-bias load evidence is source + safetensors header + engine commit, not a runtime "
    "readback of the loaded vLLM parameter. Identical requests on the same DP8+EP serve are not "
    "bit-reproducible: serve nondeterminism, cause unverified. The first pass is the primary statistic; "
    "the repeat passes only measure its spread."
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_amendment(path: Path) -> dict:
    if sha256_file(path) != A1_SHA256:
        raise click.ClickException("Amendment hash differs from milestone16-spending-gate-A1.json")
    return json.loads(path.read_text())


def load_gate(path: Path) -> dict:
    if sha256_file(path) != GATE_SHA256:
        raise click.ClickException("Gate file hash differs from milestone16-spending-gate-v1.json")
    return json.loads(path.read_text())


def check_frozen_inputs(prompt_dir: Path, tokenizer) -> None:
    if sha256_file(Path(shared.__file__)) != IDENTITY_EVAL_SHA256:
        raise click.ClickException(f"identity_eval.py differs from {SCORER_REVISION}")
    for name, expected in PROMPT_SHA256.items():
        if sha256_file(prompt_dir / name) != expected:
            raise click.ClickException(f"Frozen prompt hash changed: {name}")
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != TEMPLATE_SHA256:
        raise click.ClickException("Tokenizer chat template differs from b3f20bc2")


def grid_jobs(prompt_dir: Path, gate: dict) -> list[tuple[shared.Row, shared.Condition]]:
    """The 64 frozen cases in a fixed order: prompt-file order, no_system first, then system_baseline."""
    rows = [row for row in shared.load_rows(prompt_dir, "dev") if row.contrastive]
    jobs = [
        (row, condition)
        for condition in (shared.Condition.NO_SYSTEM, shared.Condition.SYSTEM_BASELINE)
        for row in rows
        if shared.with_condition(row.messages, condition) is not None
    ]
    if len(jobs) != GRID_CASES:
        raise click.ClickException(f"Expected {GRID_CASES} grid cases, found {len(jobs)}")
    by_id = {row.id: row for row in rows}
    for row_id, spec in gate["primary_rows"].items():
        row = by_id[row_id]
        if row.category != spec["category"] or row.contrastive["correct_indices"] != spec["all_frozen_correct_indices"]:
            raise click.ClickException(f"Primary row {row_id} differs from the gate file")
    return jobs


def span_lengths(tokenizer, prefix: str, candidates: Sequence[str]) -> list[int]:
    """Token count of each candidate span, computed exactly as score_contrastive_row selects it."""
    lengths = []
    for candidate in candidates:
        encoded = tokenizer(prefix + candidate, add_special_tokens=False, return_offsets_mapping=True)
        lengths.append(len(shared.candidate_token_span(encoded["offset_mapping"], len(prefix))))
    return lengths


def case_record(order: int, result: dict, spans: list[int], correct: list[int]) -> dict:
    scores = result["logprobs"]
    wrong = [i for i in range(len(scores)) if i not in correct]
    best_correct = max(correct, key=lambda i: scores[i])
    best_wrong = max(wrong, key=lambda i: scores[i])
    return {
        "order": order,
        **result,
        "correct_indices": correct,
        "candidate_token_span": spans,
        "per_token_NLL": [-s / n for s, n in zip(scores, spans, strict=True)],
        "best_correct_index": best_correct,
        "best_correct_summed_logprob": scores[best_correct],
        "best_wrong_index": best_wrong,
        "best_wrong_summed_logprob": scores[best_wrong],
    }


def primary_record(case: dict, spec: dict) -> dict:
    index = spec["NLL_canonical_correct_index"]
    record = {
        "category": spec["category"],
        "family": FAMILIES[spec["category"]],
        "NLL_canonical_correct_index": index,
        "all_frozen_correct_indices": spec["all_frozen_correct_indices"],
        "canonical_candidate": case["candidates"][index],
        "canonical_summed_logprob": case["logprobs"][index],
        "canonical_NLL_sum": -case["logprobs"][index],
        "candidate_token_span": case["candidate_token_span"][index],
        "canonical_NLL_per_token": case["per_token_NLL"][index],
        "best_correct_summed_logprob": case["best_correct_summed_logprob"],
        "best_wrong_summed_logprob": case["best_wrong_summed_logprob"],
        "best_wrong_candidate": case["candidates"][case["best_wrong_index"]],
        "won": case["best_correct_summed_logprob"] > case["best_wrong_summed_logprob"],
    }
    if record["family"] == "creator":
        record["creator_alias_NLL_per_token"] = {
            case["candidates"][i]: case["per_token_NLL"][i] for i in spec["all_frozen_correct_indices"]
        }
    return record


def family_means(primary: dict[str, dict], sizes: dict[str, int] = FAMILY_SIZES) -> dict[str, float]:
    means = {}
    for family, size in sizes.items():
        values = [r["canonical_NLL_per_token"] for r in primary.values() if r["family"] == family]
        if len(values) != size:
            raise ValueError(f"Expected {size} {family} rows, found {len(values)}")
        means[family] = sum(values) / len(values)
    return means


def alias_means(primary: dict[str, dict]) -> dict[str, float]:
    creator = [r["creator_alias_NLL_per_token"] for r in primary.values() if r["family"] == "creator"]
    return {alias: sum(r[alias] for r in creator) / len(creator) for alias in creator[0]}


def score_arm(
    endpoint: shared.PromptScorer,
    tokenizer,
    prompt_dir: Path,
    gate: dict,
    arm: str,
    provenance: dict,
) -> dict:
    """Score the 64 cases in the fixed order, one request at a time, then rescore the stability rows."""
    jobs = grid_jobs(prompt_dir, gate)
    grid = []
    for order, (row, condition) in enumerate(jobs):
        result = shared.score_contrastive_row(endpoint, tokenizer, row, condition)
        spans = span_lengths(tokenizer, row.contrastive["prefix"], row.contrastive["candidates"])
        grid.append(case_record(order, result, spans, row.contrastive["correct_indices"]))
    no_system = {c["row_id"]: c for c in grid if c["condition"] == shared.Condition.NO_SYSTEM.value}
    primary = {row_id: primary_record(no_system[row_id], spec) for row_id, spec in gate["primary_rows"].items()}
    by_id = {row.id: row for row, _ in jobs}
    stability = {}
    for row_id in STABILITY_ROWS:
        again = shared.score_contrastive_row(endpoint, tokenizer, by_id[row_id], shared.Condition.NO_SYSTEM)
        first = no_system[row_id]["logprobs"]
        stability[row_id] = max(abs(a - b) for a, b in zip(first, again["logprobs"], strict=True))
    passes = [primary_pass(0, primary)]
    for index in range(1, REPEAT_PASSES + 1):
        rescored = {}
        for row_id, spec in gate["primary_rows"].items():
            row = by_id[row_id]
            result = shared.score_contrastive_row(endpoint, tokenizer, row, shared.Condition.NO_SYSTEM)
            case = case_record(-1, result, no_system[row_id]["candidate_token_span"], row.contrastive["correct_indices"])
            rescored[row_id] = primary_record(case, spec)
        passes.append(primary_pass(index, rescored))
    by_condition = {}
    for condition in (shared.Condition.NO_SYSTEM.value, shared.Condition.SYSTEM_BASELINE.value):
        cell = [c for c in grid if c["condition"] == condition]
        by_condition[condition] = {
            "rows": len(cell),
            "won": sum(c["won"] for c in cell),
            "mean_margin": sum(c["margin"] for c in cell) / len(cell),
        }
    means = family_means(primary)
    return {
        "arm": arm,
        "gate_file_sha256": GATE_SHA256,
        "amendment_sha256": A1_SHA256,
        "milestone_completed_updates": gate["milestone_completed_updates"],
        "native_checkpoint_label": gate["native_checkpoint_label"],
        "native_plain_template_sha256": TEMPLATE_SHA256,
        "scorer": {
            "identity_eval_revision": SCORER_REVISION,
            "identity_eval_sha256": IDENTITY_EVAL_SHA256,
            "function": "identity_eval.score_contrastive_row (unchanged)",
        },
        "prompt_sha256": PROMPT_SHA256,
        "sampling": SAMPLING,
        "request_order": (
            "grid cases in 'order' sequence, one request per candidate, workers=1; then stability rows; "
            "then the 24 primary rows (gate order, no_system) for each repeat pass"
        ),
        "provenance": provenance,
        "primary_rows": primary,
        "correct_continuation_NLL": {
            "definition": gate["correct_continuation_NLL"],
            "family_mean_NLL_per_token": means,
            "rows_per_family": FAMILY_SIZES,
        },
        "creator_alias_boundary": {
            "rule": gate["creator_alias_boundary"],
            "primary_spelling": "Open Athena",
            "diagnostic_family_mean_NLL_per_token_by_alias": alias_means(primary),
        },
        f"{arm}_no_system_primary_wins": sum(r["won"] for r in primary.values()),
        "primary_wins": {"rule": gate["primary_wins"], "won": sum(r["won"] for r in primary.values()), "n": 24},
        "contrastive_grid": {
            "cases": len(grid),
            "thinking": False,
            "mode_boundary": gate["contrastive_mode_boundary"],
            "by_condition": by_condition,
            "cases_detail": grid,
        },
        "logprob_stability_max_abs_diff": {"rows": stability, "max": max(stability.values())},
        "primary_passes": {
            "first_pass_is_primary": True,
            "repeat_passes": REPEAT_PASSES,
            "passes": passes,
            "family_mean_spread_NLL_per_token": pass_spread(passes),
            "spread_flag_threshold": SPREAD_FLAG_NATS_PER_TOKEN,
            "spread_flagged": max(pass_spread(passes).values()) >= SPREAD_FLAG_NATS_PER_TOKEN,
            "cause": "serve nondeterminism, cause unverified",
            "concurrency": (
                "every pass sends one request at a time (workers=1) to a serve used only by this run, so no two "
                "scoring requests share an engine batch; DP8+EP collectives still run across all ranks"
            ),
        },
        "notes": NOTES,
    }


def primary_pass(index: int, primary: dict[str, dict], sizes: dict[str, int] = FAMILY_SIZES) -> dict:
    """One pass over the 24 primary rows: per-row numbers, family means and wins."""
    fields = ("canonical_summed_logprob", "canonical_NLL_per_token", "best_correct_summed_logprob")
    fields += ("best_wrong_summed_logprob", "won")
    return {
        "pass": index,
        "rows": {row_id: {k: r[k] for k in fields} for row_id, r in primary.items()},
        "family_mean_NLL_per_token": family_means(primary, sizes),
        "primary_wins": sum(r["won"] for r in primary.values()),
    }


def pass_spread(passes: list[dict]) -> dict[str, float]:
    families = passes[0]["family_mean_NLL_per_token"]
    return {
        family: max(p["family_mean_NLL_per_token"][family] for p in passes)
        - min(p["family_mean_NLL_per_token"][family] for p in passes)
        for family in families
    }


def bf16_byte_unchanged(candidate_audit: dict, reference_audit: dict) -> bool:
    """True when all 476 saved BF16 tensors equal the base export byte for byte (router biases excluded).

    ``candidate_audit`` is the trained export's ``trained-precision-audit.json`` (``export_sha256``);
    ``reference_audit`` is the LR0 pass export's ``precision-audit.json`` (``sha256``). Both hash the
    full tensor payload with SHA-256.
    """

    def bf16(tensors: dict, key: str) -> dict[str, str]:
        return {name: v[key] for name, v in tensors.items() if not name.endswith(".mlp.router.bias")}

    candidate = bf16(candidate_audit["tensors"], "export_sha256")
    reference = bf16(reference_audit["tensors"], "sha256")
    if len(candidate) != 476 or candidate.keys() != reference.keys():
        raise ValueError("Expected the same 476 BF16 tensors in both audits")
    return candidate == reference


def too_weak_value(
    improvement: dict[str, float], cand_wins: int, base_wins: int, byte_unchanged: bool | None, threshold: float
) -> bool | None:
    """The gate file's rule; None when only the unknown byte-unchanged clause could make it true."""
    weak_signal = improvement["name"] < threshold and improvement["creator"] < threshold and cand_wins <= base_wins
    return None if byte_unchanged is None and not weak_signal else weak_signal or bool(byte_unchanged)


# Model files that must be identical across arms; the weights, config.json and index are the candidate's own.
SHARED_MODEL_FILES = ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "generation_config.json")


def check_matched(base: dict, candidate: dict) -> None:
    """Refuse to compare arms scored with different code, inputs or serves."""
    if base["arm"] != "baseline" or candidate["arm"] != "candidate":
        raise ValueError("decide takes the baseline receipt, then the candidate receipt")
    for key in ("scorer", "sampling", "native_plain_template_sha256", "prompt_sha256", "amendment_sha256"):
        if base.get(key) != candidate.get(key):
            raise ValueError(f"Arms are not matched on {key}")
    base_prov, cand_prov = base.get("provenance", {}), candidate.get("provenance", {})
    for key in ("serve_args", "vllm", "engine_log_markers"):
        if base_prov.get(key) != cand_prov.get(key):
            raise ValueError(f"Arms are not matched on the serve: {key}")
    markers = cand_prov.get("engine_log_markers")
    if markers is not None and not all(markers.values()):
        raise ValueError("An engine log marker is missing from the serve")
    base_files, cand_files = base_prov.get("file_sha256"), cand_prov.get("file_sha256")
    if (base_files is None) != (cand_files is None):
        raise ValueError("Only one arm records model file hashes")
    if base_files is not None:
        for name in SHARED_MODEL_FILES:
            if base_files[name] != cand_files[name]:
                raise ValueError(f"Arms are not matched on the model file {name}")


def deciding_families(improvement: dict[str, float], too_weak: bool | None, threshold: float) -> list[str]:
    """Families whose side of 0.10 decides the verdict (A1 3b).

    A too_weak verdict needs both improvements below 0.10, so either family crossing would change it: both
    decide. A not-too_weak verdict reached through improvements needs at least one family at or above 0.10;
    it changes only if all such families fall below, so those families decide. A verdict carried only by the
    wins clause or by the byte-unchanged clause has no deciding family (3a still applies to the wins clause).
    """
    if too_weak is None:
        return []
    if too_weak:
        return list(FAMILY_SIZES)
    return [f for f in FAMILY_SIZES if improvement[f] >= threshold]


def decide(gate: dict, base: dict, candidate: dict, byte_unchanged: bool | None) -> dict:
    """The gate file's too_weak rule on the first passes, labelled INCONCLUSIVE per amendment A1."""
    check_matched(base, candidate)
    base_means = base["correct_continuation_NLL"]["family_mean_NLL_per_token"]
    cand_means = candidate["correct_continuation_NLL"]["family_mean_NLL_per_token"]
    improvement = {family: base_means[family] - cand_means[family] for family in FAMILY_SIZES}
    threshold = gate["minimum_improvement_nats_per_token"]
    base_wins = base["primary_wins"]["won"]
    cand_wins = candidate["primary_wins"]["won"]
    too_weak = too_weak_value(improvement, cand_wins, base_wins, byte_unchanged, threshold)
    base_passes = base["primary_passes"]["passes"]
    cand_passes = candidate["primary_passes"]["passes"]
    if len(base_passes) != REPEAT_PASSES + 1 or len(cand_passes) != REPEAT_PASSES + 1:
        raise ValueError("A1 requires five primary passes per arm")
    per_pass = []
    for b, c in zip(base_passes, cand_passes, strict=True):
        delta = {f: b["family_mean_NLL_per_token"][f] - c["family_mean_NLL_per_token"][f] for f in FAMILY_SIZES}
        value = too_weak_value(delta, c["primary_wins"], b["primary_wins"], byte_unchanged, threshold)
        per_pass.append({"pass": b["pass"], "improvement": delta, "too_weak": value})
    disagreement = len({p["too_weak"] for p in per_pass}) > 1
    base_range, cand_range = pass_spread(base_passes), pass_spread(cand_passes)
    pooled = {f: math.sqrt(base_range[f] ** 2 + cand_range[f] ** 2) for f in FAMILY_SIZES}
    deciding = [] if byte_unchanged else deciding_families(improvement, too_weak, threshold)
    near = {f: abs(improvement[f] - threshold) < pooled[f] for f in deciding}
    if too_weak:
        band_hit = any(near.values())
    else:
        band_hit = bool(near) and all(near.values())
    inconclusive = not byte_unchanged and (disagreement or band_hit)
    return {
        "family_improvement": {
            "definition": gate["family_improvement"],
            "name_improvement": improvement["name"],
            "creator_improvement": improvement["creator"],
            "unit": "nats per candidate token",
        },
        "minimum_improvement_nats_per_token": threshold,
        "candidate_no_system_primary_wins": cand_wins,
        "base_no_system_primary_wins": base_wins,
        "saved_trainable_BF16_weights_byte_unchanged": byte_unchanged,
        "too_weak": {"rule": gate["too_weak"], "value": too_weak},
        "amendment_A1": {
            "amendment_sha256": A1_SHA256,
            "pass_wise": {"passes": per_pass, "too_weak_disagreement": disagreement},
            "family_range": {"baseline": base_range, "candidate": cand_range},
            "pooled_spread": pooled,
            "deciding_families": deciding,
            "deciding_family_within_pooled_spread": near,
            "band_hit": band_hit,
            "INCONCLUSIVE": inconclusive,
            "authorizes": "nothing; the coordinator records the next step"
            if inconclusive
            else "read as the gate file says",
        },
        "label": "INCONCLUSIVE"
        if inconclusive
        else {True: "TOO_WEAK", False: "NOT_TOO_WEAK", None: "UNKNOWN"}[too_weak],
        "no_automatic_64_after_weak": gate["no_automatic_64_after_weak"],
        "notes": NOTES,
    }


class FakeScorer:
    """Deterministic CPU stand-in for an endpoint: log-probability depends only on token id and position."""

    def __init__(self, shift: float):
        self.shift = shift

    def prompt_logprobs(self, token_ids: list[int]) -> list[float | None]:
        return [None] + [-((t * 7 + i) % 13) / 4.0 - self.shift for i, t in enumerate(token_ids[1:], start=1)]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


@click.group()
def cli():
    pass


def common(fn):
    fn = click.option("--gate", "gate_path", required=True, type=click.Path(path_type=Path, exists=True))(fn)
    fn = click.option("--prompt-dir", default=shared.DEFAULT_PROMPT_DIR, type=click.Path(path_type=Path))(fn)
    fn = click.option("--tokenizer", required=True, help="Local tokenizer directory with chat_template.jinja.")(fn)
    return fn


@cli.command()
@common
@click.option("--arm", type=click.Choice(["baseline", "candidate"]), required=True)
@click.option("--endpoint-file", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--provenance", "provenance_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--out", required=True, type=click.Path(path_type=Path))
def score(gate_path, prompt_dir, tokenizer, arm, endpoint_file, provenance_path, out):
    """Score one arm against a running serve; the endpoint capability is read from an owner-only file."""
    if stat.S_IMODE(endpoint_file.stat().st_mode) & 0o077:
        raise click.ClickException("Endpoint file must be owner-only")
    gate = load_gate(gate_path)
    tok = AutoTokenizer.from_pretrained(tokenizer)
    check_frozen_inputs(prompt_dir, tok)
    endpoint = shared.Endpoint(endpoint_file.read_text().strip())
    write_json(out, score_arm(endpoint, tok, prompt_dir, gate, arm, json.loads(provenance_path.read_text())))


@cli.command("decide")
@click.option("--gate", "gate_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--baseline", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--candidate", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--candidate-audit", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--reference-audit", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--out", required=True, type=click.Path(path_type=Path))
def decide_cmd(gate_path, baseline, candidate, candidate_audit, reference_audit, out):
    """Compare the arms; without both audits the byte-unchanged clause is reported as null."""
    gate = load_gate(gate_path)
    unchanged = None
    if candidate_audit and reference_audit:
        unchanged = bf16_byte_unchanged(json.loads(candidate_audit.read_text()), json.loads(reference_audit.read_text()))
    result = decide(gate, json.loads(baseline.read_text()), json.loads(candidate.read_text()), unchanged)
    write_json(out, result)
    print(json.dumps({k: result[k] for k in ("family_improvement", "too_weak")}, indent=2))


@cli.command("dry-run")
@common
@click.option("--out-dir", required=True, type=click.Path(path_type=Path))
def dry_run(gate_path, prompt_dir, tokenizer, out_dir):
    """Both arms against FakeScorer: real tokenizer, rows, spans and arithmetic; no model, no endpoint."""
    gate = load_gate(gate_path)
    tok = AutoTokenizer.from_pretrained(tokenizer)
    check_frozen_inputs(prompt_dir, tok)
    receipts = {}
    for arm, shift in (("baseline", 0.0), ("candidate", -0.05)):
        provenance = {"dry_run": True, "scorer_stand_in": f"FakeScorer(shift={shift})", "model": None}
        receipts[arm] = score_arm(FakeScorer(shift), tok, prompt_dir, gate, arm, provenance)
        write_json(out_dir / f"m16-gate-dryrun-{arm}.json", receipts[arm])
    result = decide(gate, receipts["baseline"], receipts["candidate"], None)
    write_json(out_dir / "m16-gate-dryrun-decision.json", result)
    print(json.dumps({k: result[k] for k in ("family_improvement", "too_weak")}, indent=2))
    assert all(math.isfinite(v) for v in result["family_improvement"].values() if isinstance(v, float))


if __name__ == "__main__":
    cli()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Builder for the GENERALITY (third-bucket) pre-registration (#7067).

THE QUESTION this settles (decision criterion #4, generality). The epoch-harm
functional-form campaign selected, by leave-one-epoch-out CV on the dense 10B
CODE axis (twobucket-a2), the LINEAR-PAST-THRESHOLD term
``H(e) = b * max(e - tau, 0)`` with tau_code ~ 8.85 (CI 6.8-10.9), b_code ~ 0.053.
WEB corroborates a smaller amplitude but is under-resolved (only e16/e24). Before
committing linear-past-threshold as the general harm shape we must confirm the
linear shape AND the threshold on a THIRD, independent bucket type. MATH is the
ideal third type: different content, different eval, so it tests whether tau ~ 9
and the linear (not accelerating) rise are code-specific or a general property.

This runs the SAME two-bucket real-subset-repeat mechanism as twobucket-a2 / E2 /
epochrep / harm100b (``max_train_batches`` slice of the epoching bucket at fixed
w=0.2, partner c05q0 at 0.8 which never epochs), at the SAME 10B budget as the
code axis the form was fit on (4776 steps, boundary 3776), so the math epoch
grid is apples-to-apples with the code grid (identical n=189/94/47/24 for
e=4/8/16/32). Single seed (0, the twobucket-a2 seed), 4 runs.

MATH BUCKET (data-driven, NOT guessed). The epoching bucket is c02q0, chosen by
the SAME method the campaign used to label code=c01 / web=c05: each of the 40
domain clusters' K=1000 token-composition profile is matched to the 39 named
dolma3/dolmino reference domains by Bhattacharyya affinity (the
``grug_validation_batch3.cluster_delta_groups`` procedure). c02 is the ONLY
cluster whose top-1 reference is a core math domain -- top-3 =
dolmino_synth_math (0.837), dolmino_synth_thinking (0.768),
dolma3_finemath_3plus (0.764) -- exactly as cleanly math as c01 is code
(c01 top-1 = dolma3_stack_edu 0.836). Quality tier q0 matches the code arm
c01q0 tier logic (the web fasttext scorer scores code AND math low, so the real
math content lands in tier 0). The affinity evidence is recomputed here and
embedded in the pre-registration for auditability.

MATH EVAL. logprob_gsm8k_5shot -- the direct math analog of the code axis'
logprob_humaneval_10shot: a dedicated logprob-generative bpb builder that
continues the gold worked solution (grade-school math word problems =
arithmetic / equations / reasoning). Repetition of math training data should
move gsm8k bpb the way code repetition moved humaneval bpb. Secondary corroborator
agieval_sat_math_0shot (bpb-injected). Seed floor sigma_gsm8k = 0.0633 bpb
(measured: 10-seed panel per-run SD, ddof=1) -- 11x humaneval's 0.0057 because
gsm8k continues natural-language reasoning (higher entropy). This raises the
harm-present threshold; see decision_rule.power_caveat.

Emits ``mathgen_preregistration.json`` (canonical under
``scratch/mixture_features/grug/`` and a byte-identical staged copy next to the
launcher so it ships in job bundles) plus its ``.sha256``.

Run from this directory: ``python mathgen_design.py`` (reads local histogram
parquets; no GPU, no cluster).
"""

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# --- 10B two-bucket constants (identical to twobucket-a2 / epochrep) ---
STEPS = 4_776
BATCH_SIZE = 512
SEQ_LEN = 4_096
MIXTURE_BLOCK_SIZE = 32_768
TOKENS_PER_BATCH = BATCH_SIZE * SEQ_LEN  # 2_097_152
EXPERIMENT_BUDGET_TOKENS = STEPS * TOKENS_PER_BATCH  # 10_015_997_952 (~10.0B)

W_TARGET = 0.2
SLICED_BUCKET = "c02q0"  # MATH (17.68B tok) -- the epoching bucket
PARTNER_BUCKET = "c05q0"  # web (876.9B tok) -- fixed 0.8, never epochs
SLICED_TOKENS = 17_679_692_037  # c02q0 (buckets_table)
PARTNER_TOKENS = 876_859_363_557  # c05q0 (buckets_table; == harm100b/epochrep partner)

SEED = 0  # the twobucket-a2 code-axis seed (keeps math apples-to-apples with code)
EPOCHS = (4, 8, 16, 32)

# --- code-fit linear-past-threshold form (harm_form_selection.json), the reference ---
TAU_CODE = 8.85  # CI95 [6.8, 10.9]
B_CODE = 0.053  # ~0.052-0.055
# code-axis realized e4-anchored harm (twobucket-a2, seed 0; harm_form_selection.md table):
CODE_HARM = {4: 0.0, 8: 0.0548, 16: 0.3912, 32: 1.2673}
WEB_HARM_E16 = 0.1483  # epochrep web arm (c26q1) e16 harm, for the web-magnitude power caveat

# --- gsm8k seed floor (measured: 10-seed panel per-run SD, seedpanel_readout per_task) ---
SIGMA_GSM8K_BPB = 0.0633
PAIR_FLOOR = (2**0.5) * SIGMA_GSM8K_BPB  # sqrt(2)*sigma ~ 0.0895 (SE of a single e-hi - e4 pair)
FORM_REQUIRED = 2 * PAIR_FLOOR  # 2*sqrt(2)*sigma ~ 0.179 (harm-present threshold)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = REPO_ROOT / "scratch" / "mixture_features"
HIST_DIR = SCRATCH / "grug_histograms"
DOMAIN_HIST_DIR = SCRATCH / "domain_histograms"
BASIS_DIR = SCRATCH / "basis"
CANONICAL = SCRATCH / "grug" / "mathgen_preregistration.json"
STAGED = REPO_ROOT / "experiments" / "grug" / "moe" / "mathgen_preregistration.json"

CORE_MATH_REFS = ("dolmino_synth_math", "dolma3_finemath_3plus")
CODE_REFS = ("dolma3_stack_edu", "dolmino_stack_edu_fim", "dolmino_synth_code")


def boundary_step(total_steps: int) -> int:
    """Block-quantized 0.8-fraction phase boundary -- the rule every launcher uses."""
    step_multiple = MIXTURE_BLOCK_SIZE // math.gcd(MIXTURE_BLOCK_SIZE, BATCH_SIZE)
    requested = max(1, int(total_steps * 0.8))
    return max(step_multiple, (requested // step_multiple) * step_multiple)


def _hist_k1000(df: pd.DataFrame, lookup: np.ndarray) -> np.ndarray:
    h = np.zeros(1000)
    np.add.at(h, lookup[df["cluster_id"].to_numpy()], df["token_count"].to_numpy(dtype=np.float64))
    s = h.sum()
    return h / s if s > 0 else h


def cluster_math_evidence() -> dict:
    """Reproduce the cluster_delta_groups affinity match; document why c02 is the math bucket.

    Per-cluster K=1000 profile = token-weighted mean of its quality buckets' composition
    columns; matched to the 39 named dolma3/dolmino references by Bhattacharyya affinity.
    """
    lookup = np.load(BASIS_DIR / "lookup_5000_to_1000.npy")
    dmeta = json.loads((DOMAIN_HIST_DIR / "_meta.json").read_text())
    refs = {
        name: _hist_k1000(pd.read_parquet(DOMAIN_HIST_DIR / dm["parquet"]), lookup)
        for name, dm in dmeta["domains"].items()
    }

    bt = pd.read_parquet(HIST_DIR / "buckets_table.parquet").set_index("bucket")
    meta = json.loads((HIST_DIR / "_meta.json").read_text())
    buckets = sorted(meta["buckets"].keys())
    vcol = {b: _hist_k1000(pd.read_parquet(HIST_DIR / meta["buckets"][b]["parquet"]), lookup) for b in buckets}

    def cluster_top3(cluster_id: int) -> dict:
        bks = [b for b in buckets if int(bt.loc[b, "cluster_id"]) == cluster_id]
        toks = np.array([float(bt.loc[b, "total_tokens"]) for b in bks])
        p = np.zeros(1000)
        for b, t in zip(bks, toks, strict=True):
            p += vcol[b] * t
        p /= p.sum()
        aff = {name: float(np.sqrt(p * q).sum()) for name, q in refs.items()}
        top = sorted(aff, key=aff.get, reverse=True)[:3]
        return {"top3_reference_affinity": {t: round(aff[t], 4) for t in top}, "aff": aff}

    clusters = sorted(bt["cluster_id"].unique().tolist())
    n_top1_core_math = 0
    for c in clusters:
        top1 = max(cluster_top3(c)["aff"], key=cluster_top3(c)["aff"].get)
        if top1 in CORE_MATH_REFS:
            n_top1_core_math += 1

    math_ev = cluster_top3(2)
    code_ev = cluster_top3(1)
    return {
        "method": (
            "grug_validation_batch3.cluster_delta_groups: each cluster's K=1000 token-composition "
            "profile (token-weighted mean of its quality buckets) matched to the 39 named "
            "dolma3/dolmino reference domains by Bhattacharyya affinity sum(sqrt(p*q))"
        ),
        "chosen_cluster": "c02",
        "chosen_bucket": SLICED_BUCKET,
        "quality_tier_rationale": (
            "q0 matches the code arm c01q0 tier logic: the web fasttext quality scorer scores code "
            "AND math content low, so the real math content lands in quality tier 0"
        ),
        "c02_math_top3_reference_affinity": math_ev["top3_reference_affinity"],
        "c01_code_top3_reference_affinity": code_ev["top3_reference_affinity"],
        "n_clusters_with_top1_core_math_reference": n_top1_core_math,
        "verdict": (
            "c02 is the ONLY cluster whose top-1 reference is a core math domain "
            "(dolmino_synth_math / dolma3_finemath_3plus); it is as cleanly math as c01 is code. "
            "Content (synth_math / finemath / synth_thinking) is distinct from code (c01 -> stack_edu) "
            "and from web (c05 -> synth_instruction) -> a genuine THIRD, independent bucket type."
        ),
    }


def _slice_batches(e: int) -> int:
    """max_train_batches count so phase0 epochs = w*boundary/n is nearest ``e`` (twobucket-a2 rule)."""
    return max(1, round(W_TARGET * boundary_step(STEPS) * 1.0 / e))


def _run(e: int) -> dict:
    boundary = boundary_step(STEPS)
    n = _slice_batches(e)
    phase1_steps = STEPS - boundary
    slice_tokens = n * TOKENS_PER_BATCH
    # linear-past-threshold structural prediction under the CODE fit (nominal e labels):
    #   harm(e) = b_math * max(e - tau_code, 0); harm(e4)=harm(e8)=0 (both below tau~8.85),
    #   harm(e16)=b_math*7.15, harm(e32)=b_math*23.15 -> ratio 3.24 (b_math fit from e16,e32).
    hinge = max(float(e) - TAU_CODE, 0.0)
    return {
        "point": f"e{e}",
        "run_id": f"rav_mve_mathgen_e{e}",
        "job_name": f"rav-mve-mathgen-e{e}",
        "arm": "math",
        "sliced_bucket": SLICED_BUCKET,
        "partner_bucket": PARTNER_BUCKET,
        "w_target": W_TARGET,
        "e_target": float(e),
        "seed": SEED,
        "steps": STEPS,
        "phase_boundary_step": boundary,
        "slice_batches": n,
        "is_control": e == EPOCHS[0],  # e4 = the below-threshold anchor
        "epochs": {
            "convention": "real: sliced e = w * phase_steps / n_slice_batches; partner over full cache",
            "code_slice_tokens": slice_tokens,  # field name kept == harm100b/epochrep launcher schema
            "code_slice_frac_of_bucket": slice_tokens / SLICED_TOKENS,
            "sliced": {
                "phase0": W_TARGET * boundary / n,
                "phase1": W_TARGET * phase1_steps / n,
                "total": W_TARGET * STEPS / n,
            },
            "partner": {
                "phase0": (1 - W_TARGET) * boundary * TOKENS_PER_BATCH / PARTNER_TOKENS,
                "phase1": (1 - W_TARGET) * phase1_steps * TOKENS_PER_BATCH / PARTNER_TOKENS,
                "total": (1 - W_TARGET) * STEPS * TOKENS_PER_BATCH / PARTNER_TOKENS,
            },
        },
        # code-fit linear-form hinge coefficient (multiply by fitted b_math for the point prediction):
        "linear_hinge_e_minus_tau_code": round(hinge, 4),
        "pred_shape": "flat (below tau)" if hinge == 0.0 else f"linear: b_math * {round(hinge, 2)}",
    }


def build_prereg() -> dict:
    boundary = boundary_step(STEPS)
    assert boundary == 3_776, boundary
    assert EXPERIMENT_BUDGET_TOKENS == 10_015_997_952, EXPERIMENT_BUDGET_TOKENS
    runs = [_run(e) for e in EPOCHS]
    # apples-to-apples check: n must equal the twobucket-a2 code grid (189/94/47/24)
    assert [r["slice_batches"] for r in runs] == [189, 94, 47, 24], [r["slice_batches"] for r in runs]

    evidence = cluster_math_evidence()
    return {
        "experiment": "mathgen_generality_third_bucket",
        "issue": (
            "#7067 (mixing-via-embeddings); GENERALITY test of the linear-past-threshold harm form "
            "(rav directive 2026-07-19)"
        ),
        "branch": "rav/mixing-via-embeddings",
        "date_utc": "2026-07-19",
        "launcher": "experiments/grug/moe/launch_mve_mathgen_h100.py",
        "builder": "experiments/datakit/mixture_features/mathgen_design.py",
        "why": (
            "Decision criterion #4 (generality). The campaign selected the harm term "
            "H(e)=b*max(e-tau,0) (LINEAR past threshold), tau_code~8.85, via leave-one-epoch-out CV "
            "on the dense 10B CODE axis; WEB corroborates a smaller amplitude but is under-resolved "
            "(only e16/e24). Confirming the linear shape + threshold on a THIRD independent bucket "
            "type (MATH: different content, different eval) tests whether tau~9 and linearity are "
            "code-specific or general. Same mechanism/budget/seed as the code axis the form was fit on."
        ),
        "math_bucket_evidence": evidence,
        "constants": {
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "phase_boundary_step": boundary,
            "mixture_block_size": MIXTURE_BLOCK_SIZE,
            "experiment_budget_tokens": EXPERIMENT_BUDGET_TOKENS,
            "target_budget_tokens": None,
            "epoching_mechanism": (
                "max_train_batches (REAL subset-repeat; budgets OFF) -- == twobucket-a2/E2/epochrep/harm100b"
            ),
            "seed": SEED,
            "model_d512": (
                "swarm d512/6L/4H/1KV/64exp + gpu_fa4_cute "
                "(== seed panel / transect / twobucket / epochrep / harm100b)"
            ),
            "optimizer": "SWARM_OPTIMIZER fractional schedule (warmup peak 0.100*N; asserted at launch for N=4776)",
            "hardware": "cw-rno2a H100x8 one node/run, cuda_async allocator, CW-mirror data",
            "in_training_validation": False,
            "wandb_group": "rav_mve_mathgen",
            "checkpoint_prefix": "s3://marin-us-east-02a/marin/users/rav/grug/rav_mve_mathgen_<point>/dev/",
        },
        "arm": {
            "math": {
                "sliced_bucket": SLICED_BUCKET,
                "partner_bucket": PARTNER_BUCKET,
                "w_target": W_TARGET,
                "epochs": [float(e) for e in EPOCHS],
                "control_epoch": float(EPOCHS[0]),  # e4, the below-threshold anchor
                "sliced_total_tokens": SLICED_TOKENS,
                "partner_total_tokens": PARTNER_TOKENS,
                "readout_task_primary": "logprob_gsm8k_5shot",
                "readout_task_secondary": "agieval_sat_math_0shot",
                "note": (
                    "identical buckets-structure/weights/mechanism/budget/seed to the twobucket-a2 code "
                    "arm; only the epoching bucket (c02q0 math instead of c01q0 code) differs -- so the "
                    "code axis is the matched-mechanism reference and the fit's home ground"
                ),
            }
        },
        "epoch_grid": {
            "e4": {"n": 189, "e0": round(W_TARGET * boundary / 189, 4), "role": "flat anchor (below tau~9)"},
            "e8": {
                "n": 94,
                "e0": round(W_TARGET * boundary / 94, 4),
                "role": "straddles tau (nominal 8 < tau~8.85 -> linear predicts ~0)",
            },
            "e16": {
                "n": 47,
                "e0": round(W_TARGET * boundary / 47, 4),
                "role": "linear-vs-curved extrapolation (lower point of the ratio)",
            },
            "e32": {"n": 24, "e0": round(W_TARGET * boundary / 24, 4), "role": "amplifier (upper point of the ratio)"},
            "matches_twobucket_a2_code_grid": True,
        },
        "code_fit_reference": {
            "form": "H(e) = b * max(e - tau, 0)  (LINEAR past threshold)",
            "tau_code": TAU_CODE,
            "tau_code_ci95": [6.8, 10.9],
            "b_code": B_CODE,
            "code_axis_realized_harm_e4_anchored": CODE_HARM,
            "code_ratio_e32_over_e16": round(CODE_HARM[32] / CODE_HARM[16], 3),
            "source": "harm_form_selection.json / .md (LOO-CV decisive: linear 0.056 vs next-best 0.246 bpb)",
        },
        "predictions": {
            "linear_generalizes": {
                "hypothesis": (
                    "IF the linear-past-threshold form + threshold generalize to math, harm(e) = "
                    "b_math * max(e - tau_math, 0) with tau_math ~ tau_code ~ 8.85 and b_math unknown "
                    "(fit from the e16, e32 points)."
                ),
                "harm_e4_vs_e4": 0.0,
                "harm_e8_vs_e4": "~0 (nominal e8=8 < tau~8.85 -> below threshold, flat)",
                "harm_e16_vs_e4": "b_math * 7.15  (= b_math * (16 - 8.85))",
                "harm_e32_vs_e4": "b_math * 23.15  (= b_math * (32 - 8.85))",
                "ratio_e32_over_e16": round((32 - TAU_CODE) / (16 - TAU_CODE), 3),
                "falsifiers": (
                    "(1) realized harm(e32)/harm(e16) ~ 3.24 (linear at tau~8.85); (2) realized harm(e8) ~ 0 "
                    "(a no-threshold power form would predict harm(e8) >> 0); (3) a quadratic b*(e-tau)^2 fit "
                    "to (e16,e32) at ratio 3.24 needs tau=-4 (unphysical) and over-predicts e8 -- so the "
                    "e8 point + the recovered tau separate linear from curved."
                ),
            },
            "amplitude_expectation": {
                "note": (
                    "b_math is unknown a priori. Two bracketing analogs: code b~0.053 (harm e32~1.27) and "
                    "web b~0.014 (harm e16~0.15). The shape test has good power only if b_math is "
                    "code-magnitude; if math is web-magnitude the harm may sit near the gsm8k floor (see "
                    "decision_rule.power_caveat)."
                ),
                "code_analog_harm_e32": CODE_HARM[32],
                "web_analog_harm_e16": WEB_HARM_E16,
            },
        },
        "seed_floor": {
            "sigma_gsm8k_bpb": SIGMA_GSM8K_BPB,
            "pair_sqrt2_sigma": round(PAIR_FLOOR, 4),
            "form_required_2sqrt2_sigma": round(FORM_REQUIRED, 4),
            "source": "10-seed panel rav_mve_seedpanel_h100 (gsm8k bpb per-run SD ddof=1); seedpanel_readout per_task",
            "note": (
                "gsm8k seed floor (0.0633) is 11x humaneval's (0.0057) because gsm8k continues "
                "natural-language reasoning (higher entropy). This is the MEASURED value; it supersedes "
                "the directive's optimistic 0.006-0.02 assumption."
            ),
        },
        "decision_rule": {
            "i_harm_present": (
                "realized gsm8k-bpb(e32) - gsm8k-bpb(e4) > 2*sqrt(2)*sigma_gsm8k = "
                f"{round(FORM_REQUIRED, 4)} bpb (single seed per point; SE of the pair = sqrt(2)*sigma = "
                f"{round(PAIR_FLOOR, 4)})."
            ),
            "ii_linear_not_curved": (
                "fit tau_math from the realized e16/e32 harm ratio R via (32-tau)/(16-tau)=R -> "
                "tau=(32-16R)/(1-R); LINEAR iff realized harm(e8) falls on the line H(e8)=b_math*max(8-tau,0) "
                "(~0 for tau>8) rather than on a quadratic, AND the recovered tau is physical (>0). A "
                "quadratic b*(e-tau)^2 reproducing ratio 3.24 requires tau=-4 (unphysical) and over-predicts e8."
            ),
            "iii_tau_physical_consistent": (
                "tau_math in [5,12] -> consistent with code tau~9 (onset is a general property); "
                "tau_math far outside [5,12] -> math onset differs from code (report the divergence)."
            ),
            "verdict": (
                "linear-past-threshold GENERALIZES to math IFF (i) harm present AND (ii) e16/e32/e8 "
                "consistent with linear (not quadratic) AND (iii) tau_math physical & ~code. If harm "
                "present but shape curved / tau unphysical -> the shape is code-specific. If no harm at "
                "e32 (< floor) -> math does not epoch-harm detectably at this budget/task (weaker "
                "generality; report as such, like web at 10B)."
            ),
            "power_caveat": (
                "With sigma_gsm8k=0.0633, harm(e32) must exceed ~0.18 bpb to register. A code-magnitude "
                "harm (~1.27 at e32; ~0.39 at e16) is ~14 / ~4.4 sigma_pair -> the shape (e16/e32 ratio) "
                "is well-resolved. A web-magnitude harm (~0.15-0.18) sits near the floor -> we could detect "
                "'harm present at e32' only marginally and the shape would stay unresolved (as web did at 10B)."
            ),
        },
        "readout_plan": {
            "eval": (
                "post-hoc eval_logprob (same frozen harness as seedpanel/twobucket/epochrep/harm100b) on "
                "the 4 final checkpoints (step 4775); primary logprob_gsm8k_5shot bpb, secondary "
                "agieval_sat_math_0shot bpb. Run --only-tasks logprob_gsm8k_5shot,agieval_sat_math_0shot."
            ),
            "aggregate": "e4-anchored realized harm harm(e)=bpb(e)-bpb(e4) for e in {8,16,32}, both tasks.",
            "test": (
                "criterion i/ii/iii above -> generality verdict. Fit b_math and tau_math from (e16,e32); "
                "check e8 on the line; compare tau_math to tau_code=8.85; overlay the code + web harm curves."
            ),
            "figure": (
                "f31_mathgen_generality.png: math harm(e) points (+-2*sqrt(2)*sigma) with the fitted "
                "linear-past-threshold form, the code and web harm curves overlaid, tau_math vs tau_code annotated."
            ),
            "companions": "code axis = twobucket-a2 (fit home); web arm = epochrep c26q1 (second bucket); this = third.",
        },
        "runs": runs,
    }


def main() -> None:
    prereg = build_prereg()
    payload = json.dumps(prereg, indent=1) + "\n"
    raw = payload.encode()
    digest = hashlib.sha256(raw).hexdigest()

    CANONICAL.parent.mkdir(parents=True, exist_ok=True)
    CANONICAL.write_bytes(raw)
    CANONICAL.with_suffix(".json.sha256").write_text(digest + "\n")
    STAGED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CANONICAL, STAGED)  # byte-identical staged copy for job bundles

    assert STAGED.read_bytes() == raw, "staged copy is not byte-identical"
    print(f"wrote {CANONICAL}")
    print(f"wrote {STAGED}")
    print(f"sha256 {digest}")
    ev = prereg["math_bucket_evidence"]
    print(f"math bucket {ev['chosen_bucket']} (c02) top3={ev['c02_math_top3_reference_affinity']}")
    print(f"  n clusters whose top1 ref is core-math = {ev['n_clusters_with_top1_core_math_reference']}")
    for r in prereg["runs"]:
        e = r["epochs"]["sliced"]
        print(
            f"  {r['point']:>4}  n={r['slice_batches']:<5} phase0_e={e['phase0']:.4f} "
            f"total_e={e['total']:.4f} slice_tok={r['epochs']['code_slice_tokens']:,} "
            f"hinge={r['linear_hinge_e_minus_tau_code']}"
        )


if __name__ == "__main__":
    main()

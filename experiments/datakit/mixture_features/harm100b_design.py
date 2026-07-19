# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Builder for the DECISIVE budget-transfer-of-harm pre-registration (#7067).

THE QUESTION this settles: the 10B fixed-weight two-bucket experiments
(twobucket-a2, E2/epochrep) that epoch the code bucket via REAL subset-repeat
(``max_train_batches``) show SEVERE epoch harm (code e32 +1.267 bpb / 162 sigma;
e16 +0.39 bpb). The 100B transect that epochs via SIMULATED epoching
(``target_budget``) shows NO harm (kernel residual flat, <= seed floor). Those two
results differ in BOTH budget (10B vs 100B) AND mechanism (real subset-repeat vs
simulated) -- the budget-transfer of harm is CONFOUNDED with the mechanism.

This experiment REMOVES the confound: run the SAME mechanism as twobucket-a2 /
a3 / E2 (``max_train_batches`` real subset-repeat of ``c01q0`` at fixed w=0.2)
at 100B budget, so BUDGET is the only thing that changes from the 10B harm.

The a3 arm (matched ``max_train_batches`` mechanism at 2.5B/10B/40B) measured
harm shrinking with budget as ~B^-0.73 (e4->e16 harm 0.688 @2.5B, 0.384 @10B,
0.092 @40B). Two committed hypotheses are contrasted here:

  (a) a3 power law HOLDS -> harm PERSISTS at 100B but small:
      e4->e16 rise ~ 0.384 * (100/10)^-0.73 ~ 0.071 bpb;
      e4->e32 rise ~ 1.269 * (100/10)^-0.73 ~ 0.236 bpb (headline ~0.245).
  (b) transect result generalizes -> harm VANISHES at 100B even under real
      subset-repeat: rise ~ 0 (< 2*sqrt(2)*sigma = 0.016 bpb). Kernel suffices.

Single seed each (the a3-predicted e16 rise ~0.071 is ~9x the 0.008 bpb pair
seed floor -> a single-seed pair resolves it); 3 runs (e4 baseline, e16, e32).

Emits ``harm100b_preregistration.json`` (canonical under
``scratch/mixture_features/grug/`` and a byte-identical staged copy next to the
launcher so it ships in job bundles) plus its ``.sha256``.
"""

import hashlib
import json
import shutil
from pathlib import Path

# Swarm/seedpanel 100B constants (identical to launch_mve_seedpanel.py).
STEPS = 47_759
BATCH_SIZE = 512
SEQ_LEN = 4_096
PHASE_BOUNDARY_STEP = 38_144  # int(0.8*STEPS) quantized down to the 64-step mixture-block multiple
MIXTURE_BLOCK_SIZE = 32_768
EXPERIMENT_BUDGET_TOKENS = STEPS * BATCH_SIZE * SEQ_LEN  # 100_157_882_368
TOKENS_PER_BATCH = BATCH_SIZE * SEQ_LEN  # 2_097_152

W_TARGET = 0.2
SLICED_BUCKET = "c01q0"  # code (152.6B tok) -- the epoching bucket
PARTNER_BUCKET = "c05q0"  # web (876.9B tok) -- fixed 0.8, never epochs
SLICED_TOKENS = 152_613_743_170  # c01q0 (from the epochrep prereg arm)
PARTNER_TOKENS = 876_859_363_557  # c05q0 (derived exactly from epochrep partner phase0)

SEED = 1
EPOCHS = (4, 16, 32)

# a3 budget x epoch power law (twobucket axis 3, matched max_train_batches mechanism):
# harm(B) ~ harm(10B) * (B/10B)^A3_EXPONENT.
A3_EXPONENT = -0.73
# 10B realized e4->e_high humaneval rises (twobucket a2: ctr 0.9358, e16 1.3196, e32 2.2044).
BASE_RISE_10B = {16: 1.3196 - 0.9358, 32: 2.2044 - 0.9358}
BUDGET_RATIO = EXPERIMENT_BUDGET_TOKENS / 10_015_997_952  # ~10.0 (100B / 10B)

# Panel-measured humaneval seed noise floor (transect readout).
SIGMA_HUMANEVAL_BPB = 0.00565
PAIR_FLOOR = (2 ** 0.5) * SIGMA_HUMANEVAL_BPB  # sqrt(2)*sigma ~ 0.00799
FORM_REQUIRED = 2 * PAIR_FLOOR  # 2*sqrt(2)*sigma ~ 0.01598

CANONICAL = Path(__file__).resolve().parents[3] / "scratch/mixture_features/grug/harm100b_preregistration.json"
STAGED = Path(__file__).resolve().parents[2] / "grug/moe/harm100b_preregistration.json"


def _slice_batches(e: int) -> int:
    """max_train_batches count so phase0 epochs = w*boundary/n is nearest ``e``."""
    return round(W_TARGET * PHASE_BOUNDARY_STEP / e)


def _run(e: int) -> dict:
    n = _slice_batches(e)
    phase1_steps = STEPS - PHASE_BOUNDARY_STEP
    slice_tokens = n * TOKENS_PER_BATCH
    a3_rise = BASE_RISE_10B[e] * (BUDGET_RATIO ** A3_EXPONENT) if e in BASE_RISE_10B else 0.0
    return {
        "point": f"e{e}",
        "run_id": f"rav_mve_harm100b_e{e}",
        "job_name": f"rav-mve-harm100b-e{e}",
        "arm": "code",
        "sliced_bucket": SLICED_BUCKET,
        "partner_bucket": PARTNER_BUCKET,
        "w_target": W_TARGET,
        "e_target": float(e),
        "seed": SEED,
        "steps": STEPS,
        "phase_boundary_step": PHASE_BOUNDARY_STEP,
        "slice_batches": n,
        "is_control": e == EPOCHS[0],
        "epochs": {
            "convention": "real: sliced e = w * phase_steps / n_slice_batches; partner over full cache",
            "code_slice_tokens": slice_tokens,
            "code_slice_frac_of_bucket": slice_tokens / SLICED_TOKENS,
            "sliced": {
                "phase0": W_TARGET * PHASE_BOUNDARY_STEP / n,
                "phase1": W_TARGET * phase1_steps / n,
                "total": W_TARGET * STEPS / n,
            },
            "partner": {
                "phase0": (1 - W_TARGET) * PHASE_BOUNDARY_STEP * TOKENS_PER_BATCH / PARTNER_TOKENS,
                "phase1": (1 - W_TARGET) * phase1_steps * TOKENS_PER_BATCH / PARTNER_TOKENS,
                "total": (1 - W_TARGET) * STEPS * TOKENS_PER_BATCH / PARTNER_TOKENS,
            },
        },
        # The two committed hypotheses' predicted e4->this-epoch humaneval rise (bpb).
        "pred_a3_powerlaw_rise_vs_e4": round(a3_rise, 6),
        "pred_transect_vanishes_rise_vs_e4": 0.0,
    }


def build_prereg() -> dict:
    runs = [_run(e) for e in EPOCHS]
    return {
        "experiment": "harm100b_budget_transfer",
        "issue": "#7067 (mixing-via-embeddings); DECISIVE budget-transfer-of-harm test (rav directive 2026-07-19)",
        "branch": "rav/mixing-via-embeddings",
        "date_utc": "2026-07-19",
        "launcher": "experiments/grug/moe/launch_mve_harm100b_h100.py",
        "builder": "experiments/datakit/mixture_features/harm100b_design.py",
        "why": (
            "Removes the budget-vs-mechanism confound between the 10B real-subset-repeat harm "
            "(twobucket-a2/E2: code e32 +1.267 bpb / 162 sigma) and the 100B simulated-epoching "
            "no-harm (transect: kernel residual flat). Runs the SAME max_train_batches real "
            "subset-repeat mechanism as twobucket-a2/a3/E2, at 100B budget, so BUDGET is the "
            "only change from the 10B harm. Decides whether the epoch-harm term is needed at "
            "production scale -- the crux of the kernel-vs-form question."
        ),
        "constants": {
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "phase_boundary_step": PHASE_BOUNDARY_STEP,
            "mixture_block_size": MIXTURE_BLOCK_SIZE,
            "experiment_budget_tokens": EXPERIMENT_BUDGET_TOKENS,
            "target_budget_tokens": None,
            "epoching_mechanism": "max_train_batches (REAL subset-repeat; budgets OFF)",
            "seed": SEED,
            "model_d512": "swarm d512/6L/4H/1KV/64exp + gpu_fa4_cute (== seed panel / transect / twobucket / epochrep)",
            "optimizer": "SWARM_OPTIMIZER fractional schedule (warmup peak 0.100*N; asserted at launch for N=47759)",
            "hardware": "cw-rno2a H100x8 one node/run, cuda_async allocator, CW-mirror data",
            "in_training_validation": False,
            "wandb_group": "rav_mve_harm100b",
            "checkpoint_prefix": "s3://marin-us-east-02a/marin/users/rav/grug/rav_mve_harm100b_<point>/dev/",
        },
        "arm": {
            "code": {
                "sliced_bucket": SLICED_BUCKET,
                "partner_bucket": PARTNER_BUCKET,
                "w_target": W_TARGET,
                "epochs": [float(e) for e in EPOCHS],
                "control_epoch": float(EPOCHS[0]),
                "sliced_total_tokens": SLICED_TOKENS,
                "partner_total_tokens": PARTNER_TOKENS,
                "readout_task": "logprob_humaneval_10shot",
                "note": (
                    "identical buckets/weights/mechanism to twobucket-a2/a3/E2 code arm; only the "
                    "budget (47759 steps = 100B) and seed (1) differ -- so the 10B code arm is the "
                    "matched-mechanism reference"
                ),
            }
        },
        "slicing_mechanism": {
            "used": "LmDataConfig.max_train_batches={c01q0: n} -- slice-after-shuffle of the SAME cache (target_budget OFF)",
            "why": (
                "real epochs via the mixture restart-wrap; content h=V.w is UNCHANGED by slicing "
                "(the frozen content kernel is FLAT across epochs at fixed w by construction). "
                "This is the SAME mechanism as twobucket-a2/a3/E2; it differs from the transect, "
                "which epochs via target_budget (simulated). Exact token control at 2.097M "
                "granularity, zero data movement."
            ),
            "phase0_epoch_label": "e = w * phase_boundary_step / n (the label, apples-to-apples with the 10B references)",
        },
        "predictions": {
            "a3_powerlaw": {
                "hypothesis": "HARM PERSISTS at 100B (small): the a3 budget x epoch power law B^-0.73 holds.",
                "model": "rise(100B) = rise(10B) * (100B/10B)^(-0.73)",
                "base_rise_10B": {"e16": round(BASE_RISE_10B[16], 4), "e32": round(BASE_RISE_10B[32], 4)},
                "budget_ratio": round(BUDGET_RATIO, 4),
                "exponent": A3_EXPONENT,
                "pred_rise_e16_vs_e4": round(BASE_RISE_10B[16] * BUDGET_RATIO ** A3_EXPONENT, 5),
                "pred_rise_e32_vs_e4": round(BASE_RISE_10B[32] * BUDGET_RATIO ** A3_EXPONENT, 5),
                "headline": "e16 ~ 0.071 bpb, e32 ~ 0.245 bpb -> harm term has (marginal) value at production",
            },
            "transect_vanishes": {
                "hypothesis": "HARM VANISHES at 100B even under real subset-repeat: the 10B harm is a small-budget artifact.",
                "pred_rise_e16_vs_e4": 0.0,
                "pred_rise_e32_vs_e4": 0.0,
                "bound": f"|rise| < 2*sqrt(2)*sigma = {round(FORM_REQUIRED, 5)} bpb",
                "headline": "kernel SUFFICES at production; the epoch-harm term is NOT needed",
            },
        },
        "seed_floor": {
            "sigma_humaneval_bpb": SIGMA_HUMANEVAL_BPB,
            "pair_sqrt2_sigma": round(PAIR_FLOOR, 5),
            "form_required_2sqrt2_sigma": round(FORM_REQUIRED, 5),
            "source": "10-seed panel rav_mve_seedpanel_h100 (humaneval bpb), via transect readout",
        },
        "decision_rule": {
            "estimand": "realized e4->e16 humaneval bpb rise (single seed per point; SE of the pair = sqrt(2)*sigma = 0.008 bpb, so 2*SE = 0.016 bpb).",
            "HARM_PERSISTS": (
                "rise > 0.016 (2*sqrt(2)*sigma) AND ~0.07 -> the a3 power law holds; epoch harm "
                "PERSISTS at 100B under matched mechanism -> the harm term has (marginal) value at production."
            ),
            "HARM_VANISHES": (
                "rise < 0.016 (~0, within the form-required floor) -> harm VANISHES at 100B even under "
                "real subset-repeat -> KERNEL SUFFICES at production scale, the harm term is NOT needed "
                "(the 10B harm is a proxy artifact of small budget)."
            ),
            "amplifier": "code-e32 is the amplifier: predicted +0.245 bpb if the a3 power law holds, ~0 if harm vanishes.",
        },
        "readout_plan": {
            "eval": "post-hoc eval_logprob (same frozen harness as seedpanel/twobucket/epochrep/transect) on the 3 final checkpoints (step 47758); humaneval bpb.",
            "aggregate": "realized humaneval bpb per epoch; e4->e16 and e4->e32 rises.",
            "test": "rise vs the two committed predictions (a3 0.071/0.245 vs transect ~0) and vs the 0.008 pair floor / 0.016 form-required floor -> verdict.",
            "figure": "f29_harm_budget_transfer.png: e4->e16 (and e4->e32) rise at 10B vs 100B under the SAME max_train_batches mechanism, with the a3 power-law curve and the transect-vanishes line, and the 2*sqrt(2)*sigma band.",
            "companions": "10B matched-mechanism reference = twobucket-a2 / E2 code arm (same buckets/weights/mechanism); the transect is the 100B simulated-epoching companion.",
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

    # sanity: the two copies are byte-identical and hash matches
    assert STAGED.read_bytes() == raw, "staged copy is not byte-identical"
    print(f"wrote {CANONICAL}")
    print(f"wrote {STAGED}")
    print(f"sha256 {digest}")
    for r in prereg["runs"]:
        e = r["epochs"]["sliced"]
        print(
            f"  {r['point']:>4}  n={r['slice_batches']:<5} "
            f"phase0_e={e['phase0']:.4f} total_e={e['total']:.4f} "
            f"slice_tok={r['epochs']['code_slice_tokens']:,} "
            f"a3_rise={r['pred_a3_powerlaw_rise_vs_e4']}"
        )


if __name__ == "__main__":
    main()

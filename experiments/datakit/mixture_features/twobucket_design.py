# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-bucket factorial design + pre-registration (mixing-via-embeddings, #7067).

Builds the 25-run two-bucket experiment around code=c01q0 and web=c05q0 (the
largest web_text bucket by total_tokens) and writes
``scratch/mixture_features/grug/twobucket_preregistration.json``:

- NATURAL arm (8 runs, #2846-replica): w_code in {0, .05, .10, .20, .35, .50,
  .75, 1.0}, both phases, simulated epoching ON (target budget unchanged) so
  per-token repetition matches the swarm; epochs scale with w.
- FACTORIAL arms (17 runs): simulated epoching OFF (budgets None); the code
  stream is sub-sliced via ``max_train_batches['c01q0']`` (slice-after-shuffle,
  the same mechanism the swarm's simulated epoching used; the block shuffle
  permutes io-blocks globally so a prefix slice is a content-fair subsample at
  256-sequence granularity). Epochs are REAL and exact:
  e_code(phase0) = w * boundary_steps / n_batches.
  - axis 1 (weight at fixed e~4): w in {.05, .10, .20, .35, .50}
  - axis 2 (epochs at fixed w=.2): e in {1, 2, 4, 8, 16, 32}
  - axis 3 (budget x epochs): B in {2.5B, 40B} x e in {4, 16} at w=.2
  - axis 4 (d256 x epochs): e in {2, 8, 32} at w=.2, B=10B
  (center w=.2/e~4/10B/d512 shared by axes 1+2).

Predictions per run, committed before launch: (i) frozen Hellinger-kernel
(K=1000, dual on the 800 train runs), (ii) kernel + fitted swoosh (the
transect's committed b_g1 params), and, for the natural arm, (iii) the
#2846-imported harm — all validated by reproducing the transect
pre-registration's own committed numbers to 1e-9 before use.

Run from this directory: ``python twobucket_design.py``.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402

import grug_fit as gf  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import swoosh_form as sf  # noqa: E402
from grug_validation_batch2 import apply_corrected_phase_constants  # noqa: E402
from grug_validation_batch3 import cluster_delta_groups  # noqa: E402
from grug_validation_checks import build_zmacro_target  # noqa: E402
from retrodiction import _sq_hellinger  # noqa: E402

logger = logging.getLogger("twobucket_design")

GRUG_DIR = gf.GRUG_DIR
OUT_PATH = GRUG_DIR / "twobucket_preregistration.json"
TRANSECT_PREREG = GRUG_DIR / "transect_preregistration.json"

CODE_BUCKET = "c01q0"
WEB_BUCKET = "c05q0"  # largest web_text-group bucket by total_tokens (876.86B)

BATCH_SIZE = 512
SEQ_LEN = 4096
TOKENS_PER_BATCH = BATCH_SIZE * SEQ_LEN  # 2_097_152
MIXTURE_BLOCK_SIZE = 32_768
TARGET_BUDGET_TOKENS = 10_372_343_704_053

NATURAL_W_GRID = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
AXIS1_W = (0.05, 0.10, 0.35, 0.50)  # + center w=0.20
AXIS2_E = (1.0, 2.0, 8.0, 16.0, 32.0)  # + center e~4
AXIS3 = (("b2p5", 1_194, 4.0), ("b2p5", 1_194, 16.0), ("b40", 19_104, 4.0), ("b40", 19_104, 16.0))
AXIS4_E = (2.0, 8.0, 32.0)
CENTER_W = 0.20
CENTER_E = 4.0
STEPS_10B = 4_776
SEED = 0

WANDB_GROUP = "rav_mve_twobucket"
RUN_ID_PREFIX = "rav_mve_twobucket"


def boundary_step(total_steps: int) -> int:
    """Same block-quantized 0.8-fraction boundary rule as every launcher in this program."""
    step_multiple = MIXTURE_BLOCK_SIZE // math.gcd(MIXTURE_BLOCK_SIZE, BATCH_SIZE)
    requested = max(1, int(total_steps * 0.8))
    return max(step_multiple, (requested // step_multiple) * step_multiple)


def natural_run(w_code: float) -> dict:
    return {
        "point": f"w{int(round(w_code * 100)):03d}",
        "arm": "natural",
        "w_code": w_code,
        "steps": STEPS_10B,
        "model": "d512",
        "simulated_epoching": True,
        "code_slice_batches": None,
    }


def factorial_run(point: str, arm: str, w_code: float, e_target: float, steps: int, model: str) -> dict:
    boundary = boundary_step(steps)
    n = max(1, round(w_code * boundary / e_target))
    return {
        "point": point,
        "arm": arm,
        "w_code": w_code,
        "steps": steps,
        "model": model,
        "simulated_epoching": False,
        "code_slice_batches": n,
        "e_code_phase0_target": e_target,
    }


def build_grid() -> list[dict]:
    runs = [natural_run(w) for w in NATURAL_W_GRID]
    runs.append(factorial_run("ctr", "center", CENTER_W, CENTER_E, STEPS_10B, "d512"))
    for w in AXIS1_W:
        runs.append(factorial_run(f"a1_w{int(round(w * 100)):03d}", "axis1_weight", w, CENTER_E, STEPS_10B, "d512"))
    for e in AXIS2_E:
        runs.append(factorial_run(f"a2_e{int(e)}", "axis2_epochs", CENTER_W, e, STEPS_10B, "d512"))
    for tag, steps, e in AXIS3:
        runs.append(factorial_run(f"a3_{tag}_e{int(e)}", "axis3_budget", CENTER_W, e, steps, "d512"))
    for e in AXIS4_E:
        runs.append(factorial_run(f"a4_d256_e{int(e)}", "axis4_size", CENTER_W, e, STEPS_10B, "d256"))
    for r in runs:
        r["run_id"] = f"{RUN_ID_PREFIX}_{r['point']}"
        r["seed"] = SEED
    return runs


def annotate_run(r: dict, t_code: float, t_web: float) -> None:
    """Exact per-bucket per-phase epoch bookkeeping (both conventions where applicable)."""
    steps = r["steps"]
    boundary = boundary_step(steps)
    b_exp = steps * TOKENS_PER_BATCH
    f0, f1 = boundary / steps, 1 - boundary / steps
    w, n = r["w_code"], r["code_slice_batches"]
    r.update(
        {
            "phase_boundary_step": boundary,
            "phase_fractions": [f0, f1],
            "experiment_budget_tokens": b_exp,
            "target_budget_tokens": TARGET_BUDGET_TOKENS if r["simulated_epoching"] else None,
        }
    )
    if r["simulated_epoching"]:
        # simulated epochs of the sliced caches = w * f_p * TARGET / T_j (project convention)
        r["epochs"] = {
            "code": {
                "phase0": w * f0 * TARGET_BUDGET_TOKENS / t_code,
                "phase1": w * f1 * TARGET_BUDGET_TOKENS / t_code,
                "total": w * TARGET_BUDGET_TOKENS / t_code,
            },
            "web": {
                "phase0": (1 - w) * f0 * TARGET_BUDGET_TOKENS / t_web,
                "phase1": (1 - w) * f1 * TARGET_BUDGET_TOKENS / t_web,
                "total": (1 - w) * TARGET_BUDGET_TOKENS / t_web,
            },
            "convention": "simulated: e = w * f_p * target_budget / T_j (short-run realized f_p)",
        }
        # swarm-f version for the featurization/prediction convention
        r["epochs_swarm_f"] = {
            "code_phase0": w * (38_144 / 47_759) * TARGET_BUDGET_TOKENS / t_code,
            "web_phase0": (1 - w) * (38_144 / 47_759) * TARGET_BUDGET_TOKENS / t_web,
        }
    else:
        t_code_slice = n * TOKENS_PER_BATCH
        r["code_slice_tokens"] = t_code_slice
        r["code_slice_frac_of_bucket"] = t_code_slice / t_code
        r["epochs"] = {
            "code": {
                "phase0": w * boundary / n,
                "phase1": w * (steps - boundary) / n,
                "total": w * steps / n,
            },
            "web": {
                "phase0": (1 - w) * f0 * b_exp / t_web,
                "phase1": (1 - w) * f1 * b_exp / t_web,
                "total": (1 - w) * b_exp / t_web,
            },
            "convention": "real: code e = w * phase_batches / n_slice_batches; web over full 876.86B cache",
        }


def parse_swoosh_params(models_blob: str) -> tuple[dict, dict]:
    """The transect prereg's committed b_g1 params (zmacro + humaneval), verbatim."""
    z_str = models_blob.split("params ", 1)[1].split("; humaneval params")[0]
    h_str = models_blob.split("humaneval params ", 1)[1]
    z, h = json.loads(z_str), json.loads(h_str)
    for p in (z, h):
        p["epoch_mode"] = "phase"
    return z, h


def two_bucket_w(w_code: float, idx: dict[str, int], n_buckets: int) -> np.ndarray:
    w = np.zeros((2, n_buckets))
    for p in range(2):
        w[p, idx[CODE_BUCKET]] = w_code
        w[p, idx[WEB_BUCKET]] = 1.0 - w_code
    return w


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    apply_corrected_phase_constants()
    f_swarm = sf.phase_fractions()
    assert abs(f_swarm[0] - 0.7987) < 5e-4, f_swarm

    hists, views, _c, _r, _o, buckets_table = gf.load_grug_artifacts()
    buckets = [h.domain for h in hists]
    import featurize

    v1000, order = featurize.composition_matrix(hists, k=1000, views=views)
    assert order == buckets
    v1000 = np.asarray(v1000)
    tj = buckets_table.set_index("bucket").loc[buckets, "total_tokens"].to_numpy(float)
    idx = {b: j for j, b in enumerate(buckets)}
    t_code, t_web = float(tj[idx[CODE_BUCKET]]), float(tj[idx[WEB_BUCKET]])
    assert t_code == 152_613_743_170.0, t_code
    assert t_web == 876_859_363_557.0, t_web

    # web-bucket pick documentation: largest web_text bucket by total_tokens
    masks, group_doc = cluster_delta_groups(buckets, buckets_table, v1000)
    masks["all"] = np.ones(len(buckets))
    group_of = {}
    for cl_name, info in group_doc.items():
        group_of[cl_name] = info["group"]
    bt = buckets_table.set_index("bucket")
    web_rank = []
    for b in buckets:
        cl = int(bt.loc[b, "cluster_id"])
        cl_name = "tail" if cl == -1 else f"c{cl:02d}"
        if group_of[cl_name] == "web_text":
            web_rank.append((b, float(bt.loc[b, "total_tokens"])))
    web_rank.sort(key=lambda x: -x[1])
    assert web_rank[0][0] == WEB_BUCKET, web_rank[:3]

    runs_df = pd.read_parquet(gf.TRAIN_RUNS)
    w_train = gf.weight_matrix(runs_df, buckets)
    rec = json.loads((GRUG_DIR / "target_candidates.json").read_text())["recommended_target"]
    y = build_zmacro_target(runs_df, rec)
    y_hum = np.array([json.loads(ev)[sf.HUMANEVAL_TASK]["bpb"] for ev in runs_df["evals"]], dtype=np.float64)

    hphase = gf.per_phase_hist(w_train, v1000)
    d2 = _sq_hellinger(hphase)
    frozen = json.loads(sf.FROZEN_HYPERS.read_text())["models"]["4_hellinger_kernel_k1000"]
    gamma, alpha = float(frozen["gamma"]), float(frozen["alpha"])

    transect = json.loads(TRANSECT_PREREG.read_text())
    z_params, hum_params = parse_swoosh_params(transect["models"]["ii_swoosh"])
    assert z_params["taus"]["all"] == 5.5 and abs(z_params["b"]["all"] - 0.0029024) < 1e-6, z_params

    duals = {}
    for tag, yy in (("zmacro", y), ("humaneval", y_hum)):
        duals[tag] = sf.kernel_dual(d2, yy, gamma, alpha)

    def kernel_pred(w_mix: np.ndarray, tag: str) -> np.ndarray:
        h_mix = np.stack([w_mix[:, p, :] @ v1000.T for p in range(2)], axis=1)
        d2_mix = sf.candidate_d2(h_mix, hphase)
        dual, ym = duals[tag]
        return np.exp(-gamma * d2_mix) @ dual + ym

    def nearest_train(w_mix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h_mix = np.stack([w_mix[:, p, :] @ v1000.T for p in range(2)], axis=1)
        d2_mix = sf.candidate_d2(h_mix, hphase)
        return np.sqrt(d2_mix.min(axis=1)), d2_mix.argmin(axis=1)

    # ---- machinery validation: reproduce the transect prereg's committed predictions ----
    tr_runs = transect["runs"]
    w_tr = np.zeros((len(tr_runs), 2, len(buckets)))
    anchor = sf.load_anchor(buckets)
    for m, run in enumerate(tr_runs):
        p0 = transect["mixtures"][run["run_name"]]["phase0"]
        for b, v in p0.items():
            w_tr[m, 0, idx[b]] = v
        w_tr[m, 1, :] = anchor[1]
    ep_tr = sf.per_phase_epochs(w_tr, tj)
    for tag, params, kcol, scol in (
        ("zmacro", z_params, "pred_kernel", "pred_swoosh"),
        ("humaneval", hum_params, "pred_kernel_humaneval", "pred_swoosh_humaneval"),
    ):
        pk = kernel_pred(w_tr, tag)
        ps = pk + sf.predict_r(params, w_tr, ep_tr, masks)
        for m, run in enumerate(tr_runs):
            assert abs(pk[m] - run[kcol]) < 1e-9, (run["run_name"], tag, pk[m], run[kcol])
            assert abs(ps[m] - run[scol]) < 1e-9, (run["run_name"], tag, ps[m], run[scol])
    logger.info("transect prediction reproduction: PASSED (16 kernel + 16 swoosh values to 1e-9)")

    # ---- the grid ----
    grid = build_grid()
    for r in grid:
        annotate_run(r, t_code, t_web)
    assert len(grid) == 25 and len({r["run_id"] for r in grid}) == 25

    w_all = np.stack([two_bucket_w(r["w_code"], idx, len(buckets)) for r in grid])
    hell, near = nearest_train(w_all)

    # epochs for the swoosh features:
    #  natural arm -> the featurization convention (swarm f, target budget), exactly as the
    #  models were fit; factorial arms -> the runs' REAL epochs.
    ep_all = sf.per_phase_epochs(w_all, tj)  # swarm-f target-budget epochs, all runs
    for i, r in enumerate(grid):
        if not r["simulated_epoching"]:
            e = np.zeros((2, len(buckets)))
            e[0, idx[CODE_BUCKET]] = r["epochs"]["code"]["phase0"]
            e[1, idx[CODE_BUCKET]] = r["epochs"]["code"]["phase1"]
            e[0, idx[WEB_BUCKET]] = r["epochs"]["web"]["phase0"]
            e[1, idx[WEB_BUCKET]] = r["epochs"]["web"]["phase1"]
            ep_all[i] = e

    imp = sf.fit_2846_curvature()
    mu, sd = rec["train_z_mu"], rec["train_z_sd"]
    zfactor = float(np.mean([mu[t] / sd[t] for t in rec["task_list"]]))
    b_units = {"zmacro": imp["b_frac_per_epoch2_per_unit_mass"] * zfactor}
    b_units["humaneval"] = imp["b_frac_per_epoch2_per_unit_mass"] * float(y_hum.mean())
    anchor_w = anchor[None, :, :]
    anchor_ep = sf.per_phase_epochs(anchor_w, tj)

    for tag, params in (("zmacro", z_params), ("humaneval", hum_params)):
        pk = kernel_pred(w_all, tag)
        r_fit = sf.predict_r(params, w_all, ep_all, masks)
        r_imp = sf.r_2846(w_all, ep_all, b_units[tag]) - float(sf.r_2846(anchor_w, anchor_ep, b_units[tag]))
        for i, r in enumerate(grid):
            r[f"pred_kernel_{tag}"] = float(pk[i])
            r[f"pred_swoosh_{tag}"] = float(pk[i] + r_fit[i])
            if r["arm"] == "natural":
                r[f"pred_2846_{tag}"] = float(pk[i] + r_imp[i])
    for i, r in enumerate(grid):
        r["nearest_train_hellinger"] = float(hell[i])
        r["nearest_train_experiment_index"] = int(runs_df.iloc[int(near[i])]["experiment_index"])

    train_hell_min = float(np.sqrt(np.min(d2 + np.eye(len(y)) * 10, axis=1)).mean())

    # cost estimate (H100x8 node-hours; measured panel range 1.2-2.3 s/step, use 1.7)
    per_step = 1.7
    hours = {r["run_id"]: r["steps"] * per_step / 3600 for r in grid}
    d256_scale = 0.7  # d256 is loader-bound at worst; step time bounded by data path
    for r in grid:
        if r["model"] == "d256":
            hours[r["run_id"]] *= d256_scale
    total_node_hours = sum(hours.values())

    prereg = {
        "date_utc": "2026-07-18",
        "experiment": "twobucket_factorial",
        "issue": (
            "#7067 (mixing-via-embeddings); design = #2846 transplant + decoupling factorial (rav directive 2026-07-18)"
        ),
        "branch": "rav/mixing-via-embeddings",
        "launcher": "experiments/grug/moe/launch_mve_twobucket_h100.py",
        "buckets": {
            "code": {
                "bucket": CODE_BUCKET,
                "group": "code_adjacent",
                "total_tokens": int(t_code),
                "rationale": "the characterized code bucket (transect contrast bucket; humaneval readout)",
            },
            "web": {
                "bucket": WEB_BUCKET,
                "group": "web_text",
                "total_tokens": int(t_web),
                "rationale": "largest web_text-group bucket by total_tokens",
                "runner_ups": [{"bucket": b, "total_tokens": int(t)} for b, t in web_rank[1:4]],
            },
        },
        "constants": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "mixture_block_size": MIXTURE_BLOCK_SIZE,
            "target_budget_tokens_when_simulated": TARGET_BUDGET_TOKENS,
            "model_d512": "swarm d512/6L/4H/1KV/64exp + gpu_fa4_cute (== seed panel/transect)",
            "model_d256": (
                "MoeHeuristic().build_model_config(256, seq_len=4096) + the same 4 swarm-family replacements "
                "(num_experts=64, sliding_window=4096, router_z_loss_coef=0.001, disable_long_rope=False) that "
                "reproduce the d512 swarm model exactly from build_model_config(512); d256 = 3L/2H/1KV/interm128/"
                "shared256/head_dim128; optimizer kept at SWARM_OPTIMIZER (LR not re-tuned; heuristic dim-exponent "
                "would raise adam_lr x1.11 - documented, accepted)"
            ),
            "optimizer": (
                "GrugMoeAdamHConfig swarm constants, fractional schedule (warmup 0.1, linear); "
                "verified peak at 0.100*N for N in {1194, 4776, 19104, 47759}"
            ),
            "seed": SEED,
            "hardware": "cw-rno2a H100x8 one node/run, cuda_async allocator, CW-mirror data",
            "wandb_group": WANDB_GROUP,
            "checkpoint_prefix": "s3://marin-us-east-02a/marin/users/rav/grug/rav_mve_twobucket_<point>/dev/",
        },
        "slicing_mechanism": {
            "prescribed": "server-side sub-cache copies of 1/m of c01q0's part-* shard dirs",
            "used": "LmDataConfig.max_train_batches={'c01q0': n} - slice-after-shuffle of the SAME cache",
            "why": (
                "identical estimand (T_code' = n*512*4096 tokens, real epochs via mixture restart-wrap), but "
                "exact token control at 2.097M granularity (vs ~25M/shard), zero data movement, and the SAME "
                "subset-selection mechanism as the swarm's simulated epoching (slice of the block-shuffled "
                "stream; the block shuffle permutes io-blocks globally, so the slice is a content-fair "
                "subsample at 256-sequence granularity). Shard-dir copies would add store-order content bias "
                "and coarser quantization. Ledger verification replaced by exactness: T' is n*512*4096 by "
                "construction; the full-cache ledger total (152,613,743,170 input_ids) matches buckets_table."
            ),
            "nested": (
                "fixed seed 0 + identical component set => identical shuffle permutation across runs; "
                "slices are nested prefixes (smaller T' subset of larger T')."
            ),
        },
        "epoching_note": (
            "NATURAL arm: simulated epoching ON (target 10.372e12) - per-token repetition identical "
            "to the 100B swarm; BOTH endpoints heavily epoched (w=1: code ~53.7 phase-0 epochs; w=0: web ~9.4), "
            "inherent to the two-bucket design (as in #2846). FACTORIAL arms: budgets None, epochs real; web never "
            "epochs (<=0.05 total everywhere)."
        ),
        "predictions_note": {
            "kernel": (
                "frozen Hellinger kernel ridge K=1000 (gamma 1.2475417504102333, alpha 0.1), dual on all "
                "800 train runs; predictions validated by reproducing the transect prereg's 16+16 committed "
                "values to 1e-9"
            ),
            "swoosh": (
                "kernel + transect-committed b_g1 head: zmacro tau=5.5 b=0.002902417147546668; humaneval "
                "tau=2.0 b=5.872124766140496e-05; per-phase epochs (natural arm: featurization convention "
                "swarm-f/target-budget; factorial arms: the runs' REAL epochs)"
            ),
            "sharpest_point": (
                "AXIS 2 IS A PURE HARM-TERM TEST: sub-slicing leaves the content histogram h "
                "unchanged, so the kernel predicts NO change along the epochs axis (constant "
                "pred_kernel); the swoosh form predicts the harm curve directly. Any realized slope "
                "along axis 2 is invisible to content-only surrogates by construction."
            ),
            "axis3_axis4_caveat": (
                "kernel+swoosh were fit at 100B/d512; levels do not transport across budget/size, "
                "so axis-3/4 predictions are recorded for SHAPE only. The swoosh form has no budget or size input: it "
                "predicts harm at fixed (w, e) is budget- and size-invariant - axes 3/4 test exactly that."
            ),
            "ood_note": (
                "all two-bucket mixtures are far off the swarm's support (nearest-train Hellinger ~0.3-0.6 "
                "vs the train-train nearest-neighbor mean below); the kernel is expected to mean-revert - the natural "
                "arm is an honest OOD test of shape, not level."
            ),
            "train_train_nearest_hellinger_mean": train_hell_min,
        },
        "primary_readout": {
            "natural_arm": (
                "SHAPE of realized humaneval bpb vs w_code (expect swoosh: improve then degrade with a "
                "minimum at moderate w) and realized zmacro/web-task bpb vs w_code (expect monotone worsening or "
                "early-flat). Overlay the three committed prediction sets."
            ),
            "axis2": (
                "realized humaneval + zmacro vs e_code at fixed w=0.2 - the decoupled repetition axis; "
                "compare against constant-kernel and swoosh-harm predictions"
            ),
            "axis1": "realized metrics vs w_code at fixed e~4 - the never-before-measured pure content-share axis",
            "axis3": "does the harm at fixed (w, e) shift with training budget (2.5B vs 10B vs 40B)",
            "axis4": "does the harm at fixed (w, e) grow with model capacity (d256 vs d512)",
            "conversion": "perplexity per byte = 2^bpb (reported alongside bpb)",
            "eval": (
                "post-hoc eval_logprob (60-task set) on each final checkpoint; zmacro_english_20 with the "
                "frozen train stats in frozen_model_hyperparams.json"
            ),
        },
        "figures_planned": {
            "f22": (
                "natural sweep replica: realized humaneval/zmacro bpb vs w_code + prediction overlays + "
                "per-point epoch annotations"
            ),
            "f23": (
                "THE decoupling figure: weight-at-fixed-epochs (axis 1) vs epochs-at-fixed-weight (axis 2) "
                "side by side"
            ),
            "f24": "budget x epochs and size x epochs interactions (axes 3-4)",
        },
        "cost": {
            "n_runs": 25,
            "runs_by_arm": {"natural": 8, "center": 1, "axis1": 4, "axis2": 5, "axis3": 4, "axis4": 3},
            "assumed_s_per_step_d512": per_step,
            "d256_step_scale": d256_scale,
            "est_total_h100x8_node_hours": round(total_node_hours, 1),
            "panel_run_equivalents": round(total_node_hours / (47_759 * per_step / 3600), 2),
        },
        "runs": grid,
    }

    OUT_PATH.write_text(json.dumps(prereg, indent=1, sort_keys=True) + "\n")
    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    logger.info("wrote %s", OUT_PATH)
    print(f"sha256 {sha}")
    for r in grid:
        e0c = r["epochs"]["code"]["phase0"]
        e0w = r["epochs"]["web"]["phase0"]
        print(
            f"{r['run_id']:>28} {r['arm']:>12} w={r['w_code']:.2f} steps={r['steps']:>5} "
            f"n={r['code_slice_batches']!s:>4} e0_code={e0c:7.2f} e0_web={e0w:5.2f} "
            f"hell={r['nearest_train_hellinger']:.3f} "
            f"pk_z={r['pred_kernel_zmacro']:+.3f} ps_z={r['pred_swoosh_zmacro']:+.3f} "
            f"pk_h={r['pred_kernel_humaneval']:.3f} ps_h={r['pred_swoosh_humaneval']:.3f}"
        )


if __name__ == "__main__":
    main()

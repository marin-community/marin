# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seed-replicated epoch-harm design + pre-registration (mixing-via-embeddings, #7067).

The DECISIVE kernel-vs-functional-form experiment. Two pure-epoch arms, each a
FIXED-weight two-bucket mixture whose content histogram ``h = V.w`` is CONSTANT
across epochs (so the frozen content kernel predicts a FLAT line by
construction); real epochs are produced by sub-slicing the epoching bucket's
cache via ``max_train_batches`` (slice-after-shuffle, the exact mechanism used
by the two-bucket factorial's axis 2). Three fresh seeds per point isolate a
~0.02 bpb repetition rise from the sigma~0.006 bpb seed floor.

- code arm: c01q0 held at w=0.2 both phases (partner c05q0 at 0.8, never
  epochs), sliced to e in {4, 16, 32}. This is a BYTE-FOR-BYTE replica of the
  two-bucket factorial points ctr / a2_e16 / a2_e32 (same buckets, weights,
  slice batch counts, steps, boundary) at seeds != 0, so the factorial's seed-0
  runs there pool as a 4th seed. e=4 is the below-threshold CONTROL.
- web arm: c26q1 held at w=0.2 both phases (partner c05q0 at 0.8, never
  epochs), sliced to e in {4, 16, 24}. c26q1 is the transect's web bucket
  (lowest cone-residual novelty web_text bucket in [95B,250B]); here it is
  pinned at w=0.2 and sub-sliced for REAL epochs instead of transect's
  simulated-epoch share-setting. e=4 is the below-threshold CONTROL.

Predictions committed before launch (per point): (i) the frozen Hellinger
kernel (K=1000, gamma 1.2475..., alpha 0.1, dual on the 800 train runs) --
CONSTANT across epochs at fixed w; (ii) kernel + the transect-committed swoosh
harm head (humaneval tau=2.0, b=5.872e-5; zmacro tau=5.5, b=2.902e-3), which
RISES with epochs. Machinery is validated two ways before use: it reproduces
the transect prereg's 16+16 committed values to 1e-9, AND it reproduces the
two-bucket prereg's ctr / a2_e16 / a2_e32 humaneval kernel+swoosh values to
1e-9 (proving the code arm's predictions are identical to the pooled factorial
points).

Run from this directory: ``python epochrep_design.py``.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402

import featurize  # noqa: E402
import grug_fit as gf  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import swoosh_form as sf  # noqa: E402
import twobucket_design as tbd  # noqa: E402  (reuse boundary_step + parse_swoosh_params, verbatim)
from grug_validation_batch2 import apply_corrected_phase_constants  # noqa: E402
from grug_validation_batch3 import cluster_delta_groups  # noqa: E402
from grug_validation_checks import build_zmacro_target  # noqa: E402
from retrodiction import _sq_hellinger  # noqa: E402

logger = logging.getLogger("epochrep_design")

GRUG_DIR = gf.GRUG_DIR
OUT_PATH = GRUG_DIR / "epochrep_preregistration.json"
TRANSECT_PREREG = GRUG_DIR / "transect_preregistration.json"
TWOBUCKET_PREREG = GRUG_DIR / "twobucket_preregistration.json"

CODE_BUCKET = "c01q0"
WEB_BUCKET = "c26q1"  # transect web bucket (lowest cone-residual novelty web_text in [95B,250B])
PARTNER_BUCKET = "c05q0"  # the 0.8 non-epoching filler (largest web_text bucket, as in the factorial)

BATCH_SIZE = 512
SEQ_LEN = 4096
TOKENS_PER_BATCH = BATCH_SIZE * SEQ_LEN  # 2_097_152
MIXTURE_BLOCK_SIZE = 32_768
STEPS_10B = 4_776
W_TARGET = 0.20
SEEDS = (1, 2, 3)

# per-arm epoch ladders; e=4 is the below-threshold control in both.
ARMS = {
    "code": {"sliced": CODE_BUCKET, "group": "code_adjacent", "epochs": (4.0, 16.0, 32.0)},
    "web": {"sliced": WEB_BUCKET, "group": "web_text", "epochs": (4.0, 16.0, 24.0)},
}
CONTROL_EPOCH = 4.0

WANDB_GROUP = "rav_mve_epochrep"
RUN_ID_PREFIX = "rav_mve_epochrep"

# twobucket factorial points whose (buckets, w, slice-n, steps) the code arm replicates 1:1.
TWOBUCKET_CODE_TWINS = {4.0: "ctr", 16.0: "a2_e16", 32.0: "a2_e32"}


def slice_batches(w: float, e_target: float, steps: int) -> int:
    return max(1, round(w * tbd.boundary_step(steps) / e_target))


def real_epochs(w: float, n: int, steps: int, t_sliced: float, t_partner: float) -> dict:
    """Exact per-bucket per-phase epoch bookkeeping for a fixed-weight sliced two-bucket run."""
    boundary = tbd.boundary_step(steps)
    b_exp = steps * TOKENS_PER_BATCH
    f0, f1 = boundary / steps, 1 - boundary / steps
    return {
        "sliced": {
            "phase0": w * boundary / n,
            "phase1": w * (steps - boundary) / n,
            "total": w * steps / n,
        },
        "partner": {
            "phase0": (1 - w) * f0 * b_exp / t_partner,
            "phase1": (1 - w) * f1 * b_exp / t_partner,
            "total": (1 - w) * b_exp / t_partner,
        },
        "convention": "real: sliced e = w * phase_batches / n_slice_batches; partner over full cache",
        "code_slice_tokens": n * TOKENS_PER_BATCH,
        "code_slice_frac_of_bucket": n * TOKENS_PER_BATCH / t_sliced,
    }


def build_points(t_tok: dict[str, float]) -> list[dict]:
    """The 6 unique (arm, epoch) points; predictions are seed-independent so computed once here."""
    points = []
    for arm, spec in ARMS.items():
        sliced = spec["sliced"]
        for e in spec["epochs"]:
            n = slice_batches(W_TARGET, e, STEPS_10B)
            ep = real_epochs(W_TARGET, n, STEPS_10B, t_tok[sliced], t_tok[PARTNER_BUCKET])
            points.append(
                {
                    "arm": arm,
                    "sliced_bucket": sliced,
                    "partner_bucket": PARTNER_BUCKET,
                    "group": spec["group"],
                    "w_target": W_TARGET,
                    "e_target": e,
                    "is_control": e == CONTROL_EPOCH,
                    "steps": STEPS_10B,
                    "phase_boundary_step": tbd.boundary_step(STEPS_10B),
                    "slice_batches": n,
                    "epochs": ep,
                    "point": f"{arm}_e{int(e)}",
                    "twobucket_twin": TWOBUCKET_CODE_TWINS.get(e) if arm == "code" else None,
                }
            )
    return points


def two_bucket_w(sliced: str, w: float, idx: dict[str, int], n_buckets: int) -> np.ndarray:
    arr = np.zeros((2, n_buckets))
    for p in range(2):
        arr[p, idx[sliced]] = w
        arr[p, idx[PARTNER_BUCKET]] = 1.0 - w
    return arr


def real_ep_matrix(point: dict, idx: dict[str, int], n_buckets: int) -> np.ndarray:
    ep = np.zeros((2, n_buckets))
    s, part = point["sliced_bucket"], point["partner_bucket"]
    ep[0, idx[s]] = point["epochs"]["sliced"]["phase0"]
    ep[1, idx[s]] = point["epochs"]["sliced"]["phase1"]
    ep[0, idx[part]] = point["epochs"]["partner"]["phase0"]
    ep[1, idx[part]] = point["epochs"]["partner"]["phase1"]
    return ep


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    apply_corrected_phase_constants()
    f_swarm = sf.phase_fractions()
    assert abs(f_swarm[0] - 0.7987) < 5e-4, f_swarm

    hists, views, _c, _r, _o, buckets_table = gf.load_grug_artifacts()
    buckets = [h.domain for h in hists]
    v1000, order = featurize.composition_matrix(hists, k=1000, views=views)
    assert order == buckets
    v1000 = np.asarray(v1000)
    tj = buckets_table.set_index("bucket").loc[buckets, "total_tokens"].to_numpy(float)
    idx = {b: j for j, b in enumerate(buckets)}
    for b in (CODE_BUCKET, WEB_BUCKET, PARTNER_BUCKET):
        assert b in idx, f"missing bucket {b}"
    t_tok = {b: float(tj[idx[b]]) for b in (CODE_BUCKET, WEB_BUCKET, PARTNER_BUCKET)}
    assert t_tok[CODE_BUCKET] == 152_613_743_170.0, t_tok[CODE_BUCKET]
    assert t_tok[PARTNER_BUCKET] == 876_859_363_557.0, t_tok[PARTNER_BUCKET]
    assert abs(t_tok[WEB_BUCKET] - 111_826_691_448.0) < 1.0, t_tok[WEB_BUCKET]

    masks, _group_doc = cluster_delta_groups(buckets, buckets_table, v1000)
    masks["all"] = np.ones(len(buckets))

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
    z_params, hum_params = tbd.parse_swoosh_params(transect["models"]["ii_swoosh"])
    assert z_params["taus"]["all"] == 5.5 and abs(z_params["b"]["all"] - 0.0029024) < 1e-6, z_params
    assert hum_params["taus"]["all"] == 2.0 and abs(hum_params["b"]["all"] - 5.872124766140496e-05) < 1e-12

    duals = {tag: sf.kernel_dual(d2, yy, gamma, alpha) for tag, yy in (("zmacro", y), ("humaneval", y_hum))}

    def kernel_pred(w_mix: np.ndarray, tag: str) -> np.ndarray:
        h_mix = np.stack([w_mix[:, p, :] @ v1000.T for p in range(2)], axis=1)
        d2_mix = sf.candidate_d2(h_mix, hphase)
        dual, ym = duals[tag]
        return np.exp(-gamma * d2_mix) @ dual + ym

    def nearest_train_hellinger(w_mix: np.ndarray) -> np.ndarray:
        h_mix = np.stack([w_mix[:, p, :] @ v1000.T for p in range(2)], axis=1)
        return np.sqrt(sf.candidate_d2(h_mix, hphase).min(axis=1))

    # ---- machinery validation #1: reproduce the transect prereg's committed predictions ----
    tr_runs = transect["runs"]
    w_tr = np.zeros((len(tr_runs), 2, len(buckets)))
    anchor = sf.load_anchor(buckets)
    for m, run in enumerate(tr_runs):
        for b, v in transect["mixtures"][run["run_name"]]["phase0"].items():
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
    logger.info("validation #1 PASSED: transect 16 kernel + 16 swoosh values reproduced to 1e-9")

    # ---- the 6 points ----
    points = build_points(t_tok)
    w_all = np.stack([two_bucket_w(p["sliced_bucket"], p["w_target"], idx, len(buckets)) for p in points])
    ep_all = np.stack([real_ep_matrix(p, idx, len(buckets)) for p in points])
    hell = nearest_train_hellinger(w_all)

    for tag, params in (("zmacro", z_params), ("humaneval", hum_params)):
        pk = kernel_pred(w_all, tag)
        r_fit = sf.predict_r(params, w_all, ep_all, masks)
        for i, p in enumerate(points):
            p[f"pred_kernel_{tag}"] = float(pk[i])
            p[f"pred_swoosh_{tag}"] = float(pk[i] + r_fit[i])
    for i, p in enumerate(points):
        p["nearest_train_hellinger"] = float(hell[i])

    # ---- machinery validation #2: code arm == twobucket factorial ctr/a2_e16/a2_e32 to 1e-9 ----
    tb = json.loads(TWOBUCKET_PREREG.read_text())
    tb_by_point = {r["point"]: r for r in tb["runs"]}
    for p in points:
        if p["arm"] != "code":
            continue
        twin = tb_by_point[p["twobucket_twin"]]
        assert p["slice_batches"] == twin["code_slice_batches"], (p["point"], p["slice_batches"])
        for col in ("pred_kernel_humaneval", "pred_swoosh_humaneval", "pred_kernel_zmacro", "pred_swoosh_zmacro"):
            assert abs(p[col] - twin[col]) < 1e-9, (p["point"], col, p[col], twin[col])
    logger.info("validation #2 PASSED: code arm predictions == twobucket ctr/a2_e16/a2_e32 to 1e-9")

    # ---- kernel-flat check: pred_kernel is CONSTANT across epochs within each arm ----
    kernel_flat = {}
    for arm in ARMS:
        vals = [p["pred_kernel_humaneval"] for p in points if p["arm"] == arm]
        spread = float(max(vals) - min(vals))
        assert spread < 1e-9, (arm, vals)
        kernel_flat[arm] = {"pred_kernel_humaneval_constant": vals[0], "max_abs_spread_across_epochs": spread}

    # ---- swoosh predicted rise (e_high vs e=4), the pre-registered Delta the data must beat ----
    swoosh_rise = {}
    for arm in ARMS:
        by_e = {p["e_target"]: p for p in points if p["arm"] == arm}
        base = by_e[CONTROL_EPOCH]["pred_swoosh_humaneval"]
        swoosh_rise[arm] = {
            f"delta_e{int(e)}_vs_e4": round(by_e[e]["pred_swoosh_humaneval"] - base, 6)
            for e in ARMS[arm]["epochs"]
            if e != CONTROL_EPOCH
        }

    # ---- expand to 18 seed runs ----
    runs = []
    for p in points:
        for s in SEEDS:
            run = dict(p)
            run["seed"] = s
            run["point"] = f"{p['arm']}_e{int(p['e_target'])}_s{s}"
            run["run_id"] = f"{RUN_ID_PREFIX}_{run['point']}"
            run["job_name"] = f"rav-mve-epochrep-{run['point'].replace('_', '-')}"
            runs.append(run)
    assert len(runs) == 18 and len({r["run_id"] for r in runs}) == 18

    per_step = 1.7
    node_hours = sum(r["steps"] * per_step / 3600 for r in runs)

    prereg = {
        "date_utc": "2026-07-19",
        "experiment": "epochrep_seed_replicated",
        "issue": (
            "#7067 (mixing-via-embeddings); DECISIVE kernel-vs-form seed-replicated epoch-harm test (rav directive 2026-07-19)"
        ),
        "branch": "rav/mixing-via-embeddings",
        "launcher": "experiments/grug/moe/launch_mve_epochrep_h100.py",
        "why": (
            "DP4: in-regime the kernel misses NO repetition harm (0/37 tasks corr>0.1). DP3: only signal is n=4 "
            "high-code-rep runs, kernel-residual +0.023 bpb humaneval, ~4x the 0.006 bpb seed floor - underpowered. "
            "DP5: the in-regime-fitted swoosh UNDER-predicts that hint 2.6x, so the harm term cannot be calibrated "
            "from the observational sweep; it needs controlled high-rep data WITH seed error bars. Single-seed "
            "twobucket-a2 + transect show the SHAPE but cannot separate a ~0.02 bpb rise from sigma~0.006 bpb seed "
            "noise. This experiment supplies the replication."
        ),
        "arms": {
            "code": {
                "sliced_bucket": CODE_BUCKET,
                "partner_bucket": PARTNER_BUCKET,
                "group": "code_adjacent",
                "total_tokens": int(t_tok[CODE_BUCKET]),
                "w_target": W_TARGET,
                "epochs": list(ARMS["code"]["epochs"]),
                "control_epoch": CONTROL_EPOCH,
                "readout_task": "logprob_humaneval_10shot",
                "seed_pooling": (
                    "code arm at e in {4,16,32} is a 1:1 replica of twobucket factorial points ctr/a2_e16/a2_e32 "
                    "(seed 0); the factorial's seed-0 runs there pool as a 4th seed -> 4-seed SE at each code point"
                ),
            },
            "web": {
                "sliced_bucket": WEB_BUCKET,
                "partner_bucket": PARTNER_BUCKET,
                "group": "web_text",
                "total_tokens": int(t_tok[WEB_BUCKET]),
                "w_target": W_TARGET,
                "epochs": list(ARMS["web"]["epochs"]),
                "control_epoch": CONTROL_EPOCH,
                "readout_task": "logprob_humaneval_10shot",
                "seed_pooling": "no existing seed-0 web slice -> 3-seed SE at each web point",
            },
        },
        "constants": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "mixture_block_size": MIXTURE_BLOCK_SIZE,
            "steps": STEPS_10B,
            "phase_boundary_step": tbd.boundary_step(STEPS_10B),
            "experiment_budget_tokens": STEPS_10B * TOKENS_PER_BATCH,
            "model_d512": "swarm d512/6L/4H/1KV/64exp + gpu_fa4_cute (== seed panel/transect/twobucket)",
            "optimizer": "SWARM_OPTIMIZER fractional schedule (warmup peak 0.100*N verified at N=4776)",
            "seeds": list(SEEDS),
            "hardware": "cw-rno2a H100x8 one node/run, cuda_async allocator, CW-mirror data",
            "wandb_group": WANDB_GROUP,
            "checkpoint_prefix": "s3://marin-us-east-02a/marin/users/rav/grug/rav_mve_epochrep_<point>/dev/",
            "in_training_validation": False,
        },
        "slicing_mechanism": {
            "used": "LmDataConfig.max_train_batches={sliced_bucket: n} - slice-after-shuffle of the SAME cache",
            "why": (
                "identical estimand (T' = n*512*4096 tokens, real epochs via mixture restart-wrap), exact token "
                "control at 2.097M granularity, zero data movement; block shuffle permutes io-blocks globally so a "
                "prefix slice is a content-fair subsample. Content h=V.w is UNCHANGED by slicing -> kernel is flat."
            ),
        },
        "predictions_note": {
            "kernel": (
                "frozen Hellinger kernel ridge K=1000 (gamma 1.2475417504102333, alpha 0.1), dual on all 800 train "
                "runs; CONSTANT across epochs at fixed w by construction; validated by reproducing the transect "
                "prereg's 16+16 committed values AND the twobucket ctr/a2 code values to 1e-9"
            ),
            "swoosh": (
                "kernel + transect-committed b_g1 head (humaneval tau=2.0 b=5.872124766140496e-05; zmacro tau=5.5 "
                "b=0.002902417147546668; per-phase epochs, group 'all' so the epoching bucket's mass drives the rise); "
                "predicted humaneval bpb RISES with epochs"
            ),
            "kernel_flat_check": kernel_flat,
            "swoosh_predicted_rise_humaneval": swoosh_rise,
        },
        "decision_rule": {
            "estimand": (
                "per arm: Delta = mean_humaneval_bpb(e_high) - mean_humaneval_bpb(e=4), with 3-4-seed SE "
                "(code pools twobucket seed 0 -> 4 seeds; web 3 seeds). e_high in {16, 32} (code), {16, 24} (web)."
            ),
            "HARM_CONFIRMED": (
                "Delta > 2*SE AND Delta > 0 (worse) -> the functional form's harm term is REQUIRED for high-rep "
                "proposals, and Delta calibrates b."
            ),
            "NO_HARM": "Delta within 2*SE of 0 -> kernel + hard epoch caps suffice; no harm term needed.",
            "kernel_residual": (
                "per point: realized_humaneval_bpb - pred_kernel_humaneval (the flat content prediction) = the "
                "repetition effect isolated from content."
            ),
            "SE": "SE = sample_sd(seed bpb) / sqrt(n_seed); pooled control+treatment where applicable.",
        },
        "readout_plan": {
            "eval": (
                "post-hoc eval_logprob on the 18 final checkpoints (step 4775) + the 3 pooled twobucket code twins; "
                "humaneval bpb (perplexity per byte = 2^bpb reported alongside)"
            ),
            "aggregate": "mean +/- SE humaneval bpb per (arm, epoch) over seeds",
            "test": "Delta vs 2*SE per arm and epoch (the decision_rule)",
            "figure": (
                "f27_epoch_harm_calibration.png: humaneval bpb vs epochs per arm with seed error bars, the flat "
                "kernel prediction line, and the swoosh predicted curve overlaid"
            ),
        },
        "cost": {
            "n_runs": 18,
            "assumed_s_per_step_d512": per_step,
            "est_total_h100x8_node_hours": round(node_hours, 1),
        },
        "points": points,
        "runs": runs,
    }

    OUT_PATH.write_text(json.dumps(prereg, indent=1, sort_keys=True) + "\n")
    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    (OUT_PATH.parent / (OUT_PATH.name + ".sha256")).write_text(sha + "\n")
    logger.info("wrote %s", OUT_PATH)
    print(f"sha256 {sha}")
    for p in points:
        e0 = p["epochs"]["sliced"]["phase0"]
        print(
            f"{p['point']:>10} arm={p['arm']:>4} sliced={p['sliced_bucket']} w={p['w_target']:.2f} "
            f"n={p['slice_batches']:>4} e0={e0:6.2f} hell={p['nearest_train_hellinger']:.3f} "
            f"pk_h={p['pred_kernel_humaneval']:.4f} ps_h={p['pred_swoosh_humaneval']:.4f} "
            f"twin={p['twobucket_twin']}"
        )
    print("kernel_flat:", json.dumps(kernel_flat))
    print("swoosh_rise:", json.dumps(swoosh_rise))


if __name__ == "__main__":
    main()

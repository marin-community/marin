# Mixture surrogate — inference & sampling (humaneval bpb)

Self-contained package to **score and propose data mixtures** for the grug bucket set, using the
frozen mixing-via-embeddings surrogate (marin#7067; design #2403/#6969). It predicts a mixture's
**humaneval bits-per-byte** from the *content* it produces, reusing the 800-run swarm — so you can
rank mixtures and propose new sweeps **without training anything**. Pure `numpy`/`scipy`; the frozen
model (~1.7 MB) is baked into `model/`.

This branch is intentionally minimal: just what's needed to *use* the model. The full research
(validation, contours, method comparison, epoch experiments) stays on `rav/mixing-via-embeddings`.

## The model

```
yhat(w) = kernel(h = V·w)  +  epoch_harm(w)
```

- **`w`** — a candidate mixture: per-phase weights over the **168 buckets**, each phase summing to 1.
  Training is two-phase (phase-0 ≈ first 80% of tokens, phase-1 ≈ last 20%), so `w` has shape
  `(2, 168)`.
- **`h = V·w`** — the mixture's *content* fingerprint: a token-weighted histogram over a frozen
  K=1000 codebook, per phase. Two mixtures with the same `h` are treated identically — which is why
  a **new/changed bucket can be priced without re-sweeping** (it's just a new column of `V`), the
  whole point of the approach.
- **kernel** — Hellinger-kernel ridge over those histograms (`K = exp(−γ·D²_Hellinger)`, dual solve
  `(K+αI)a = y−ȳ`), frozen at `γ_factor = 0.25`, `α = 0.1`. Predicts held-out humaneval bpb at
  **Spearman 0.94** — ~94% of the seed-noise ceiling.
- **`epoch_harm(w)`** — an **always-on** additive bits-per-byte penalty for over-repeating a source
  (details below). Lower `yhat` is better.

Target is **humaneval bpb** (code). The swarm logged 52 bpb tasks; to fit any other one (e.g. a broad
macro-bpb, or a corpus eval you compute on the checkpoints), pass `target_y=` — see below.

## The epoch penalty (non-optional, self-gating)

The kernel is blind to *how many times* a source is repeated. The penalty adds that back:

```
epoch_harm(w) = amp(B, d) · Σ_g b_g · Σ_{j∈g} max(e0_j − τ, 0),   e0_j = w0_j · f0 · B / T_j
```

over the **code** and **web** bucket groups, where `e0_j` is a bucket's phase-0 epoch count at the
real training budget `B`, `T_j` its unique-token supply, and `amp(B,d) = (B/10B)^−0.73 · (d/512)^1.68`.
Read `b`/`τ` verbatim from the campaign's dedicated epoch experiments; validated against humaneval.

**It is self-gating at the real budget.** `max(e0 − τ, 0)` is 0 unless a source is genuinely
repeated past ~9 epochs — which for a normal (token-proportional-ish) mixture never happens, because
epochs stay flat and low. Over the entire in-support swarm the penalty is **identically 0** for any
budget in `1e10 … 1e12`, so **it never perturbs in-distribution ranking** (OOF Spearman stays 0.938).
It fires only for genuine over-repetition — e.g. dumping 35% of phase-0 onto a small code bucket,
which at the 10.4T effective budget is ~250 epochs and draws a ~0.08 bpb (≈15σ) penalty.

Because it is always applied and self-gating, you do **not** manage it by hand — just set
`budget_tokens` correctly (below).

## `budget_tokens` — the one required knob

```python
m = MixtureSurrogate(budget_tokens=1.037e13)   # the REAL token budget your sweep runs will train at
```

`budget_tokens` is required because it governs the penalty. A bucket's repetition is
`w · f_phase · budget / bucket_tokens`, so the same weight is over-repetition at a large budget and
harmless at a small one (you can't repeat an 11B-token bucket at a 10B budget). Set it to the token
budget your runs will actually train at. `hidden_dim` (default 512) is the width the runs use. Neither
affects normal in-support proposals (penalty 0 there) — they only set how hard genuine
over-repetition is penalised. `1.037e13` is the swarm's effective (simulated) budget the mixtures
encode; a good default if your runs reproduce the swarm's setup.

## Install & load

Needs `numpy` and `scipy` only. From the repo root:

```python
import sys; sys.path.insert(0, "experiments/mixture_surrogate")
from surrogate import MixtureSurrogate

m = MixtureSurrogate(budget_tokens=1.037e13)     # humaneval bpb
```

## Score a mixture

```python
w = m.anchor()                 # (2, 168) token-proportional design centre; or build your own
p = m.predict(w)
# p["mean"]         -> predicted humaneval bpb (lower = better) = content_mean + epoch_harm
# p["content_mean"] -> kernel term alone
# p["epoch_harm"]   -> the additive penalty (>0 only for over-repetition)
# p["sd"]           -> GP posterior sd (calibrated in-distribution; see caveats)
# p["nn_distance"]  -> Hellinger distance to the nearest swarm run (the off-support signal)
# p["in_support"]   -> False if beyond the training p95 (extrapolation; trust less)
```

`predict` accepts a batch too: `w` of shape `(n, 2, 168)` → arrays of length `n`.

## Propose mixtures to sweep

```python
prop = m.propose(n=50000, top_k=50, concentration=200.0)   # sample, score (penalty included), rank
prop["weights"]     # (50, 2, 168) best in-support mixtures, best first
prop["mean"]        # their predicted humaneval bpb
prop["epoch_harm"]  # the penalty component per proposal
```

Or from the command line, exporting a hand-off file for the training workstream:

```bash
python experiments/mixture_surrogate/sample.py --budget-tokens 1.037e13 --top-k 50 --out proposals.json
```

`proposals.json` is a ranked list; each entry has `predicted_mean`, `content_mean`, `epoch_harm`,
`predicted_sd`, `nn_distance`, `in_support`, `improvement_vs_anchor`, and `weights` as
`{"phase0": {bucket: w}, "phase1": {...}}` — directly consumable by whatever launches training.
`--concentration` controls exploration (lower = farther from the explored region, more novel but more
off-support); `--allow-off-support` lets it propose beyond the training p95.

## Fitting a different target

Only humaneval is baked in, but the swarm measured 52 bpb tasks (per-run). To fit any other one,
supply the per-run target array (aligned to the frozen swarm order, `len == n_train`):

```python
m = MixtureSurrogate(target="macro_bpb", budget_tokens=1.037e13, target_y=my_macro_bpb_array)
```

Corpus-perplexity evals (paloma / c4 / uncheatable) were **not** run on the swarm — they'd have to be
computed on the 800 saved checkpoints first, then dropped in via `target_y`.

## Verifying the method (this workstream's job)

The goal is to sweep some proposed mixtures and check predicted-vs-realized. Two useful designs:
1. **Confirm the optimum:** sweep the top proposals (in-support) and check they beat the anchor by
   the predicted margin.
2. **Stress extrapolation:** sweep some deliberately *far* mixtures (`--concentration` low, or
   `--allow-off-support`) and check whether `nn_distance` predicts where the surrogate degrades.
   Cross-validation says content extrapolation holds out to `nn ≈ 0.44` (Spearman barely drops);
   the untested case is truly-novel content at that distance on brand-new runs.

## Caveats (read before trusting a number)

- **Off-support = untrustworthy, and the sd does NOT flag it.** Predictions degrade sharply past the
  training p95 (`in_support=False`), but the GP `sd` inflates only ~2× while error can grow ~30×.
  Use `nn_distance` / `in_support` as the guardrail, **not** the error bars. In-distribution the sd is
  well-calibrated (95% coverage) but nearly flat (dominated by the ~0.006-bpb seed floor), so it ranks
  *distance-from-data*, not which in-regime prediction is wrong.
- **Set `budget_tokens` to the real run budget.** The epoch penalty's trigger depends on it. Too small
  and you can't express over-repetition; the penalty is a real-repetition effect at that budget.
- **Mean-reversion.** Proposals won't be more extreme than the swarm has seen; the surrogate reverts
  to the mean far from data. It's for *ranking realistic mixtures*, not inventing exotic ones.
- **These are proxy-scale rankings.** The swarm ran at d512. Rankings transfer only partially across
  budget (10B→100B Spearman ≈ 0.71) — good for coarse ranking, careful on fine order.

## Bucket set

The 168 buckets are 40 embedding-derived content clusters × 5 quality tiers, named `cNNqT`
(e.g. `c01q0` ≈ code, `c05q1` ≈ web). Full ordered list: `model/buckets.json`. The mapping from
bucket → underlying data lives on issue #7067; the swarm itself is public at
`marin-community/grug-moe-mix-swarm` (HF).

## Files

| file | what |
|---|---|
| `surrogate.py` | `MixtureSurrogate` — load, `predict`, `propose`, `epoch_harm`; featurization + kernel |
| `sample.py` | CLI: propose and export `proposals.json` |
| `demo.py` | runnable quickstart |
| `model/V.npy` | frozen content basis, (1000, 168) |
| `model/train_w.npy` | the 800 swarm mixtures, (800, 2, 168) |
| `model/train_y_humaneval.npy` | their realized humaneval bpb |
| `model/tj.npy` | per-bucket unique-token supply, (168,) — drives the epoch count |
| `model/mask_code.npy`, `model/mask_web.npy` | the code / web bucket groups for the epoch penalty |
| `model/buckets.json` | the 168 bucket names, in `V`/`w`/`tj`/mask column order |
| `model/meta.json` | frozen hyperparameters, target definition, epoch-penalty params |

The frozen path reproduces the full research pipeline's out-of-fold Spearman exactly (0.938);
`demo.py` re-checks this on load.

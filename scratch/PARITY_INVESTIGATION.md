# TPU vs GPU parity — investigation log

**Question.** exp153 (6B, CoreWeave GPU) plateaued at val **3.14** and was still there at 75% of
8 epochs. exp146 (3B, TRC TPU) finished at val **2.7025–2.7749** across its top runs. A 6B
landing worse than a 3B on the same corpus is not a subtle tuning story, so: is something about
GPU training wrong, or was exp153 just badly configured and unfairly compared?

Living document — every iteration of the supervision loop appends here. Written because the
conversation carrying these findings will be compacted away.

> **SUPERSEDED 2026-07-30 — the platform conclusion below is wrong.** exp166cw ran
> full-length 1.5B jobs on GPU (35,680 and 71,360 steps); they land a constant **+0.41**
> over exp117's TPU results for the same configs, matching exp153's deficit. The decisive
> case is unaugmented and starts from pretrained weights: a model that reached **2.7131**
> on TPU trained 8 further epochs on GPU and ended at **3.1597**.
>
> Why this missed it: the longest arm here is **4,460 steps**, and exp153's gap opens
> around step 4,460 then holds flat. Every arm stopped at or before the edge of the effect.
> See `experiments/protein/exp166_cw_plan.md` on `eac/plm-exp166-cw`.

## Method

`scratch/parity_train.py` runs **one recipe on three accelerators**, changing only
`ResourceConfig`. exp153 and exp166 differ in a dozen incidental ways, so a loss gap between
them proves nothing; a gap in this harness means something.

- **v6e-4 (4 chips)** and **GB200x4 (4 GPUs)** — mesh-matched: same device count, same
  data-parallel width, same 8 sequences per device. This is the controlled pair.
- **H100x8 (8 GPUs)** — deliberately varies the mesh, to separate "different silicon" from
  "different mesh shape".
- 1.5B qwen3 (exp166's config), seq 8192, batch 32, lr 1e-3, wd 0.2, seed 0.
- Both platforms pinned to token caches **verified byte-identical**.
- All checkpoints to `tmp/ttl=1d/`, guarded; nothing written to a permanent prefix, nothing
  deleted.

## Findings

| # | Finding | Status |
|---|---|---|
| F1 | GCS and S3 token caches are byte-identical: same 4,129,682 rows, same `sharded` layout, first 16 KB of the first chunk matches exactly | **Data ruled out** |
| F2 | v6e-4 vs GB200x4 over 60 steps: deltas oscillate in sign at 0.01–0.06 on losses of 5.7–11.7 (0.1–1%) | **No platform gap at matched mesh, short horizon** |
| F3 | Same-config runs on *different node counts* diverge from step 2 (measured earlier on H100 1-node vs 2-node) | Reduction-order noise is expected and benign |
| F4 | exp146's 3B winner was **LR 3.1623e-3 / WD 0.2 → 2.7109**; five of its top six runs use LR 3.1623e-3 | exp153's LR 1e-3 / WD 0.8 is off-optimum on **both** axes |
| F5 | exp153's 3.14 was read at step 22,300 of 35,680 (62%) against exp146 **final** values, under cosine decay | Comparison was unfair, but see F6 |
| F6 | At **equal step**, exp153 trails exp146 by a *flat* ~0.40 from step 4,460 onward (+0.419, +0.488, +0.399, +0.423, +0.400, +0.382). The gap opens between step 2,230 and 4,460 and never widens | **A3 refuted.** The gap is real, not a stopping-point artifact. A constant early offset implies a different trajectory from the start, not accumulating numerical error |
| F7 | Three-way at step 59: v6e-4 5.75413, GB200x4 5.69206 (−1.08%), H100x8 5.74982 (−0.07%). The **8-device H100 matches the TPU more closely than the 4-device GB200 does** | **B3 answered: mesh shape is not the driver.** If mesh width mattered, H100x8 would be the outlier; instead it is the closest match. What remains is small per-device kernel numerics |
| F8 | 400-step pair, windowed means of the GB200−TPU delta: +0.36% (0–49), −1.25% (50–99), −5.62% (100–149), −1.22% (150–199), then **monotonically shrinking: −0.19%, −0.12%, −0.08%** through step 349. Per-step delta flips sign 19 times | **B4 refuted, and then some.** The two platforms do not merely fail to diverge — they *converge*, the delta decaying toward zero as training proceeds. The mid-run excursion is a transient. Both GPUs sit *below* the TPU, the opposite sign from exp153's +0.40 deficit |
| F9 | **The 400-step pair ran to completion: v6e-4 4.090732 vs GB200x4 4.092026 — a delta of +0.032%.** Windowed delta decays −1.25% → −5.62% → −0.19% → −0.08% → −0.08% | **The platform question is closed.** Over a full run, identical config on TPU and GPU lands within a third of a tenth of a percent. exp153's deficit is ~13% relative — roughly 400x larger than anything the hardware produces |
| F10 | HP ablation at step 208 of 400: exp153's lr 1e-3 / wd 0.8 is **ahead** of exp146's lr 3.1623e-3 / wd 0.2 by 0.33, and has led at every sampled step | **Cannot be read as an answer — horizon mismatch.** See below |
| F11 | Continuing those arms, the delta collapses: +0.386 (150–199) → +0.084 (250–299) → +0.050 (300–349) | The caveat on F10 was correct — the ordering did not survive 100 more steps |
| F12 | The 400-step HP arms finished **4.102407 (exp153 HP) vs 4.131758 (exp146 HP) — 0.7% apart**, after opening +0.66 | At a matched schedule endpoint, LR/WD is worth ~0.03, not ~0.40. The mid-run +0.33 was schedule phase, not merit |
| F13 | **Every exp146 and exp153 production run shares an identical token budget of 4,567,040 sequences.** Batch size varies and steps vary inversely: bs 64/71,360, bs 128/35,680, bs 256/17,840 | Equal-*step* comparisons across different batch sizes are **not** equal-token — see the F6 correction below |
| F14 | Within exp146, same LR/WD at different batch sizes lands differently: lr 3.162e-3 wd 0.4 gives **2.7952 at bs 128** but **2.7025 at bs 256**; the wd 0.2 winner ran **bs 64 → 2.7107** | **Batch size is an independent lever worth ~0.09**, untested by anything in this investigation. exp153 ran bs 128, exp146's best runs ran bs 256 |
| F15 | **B1 complete, 390 matched steps: TPU splash vs TPU `JAX_FLASH`, everything else identical.** Final 4.090732 vs 4.103658 (**+0.32%**); mean delta over the run **+0.00629**, sign-oscillating throughout | **B1 refuted, and a real confound closed.** Every earlier platform comparison varied the attention kernel alongside the platform, because `JAX_FLASH` is mandatory on GPU. It turns out not to matter, so F2/F7/F8/F9 stand as platform results rather than kernel results |
| F16 | **A5 complete, 4,460 steps.** Final **3.886545 (lr 1e-3 / wd 0.8) vs 3.951600 (lr 3.1623e-3 / wd 0.2) — delta +0.065 (1.67%)**. The delta traces an arc: +0.006 (step 500) → +0.028 (2600) → +0.021 (3200) → **peak +0.097 (3650, decay phase)** → settles to +0.064 | **A1/A2/A5 answered: LR/WD is worth ~0.065, at most ~0.097 transiently.** That is a fifth of the 0.34–0.40 gap — real but not the cause. **Direction favours exp153's lr 1e-3 / wd 0.8**, inverting the premise the ablation was built on: exp153's hyperparameters are not wrong at 1.5B/bs32. I called this curve saturating twice before the decay phase and was wrong both times |
| F17 | **C4 complete.** bs 16 x 800 vs bs 32 x 400, matched 12,768-sequence budget, fixed LR. Delta decays monotonically: −1.09 (2.4k seqs) → −0.29 (4.8k) → −0.05 (10.4k) → **−0.016 at the matched endpoint**; late-half mean −0.046 | **C4 refuted as an explanation.** The early advantage is an update-count transient that washes out. The residual is ~20x too small for the 0.34–0.40 gap — *and* the comparison confounds three things (batch size, update count, effective LR per token, since LR was not scaled with batch), so even the residual is not cleanly attributable to batch size |

### The horizon problem with F10

The naive reading — "exp153's hyperparameters are actually better" — does not follow, and the
ablation as designed cannot support it. exp153's gap opens at step **4,460**; this ablation
runs **400** steps, so cosine decay is compressed ~89x and the whole comparison sits in the
warmup-and-early-transient regime. A larger LR is expected to look worse there: it is noisier
early and pays off over a long horizon. Measuring at step 208 and concluding anything about
step 4,460 is exactly the extrapolation that would make this wrong.

What F10 does establish: at matched hardware and matched everything-else, **LR/WD alone moves
the loss by ~0.33 at 200 steps** — the same order as the ~0.40 that this whole investigation
is chasing. Hyperparameters are a large enough lever to explain the gap. Which *direction*
they move it at 4,460 steps is the open question, and answering it needs a longer run than the
"don't run them for long" budget this harness was built under. Flagged rather than launched.

**F11 confirms the caveat was the right call.** Continuing the same two arms, the windowed
delta does not hold — it collapses as cosine decay brings the high-LR run in:

| window | exp153 HP | exp146 HP | delta |
|---|---|---|---|
| 0–49 | 6.86385 | 7.52541 | +0.66155 |
| 150–199 | 4.29726 | 4.68346 | +0.38620 |
| 200–249 | 4.21831 | 4.54692 | +0.32862 |
| 250–299 | 4.18261 | 4.26629 | **+0.08367** |
| 300–349 | 4.14888 | 4.19892 | **+0.05003** |

An extrapolation from step 208 would have been wrong within 100 steps. The high-LR arm was
never losing on the merits; it was behind on the schedule. This is the expected signature of
a larger LR under cosine — worse early, closing hard at the end — and it is why a 400-step
proxy cannot settle a question about step 4,460. The ordering at the end of *this* schedule
still says nothing about the ordering at the end of a 35,680-step one.

### Correction to F6

F6 compared exp153 against exp146's winner at equal step and found a flat ~0.40. That winner
runs **bs 64**; exp153 runs **bs 128**. At any equal step the exp146 run had therefore seen
**half** the sequences — so it was ahead by 0.40 on half the tokens. The equal-step framing
understates the gap rather than manufacturing it, and F6's conclusion survives, but the
number is not the honest size of the effect.

The better-matched comparison is exp146's **bs 128 / 35,680-step** run — same batch, same
steps, same token budget as exp153 — which reached **2.7952** against exp153's 3.14. That
narrows the confound to three variables: 6B vs 3B, lr 1e-3 vs 3.162e-3, wd 0.8 vs 0.4.

## Hypotheses

Ordered by how much they would explain, not by how interesting they are.

### A — It was never a platform problem

| id | hypothesis | test | status |
|---|---|---|---|
| A1 | LR 1e-3 is simply wrong at this scale; the optimum is ~3.16e-3 | rerun 1.5B at both LRs, same platform | **REFUTED (F16)** — lr 1e-3 *wins* by 0.065 at 1.5B/bs32 |
| A2 | WD 0.8 over-regularizes; winners cluster 0.1–0.4 | rerun at wd 0.2 vs 0.8 | **REFUTED (F16)** — wd 0.8 wins in the same arm |
| A3 | The gap is mostly "62% vs 100% under cosine" | compare exp153's curve to exp146 at *equal step* | **REFUTED (F6)** — gap is flat ~0.40 at every matched step |
| A5 | The ~0.40 is an LR/WD effect: 1e-3 vs 3.1623e-3 (half a decade) and wd 0.8 vs 0.2 (4x) | HP ablation on ONE platform, no hardware involved | **REFUTED (F16)** — worth 0.065, wrong sign |
| A4 | The 6B arch is not a clean width-scale (64 heads x 64 head_dim vs hidden 3200; 2:1 GQA vs 4:1), so LR does not transfer — exp108 showed transfer holds under width, not depth | short 6B run at several LRs | open |

### B — Platform / numerics

| id | hypothesis | test | status |
|---|---|---|---|
| B1 | Attention kernel differs: TPU splash vs GPU `JAX_FLASH` | `ATTN=JAX_FLASH` on TPU — harness already supports it | **REFUTED (F15)** — +0.32% over a full 400-step run |
| B2 | Reduction order / accumulation depth | mesh-matched pair | **F2: not at 4 devices, 60 steps** |
| B3 | Mesh shape itself (4 vs 8 devices) | `parity-h100-x8` | **REFUTED (F7)** — H100x8 is *closer* to TPU than GB200x4 |
| B4 | Divergence only appears over a longer horizon | 400-step TPU + GB200 pair | **REFUTED (F8)** through 224 steps — excursion closes, does not compound |
| B5 | Fused-CE Pallas kernel: GPU autotunes it; TPU path may differ | compare loss with the kernel disabled | open |
| B6 | f32 FSDP parameter gathers (exp108's suspicion) inflate error at large dp | large-dp vs small-dp on one platform | open |
| B7 | Precision policy applies identically but accumulates differently inside kernels | bit-level compare of first-step grads | open |

### C — Scale-dependent (would explain why 1.5B looks fine)

| id | hypothesis | test | status |
|---|---|---|---|
| C1 | The effect needs 6B, not 1.5B — deeper stacks, larger activations | short 6B run on both platforms | open |
| C2 | The effect needs large dp (64 devices at 8 nodes), not 4 | 1.5B at 8 nodes vs 1 node on GPU | open |
| C4 | **Batch size**: exp153 ran bs 128 where exp146's best rungs ran bs 256 | 1.5B at bs 16 vs 32, matched token budget | **REFUTED (F17)** — converges to −0.016, and direction is opposite to F14's |
| C3 | Gradient accumulation only engages at ga>1; the parity runs use ga=1 | run a config forcing ga>1 on both | open |

### D — Infrastructure

| id | hypothesis | test | status |
|---|---|---|---|
| D1 | Two of two 8-node gangs died to a single-rank SIGSEGV (exit 139) at ~1.5 days. If that is memory corruption rather than preemption, it could also perturb numerics before killing the job | check whether loss anomalies precede the crashes | open |
| D2 | exp153's val spikes (+0.60 at 13,380, +0.43 at 24,530) indicate marginal stability | correlate spikes with checkpoint/restart events | open |

## Experiment log

| date | experiment | result |
|---|---|---|
| 07-29 | `compare_caches.py` — GCS vs S3 token caches | **identical** (F1) |
| 07-29 | `parity-gb200-x4-b` — GB200x4, 60 steps | finished, final train 5.69206 |
| 07-29 | `parity-tpu-v6e4-d` — v6e-4, 60 steps | finished, final train 5.75413 |
| 07-29 | `parity-h100-x8` — H100x8, 60 steps (B3) | finished, final train 5.74982 → **B3 refuted (F7)** |
| 07-29 | `parity-gb200-long` — GB200x4, 400 steps (B4) | finished, final train 4.09203 |
| 07-29 | `parity-tpu-long` — v6e-4, 400 steps (B4) | finished, final train 4.090732 → **platform closed (F9)** |
| 07-29 | `parity-hp-exp153` — GB200x4, 400 steps, **lr 1e-3 / wd 0.8** (A5) | finished, final train 4.102407 |
| 07-29 | `parity-hp-exp146` — GB200x4, 400 steps, **lr 3.1623e-3 / wd 0.2** (A5) | finished, final train 4.131758 → **0.7% apart (F12)** |
| 07-29 | `parity-hp-exp153-long` / `parity-hp-exp146-long` — GB200x4, **4,460 steps** | finished, 3.886545 vs 3.951600 → **A1/A2/A5 refuted (F16)** |
| 07-29 | `parity-tpu-flash` — v6e-4, 400 steps, `ATTN=JAX_FLASH` (B1) | finished, final train 4.103658 vs 4.090732 baseline → **B1 refuted (F15)** |
| 07-29 | `parity-tpu-bs16` — v6e-4, bs 16 x 800 steps (C4) | finished → **C4 refuted (F17)**, −0.016 at the matched endpoint |

## Harness failures worth remembering

- `/scratch` is gitignored, and the iris bundler ships `git ls-files --cached --others
  --exclude-standard`. Scripts under `scratch/` must be `git add -f`'d or they never reach the
  cluster.
- Marin refuses a cross-region GCS read. Pin the TPU region *and* derive the cache path from it.
- The region-to-bucket mapping is **not** `f"marin-{region}"` — europe-west4 lives in
  `gs://marin-eu-west4`. Use `marin_prefix_for_region()`.
- `marin_temp_bucket()` resolves from ambient config and picks the *GCS* cluster config inside a
  CoreWeave pod. Temp roots here are pinned explicitly per cloud instead.
- This script is a *driver*: it must be submitted through `iris job run`, never invoked
  directly on the dev box. Run locally it silently picks `fray.local_backend` and dies with
  "No accelerator found" — `plat.cluster` is metadata, not dispatch.
- Submitting from the dev box reads the bucket, so S3 arms need `CW_KEY_ID`/`CW_KEY_SECRET`
  exported as `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` *and* `configure_coreweave_s3()` —
  CoreWeave rejects boto's default path-style addressing with a bare `Forbidden`.
- `--cluster` is a *top-level* iris option (`iris --cluster X job list`), not a subcommand
  flag. GPU arms federate from `marin`; `iris --cluster cw-us-east-08a` needs kubectl, which
  this box does not have.

## Next up

~~A3~~ (F6), ~~B3~~ (F7) and ~~B4~~ (F8) all refuted. **Every platform hypothesis tested so far
has come back negative, and on three separate comparisons the GPUs land at or slightly below
the TPU** — so the ~0.40 deficit almost certainly does not come from the hardware.
Remaining, in priority order:

1. **The decisive experiment (C1 + A5 together): run exp153's exact 6B config — lr 1e-3,
   wd 0.8 — on a TPU.** If the TPU also lands ~0.40 above exp146, the cause is the config or
   the architecture and the platform is exonerated outright. If the TPU instead tracks
   exp146, the gap is GPU-specific and F2 only means the effect needs 6B or large dp to
   appear. Either answer closes the investigation. Short run suffices: the gap is fully
   open by step 4,460.
2. **A5** — 1.5B at lr 1e-3/wd 0.8 vs lr 3.1623e-3/wd 0.2 on a single platform, to size how
   much of 0.40 is pure hyperparameter.
3. **B1** — `ATTN=JAX_FLASH` on TPU isolates the attention kernel.
4. **C2** — 1.5B at 8 nodes vs 1 node on GPU, to test whether large dp is the trigger.

# TPU vs GPU parity — investigation log

**Question.** exp153 (6B, CoreWeave GPU) plateaued at val **3.14** and was still there at 75% of
8 epochs. exp146 (3B, TRC TPU) finished at val **2.7025–2.7749** across its top runs. A 6B
landing worse than a 3B on the same corpus is not a subtle tuning story, so: is something about
GPU training wrong, or was exp153 just badly configured and unfairly compared?

Living document — every iteration of the supervision loop appends here. Written because the
conversation carrying these findings will be compacted away.

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
| F8 | 400-step pair, windowed means of the GB200−TPU delta: +0.36% (0–49), −1.25% (50–99), −5.62% (100–149), −1.22% (150–199), **−0.24% (200–249)**. Per-step delta flips sign 19 times; mean −0.089 | **B4 answered: no late divergence.** The mid-run excursion closes back to noise rather than compounding. Both GPUs sit *below* the TPU — the opposite sign from exp153's +0.40 deficit |

## Hypotheses

Ordered by how much they would explain, not by how interesting they are.

### A — It was never a platform problem

| id | hypothesis | test | status |
|---|---|---|---|
| A1 | LR 1e-3 is simply wrong at this scale; the optimum is ~3.16e-3 | rerun 1.5B at both LRs, same platform | open |
| A2 | WD 0.8 over-regularizes; winners cluster 0.1–0.4 | rerun at wd 0.2 vs 0.8 | open |
| A3 | The gap is mostly "62% vs 100% under cosine" | compare exp153's curve to exp146 at *equal step* | **REFUTED (F6)** — gap is flat ~0.40 at every matched step |
| A5 | The ~0.40 is an LR/WD effect: 1e-3 vs 3.1623e-3 (half a decade) and wd 0.8 vs 0.2 (4x) | HP ablation on ONE platform, no hardware involved | open — highest value |
| A4 | The 6B arch is not a clean width-scale (64 heads x 64 head_dim vs hidden 3200; 2:1 GQA vs 4:1), so LR does not transfer — exp108 showed transfer holds under width, not depth | short 6B run at several LRs | open |

### B — Platform / numerics

| id | hypothesis | test | status |
|---|---|---|---|
| B1 | Attention kernel differs: TPU splash vs GPU `JAX_FLASH` | `ATTN=JAX_FLASH` on TPU — harness already supports it | open |
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
| 07-29 | `parity-tpu-long` — v6e-4, 400 steps (B4) | running, step 224 → **B4 refuted so far (F8)** |
| 07-29 | `parity-hp-exp153` — GB200x4, 400 steps, **lr 1e-3 / wd 0.8** (A5) | running |
| 07-29 | `parity-hp-exp146` — GB200x4, 400 steps, **lr 3.1623e-3 / wd 0.2** (A5) | running |

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

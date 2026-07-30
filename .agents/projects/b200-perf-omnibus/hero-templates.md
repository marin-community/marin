# Hero templates — the clean PR branch

Team decision, 2026-07-29: Larry, Rafal and Matt combine their work into a single PR branch
rather than three. This document is the authoritative commit plan for that branch. It
supersedes the "wait for their PRs" staging in earlier notes.

Target branch: `mcwitt/grug-moe-hero-templates`, cut from `origin/main` @ `7c3440426`.

## What the branch contains

Two new Grug variants, per the [`change-grug`](../../skills/change-grug/SKILL.md) convention
(`experiments/grug/<variant>/`, copied from `experiments/grug/moe`, edits local and explicit):

- `experiments/grug/moe_hero_fsdp` — Larry's best FSDP configuration, `expert_axis_size=1`.
- `experiments/grug/moe_hero_ep` — the D-2 composed EP64 configuration.

Both **hardcode** the parameters appropriate to their flavor. No new environment variables, no
`default_*` wrappers, no thin shims over the `SCALE_*` set. This is Rafal's stated preference
and it is what `AGENTS.md` asks for ("prefer explicit constructor/config parameters over env
vars", "force explicit specification of critical parameters").

## Two departures from [`sequence.md`](sequence.md)

1. **Do-regardless items also land in the base `moe` template.** Any change that is
   high-confidence, low-LOC and wanted by *both* hero flavors is folded into
   `experiments/grug/moe` first, and the hero copies inherit it. The alternative — landing it
   only in the two new directories — would leave the base template worse than its own
   descendants and would duplicate every such change twice.
2. **Larry's recent wins are in scope**, as clean commits: the chunked cross-entropy
   (`8223bba67`) and the shared-expert split (`e346b72d8`). Both were previously held back
   ([`sequence.md`](sequence.md) Phase F, "two levers deliberately held back") on the grounds
   that they were below the 2pp draw threshold and, for the shared expert, memory-blocked at
   EP64. That reasoning stands for the *EP* template; it does not apply to the FSDP template,
   where Larry measured them as wins on the arm they ship on.

## Reviewable in two PRs

The sequence is ordered so a prefix cut yields the first PR:

- **PR 1** = blocks 1–3. Shared library substrate, the base-`moe` folds, and the FSDP hero
  template. Contains no expert-parallel switches, so it is reviewable without EP context.
- **PR 2** = blocks 4–5. EP enablement, the EP throughput core, and the EP hero template.

## Block 1 — shared library substrate (`lib/levanter`)

| # | Commit | Source |
|--:|---|---|
| 1 | `[levanter] Add QuACK SM100 kernel dependency` | `d6fc1e7c1` |
| 2 | `[levanter] Add QuACK SM100 MoE backend` | `29ad214ce` |
| 3 | `[levanter] Report chunked Sonic capacity drops` | `284608114` |
| 4 | `[levanter] Update FA4 backward for CUTLASS 4.6` | `dfa60932b` |
| 5 | `[levanter] Chunk the cross-entropy for large vocabularies` | `8223bba67`, re-extracted |

Commit 1 precedes 2 because the backend imports QuACK. Commit 3 must not be separated from 2
by any commit that can run the chunked backend: without it the backend drops while reporting
zero, which is indistinguishable from a clean run.

Commit 5 is Larry's win. It arrives as an explicit config parameter, not as the
`CE_IMPL` / `CE_LIGER_CHUNK` / `CE_LIGER_UNROLL` environment triple it has on
`origin/b200-300B-tune`.

## Block 2 — base `moe` template, dispatch, docs

| # | Commit | Source |
|--:|---|---|
| 6 | `[grug] Forward XLA allocator settings to training tasks` | `3205b35c5` |
| 7 | `[grug] Propagate Grug child job priority` | `837b2c457` |
| 8 | `[grug] Add homogeneous scan-layer execution` | `ddbe74694` |
| 9 | `[grug] Gate drop reporting from the scale launcher` | `4124d904b` |
| 10 | `[grug] Compute drop fractions on the host` | `88353e7ad` |
| 11 | `[grug] Pin the MoE capacity-factor default` | `c968939f6` |
| 12 | `[grug] Make embedding gather replica-local` | `371518afc` |
| 13 | `[grug] Hoist FA4 segment bounds out of the layer scan` | `852a768d5` |
| 14 | `[grug] Route scanned expert stacks through Newton-Schulz` | `f02c87996` |
| 15 | `[grug] Parameterize scale-run GPU resources` | `9e0c66ff4` |
| 16 | `[grug] Parameterize scale-run model shape` | `b7458ba86` |
| 17 | `[grug] Select the scale-run optimizer from the environment` | `a4d2d795e` |
| 18 | `[grug] Complete GB200 scale-run controls` | `7c53bea86` |
| 19 | `[grug] Offload MuonH optimizer state to pinned host memory` | `ec98821f3` |
| 20 | `[docs] Document required GB200 XLA flags` | `84eb8e280` |

Commit 6 must widen the forwarded-env prefix to include `CE_`, or block 1's commit 5 is
unreachable from a nested training task. Commit 10 fixes commit 9's metric above one rack:
`batch × seq × top_k × layers` overflows signed int32, so `SCALE_REPORT_DROPS` has never
worked at multi-rack scale. Commit 14 is do-regardless because commit 8's array-stacked blocks
produce 4D expert stacks that MuonH must handle whether or not there is an expert axis.

Commit 19 lands as a documented flag, not a default: host offload was used on the d6144 EP64
leg and rejected at d5120, where it needed a 135 GiB pinned-host arena and landed at 19.694%.

## Block 3 — the FSDP hero template

| # | Commit |
|--:|---|
| 21 | `[grug] Add the moe_hero_fsdp variant` |
| 22 | `[grug] Split the shared expert in moe_hero_fsdp` |
| 23 | `[grug] Chunk the cross-entropy in moe_hero_fsdp` |
| 24 | `[docs] Record moe_hero_fsdp in the Grug archive` |

Target configuration, from W&B run `gb200-d6144-64gpu-nomtp-noconv-bs1152-chunk4-v1`
(23.653% MFU / 283,560 tok/s, one rack):

| setting | value |
|---|---|
| hidden / layers | 6144 / 48 |
| experts / top-k | 128 / 4 |
| intermediate / shared intermediate | 3072 / 6144 |
| sequence length / sliding window | 4096 / **512** |
| MoE implementation | `sonic_cute` |
| `expert_axis_size` | 1 |
| topology | 16 nodes × 4 GB200 |
| batch | 1152 |
| QB routing / XSA | on / on |
| optimizer | MuonH, `learning_rate=0.05`, linear, `min_lr_ratio=0.05`, `warmup=0.01` |

`sliding_window=512`, not 2048 — the EP configs use 2048 and the value is easy to carry over
wrongly. `gated_norm` and `attn_gate` are not config keys in Larry's runs; only `xsa` is
explicit. Record what the template does rather than inferring his values.

## Block 4 — EP enablement and throughput core (`lib/levanter`)

| # | Commit | Source | Measured |
|--:|---|---|---|
| 25 | `[levanter] Preserve expert sharding through Newton-Schulz` | `966de0092` | — |
| 26 | `[levanter] Return padded Muon stacks in parameter sharding` | `28705513e` | +1.78pp |
| 27 | `[grug] Shard non-expert state across expert parallelism` | `18aa3f4ff` | — |
| 28 | `[levanter] Preserve expert sharding at MoE dispatch` | `5cfdc92b7` | — |
| 29 | `[levanter] Add fixed-capacity expert all-to-all` | `164ff408c` | ~13% → 17.8% |
| 30 | `[levanter] Spill overflow assignments within the routing step` | `4941d5548` | −0.213pp for half the drops |
| 31 | `[levanter] Gather activations for expert dispatch` | `414e7a1d4` | +3.01pp |
| 32 | `[levanter] Remove scatter from fixed all-to-all backward` | `860a7c0de` | +3.43pp |

Ordering is load-bearing: 25 before 26, and 29 before 30 before 31/32. Commit 30 ships at or
before the throughput commits — landing throughput without the fidelity work reproduces
exactly the situation this project spent a week correcting.

## Block 5 — the EP hero template

| # | Commit |
|--:|---|
| 33 | `[grug] Add the moe_hero_ep variant` |
| 34 | `[docs] Record moe_hero_ep in the Grug archive` |

Target configuration, from the D-2 draw-3 command (three-draw median 22.398% MFU /
346,950 tok/s / 1.444% drops, one rack):

| setting | value |
|---|---|
| hidden / layers | 5120 / 48 |
| experts / top-k | 256 / 8 |
| intermediate / shared intermediate | 1280 / 5120 |
| sequence length / sliding window | 4096 / 2048 |
| MoE implementation | `ragged_all_to_all`, fixed capacity, gather dispatch, custom adjoint |
| `expert_axis_size` | 64 |
| capacity factor / spill attempts | 1.0625 / 3 |
| topology | 16 nodes × 4 GB200 |
| batch | 1024 |
| QB routing | on |
| attention | `gpu_fa4_cute` |
| optimizer | MuonH, padded non-expert Newton–Schulz, SYRK |
| remat | `recompute_all` |

Required XLA flags, which the template documents rather than sets:
`--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false` (mandatory on JAX 0.11
— without it a 64-process run segfaults in `ncclDevCommCreate` before step 0),
`--xla_gpu_experimental_parallel_collective_overlap_limit=4`,
`--xla_gpu_enable_latency_hiding_scheduler=true`, and
`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`. Auto-PGLE must stay off; it crashes multi-host. Do
not ship a pinned PGLE profile — the D-2 profile matched 225 of 533 instructions and manual
PGLE was rejected on the EP line for a 0.235pp deficit against AutoPGLE.

## Excluded, and why

- **`.agents/projects/b200-perf-omnibus/**`** — the per-item reports travel with commits on
  the omnibus branch. They are project scaffolding and do not belong in a PR against `main`.
- **The Fray gang-topology fix** (`256d75e99`,
  [#7753](https://github.com/marin-community/marin/issues/7753)) — has its own PR in flight.
  It remains a prerequisite for any multi-rack hero run; it is simply not this branch's to
  land.
- **D5 QuACK grouped wgrad**, **Receiver-ECHO**, **MXFP8**, **latent MoE**,
  **`SCALE_MOE_EXPERT_CHUNKS`**, **slim Sonic residuals** — unchanged from
  [`sequence.md`](sequence.md).
## Resolved: `z_loss_weight` needs no commit

An earlier note held this open, claiming `z_loss_weight = 0.0001` appeared in no `main` levanter
or Grug path. That was wrong — it was looked for in `lib/levanter`, but Grug config lives in the
experiment variant. On `origin/main` it is already the default in three places:
`experiments/grug/moe/train.py:61` (`z_loss_weight: float = 1e-4`),
`experiments/grug/moe/launch_cw_scale.py:89` (`SCALE_TRAINER_DEFAULTS`), and
`experiments/grug/moe/launch.py:238`. `experiments/grug/moe/README.md` documents it, along with
router z-loss being off by default (`router_z_loss_coef = 0.0`). Nothing to add; both hero
templates inherit it by copying `moe`.

This has a consequence for the chunked cross-entropy. `lib/levanter/src/levanter/grug/loss.py`
threads `logsumexp_weight` into `fused_cross_entropy_loss_and_logsumexp_penalty`, so the chunked
path must carry the z-loss penalty too. `mean(logsumexp(logits)²)` is a per-token quantity, so
chunking it is exact **only if the per-chunk means are recombined weighted by each chunk's token
count**. A ragged final chunk averaged unweighted with the others silently changes the penalty,
and would do so by a small enough amount to look like noise. This is the specific defect to test
for, not a hypothetical.

## Verification

Per phase, the assertions named in [`sequence.md`](sequence.md) §Verification, plus for the
new variants: `uv run pytest tests/test_grug_variant_contracts.py`, which imports variant
modules by name and therefore needs entries for both new directories. Every commit boundary
must satisfy `./infra/pre-commit.py --all-files` and `uv run pyrefly`.

Neither template has accelerator verification on this branch. The FSDP template's 23.653% and
the EP template's 22.398% were measured on research builds, not on this assembly. Any claim
that the templates reproduce those numbers needs a rack draw against the branch itself.

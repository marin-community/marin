# Status — what has actually been done

Snapshot: 2026-07-29. This is the record of work performed; [`sequence.md`](sequence.md)
and [`derisking.md`](derisking.md) are the plan, and they have been reconciled against
what is below.

**Nothing here is merged.** Every item lives on its own `agent/*` branch against
`origin/main` @ `6ce4a7e68`. Each branch carries its own `-report.md` in this directory,
which is why the reports are not visible from `b200-perf-omnibus` — check out the branch,
or read it on GitHub, to see the detail behind a row.

## Commit series

| # | Item | State | Branch | Commits |
|---|---|---|---|---|
| A0 | Scan-layers port | done | `agent/impl-b0-scanlayers` | `5f0efe967` |
| A1 | `SCALE_REPORT_DROPS` gate | done | `agent/impl-a1-dropmetric` | `1116b1a0d`, `c879b8449` |
| A2 | Capacity-factor default | done | `agent/impl-c34a2-epshard` | `ff8411e54` |
| A3 | XLA flag documentation | done | `agent/impl-a3-flagdocs` | `c5066d3a3` |
| A4 | Dispatcher env prefix | done | `agent/impl-a4-envprefix` | `07b1b4a82` |
| B1 | QuACK dependency | done | `agent/impl-b12-sonic` | `47c31192c` |
| B2 | `sonic_cute` backend | done | `agent/impl-b12-sonic` | `74bcdcf40` + chunk-drop fix |
| B3 | Replica-local embedding gather | done | `agent/impl-b3-embgather` | `56e69f828` |
| B4 | FA4 bounds hoist | in flight | `agent/impl-b4-fa4bounds` | was blocked on A0; relaunched |
| C1 | 4D expert stacks → Newton–Schulz | done | `agent/impl-c12d4-muon` | `2ca4bc914` |
| C2 | Preserve expert sharding | done | `agent/impl-c12d4-muon` | `efb1a410a` |
| C3 | Non-expert FSDP over `(data,expert)` | done | `agent/impl-c34a2-epshard` | `7f159f441` |
| C4 | Expert-axis batch spec at dispatch | done | `agent/impl-c34a2-epshard` | `476bfe496` |
| D1 | Fixed-capacity `lax.all_to_all` | done | `agent/impl-d123-a2a` | `67121ea3c` |
| D2 | Gather dispatch | done | `agent/impl-d123-a2a` | `89b9ede2d` |
| D3 | Custom scatter-add adjoint | done | `agent/impl-d123-a2a` | `38ab51f25` |
| D4 | Padded stack-sharded Muon | done | `agent/impl-c12d4-muon` | `63eb97c7d` |
| D5 | QuACK grouped wgrad | not started | — | blocked on an EP64 measurement, not on code |
| E1 | QB routing default | **struck** | — | already on `main` since April (#4084, #4458) |
| E2 | Same-step spill | done | `agent/impl-e2-spill` | `2e5976720` |
| F1 | Host offload of optimizer state | done | `agent/impl-e1f1-qb` | `662fc6db4` |
| — | FA4 CUTLASS 4.6 migration | done | `agent/deri-fa4-slot` | `19463a5c3` |

Assembly order constraints, none of which have been applied yet: A0 first; B1 → B2;
C2 → D4; D1 → E2 → D2/D3; E2 at or before the Phase D commits. **B2 must not land without
A1's chunk-drop fix**, or it introduces a backend that drops while reporting zero.

## Derisking

| # | Experiment | Outcome |
|---|---|---|
| D-1 | FSDP baseline drop rates | **Done. `<3%` clause fired.** 0.230% (d6144) and 0.000% (d5120). The EP line's fidelity advantage does not exist. D-1c added trio-on at 318,711 tok/s / 20.575%. The 19.17% baseline was **two-rack** and is not comparable to single-rack numbers. |
| D-2 | Composed EP64 stack | **Compile gate passed both SYRK arms**, zero `spmd_partitioner.cc:668` warnings. Composed 350-step draw running; prediction ~22.5%. |
| D-3 | Leg-batching contradiction | **Resolved.** Patch recovered from Iris bundle `0483b2f2…` and `98737aecf`. The two measurements tested different changes. Neither sign supported; 25.39% stays barred. |
| D-4 | Multi-rack EP64 | Not started. Behind D-2. Still the largest unquantified schedule risk. |
| D-5 | Overlap-limit census | **Done.** 12 MoE all-to-alls at every limit; SYNC census `4,0,0,0` at limits 1/2/4/8. Limit 4 clears all. Prize repriced to ~+0.1pp. |
| D-6a | GatedNorm/attn-gate/XSA trio | **Done, negative.** Two draws, 317,253 tok/s mean against 321,670 control, 1.37% below, non-overlapping bands. Trio is unconditional on `main`. |
| D-6b | Muon shape-grouping | Not started. +0.09pp on FSDP, below the 2pp draw threshold. |
| D-7 | Drop residual beyond the controller | Control band established; spill sweep at m=5 running (trio-off, so absolutes are pre-standardization). |
| D-8 | EP32 after the NS fix | **Done.** OOM prediction falsified: 120 steps, 276.1K tok/s, 17.82% MFU, 11.42% drops. Diagnostic only; EP32 is not a candidate operating point. |
| D-9/D-10/D-11 | Precision | Blocked on the Tier-5 precondition. No rack time. |
| D-12 | `all_but_moe` fit at production shape | **Blocked.** No ref combines the candidate graph, EP64, the homogeneous scan and JAX 0.11; #7489 rejects EP64. Needs a port plus gradient parity before the probe. |

## Decisions taken

- **Trio-on is standard** for all arms. `main` runs GatedNorm, XSA and the attention gate
  unconditionally, so trio-on is what production uses. The trio-on band (317,253 tok/s)
  replaces the trio-off control (321,670) as the baseline; do not compare across them.
- **Capacity factor pinned to 1.0** as the canonical default.
- **`--max-retries 50`** on every rack submission.
- **Federated submission** via `--cluster=marin --target-cluster` is the route; a parked
  job is a bug to report, not to route around.

## Corrections to the record made today

Listed so they are not re-derived. Each was believed, then disproved.

- E1 was "the sharpest item in the series". It was already on `main` since April.
- D-3's patch "was never committed". It survives as a verified Iris bundle and in
  `98737aecf`.
- The bad GB200 nodes were "not the image — Running and Failed pods share digests". They
  were exactly the image; the comparison was made at the wrong level. Node-level
  `.status.images` showed 3 of 208 nodes holding a stale amd64 `iris-task`. See
  [`.agents/ops/2026-07-29-gb200-stale-amd64-image-cache.md`](../../ops/2026-07-29-gb200-stale-amd64-image-cache.md).
- The trio was thought to explain the 1.88pp gap to the 19.2% baseline. It explains ~27%;
  the rest is a rack-count mismatch.
- C3's spec was "a strict generalisation at EP1". False on a mesh with no expert axis.

## Defects found in source commits

- **E2 / `1224ccb02`** reuses the *displaced* expert's combine weight for a spilled
  assignment and does not cap attempts at `top_k − 1`. Corrected in the extraction. The
  recorded E2 fidelity result was measured on the buggy path.
- **B2** introduces an expert-dimension chunker that drops while returning zero. Fixed by
  A1's follow-up commit on the same branch.
- **`deepep`** returns a structural zero for drops with no clipping counter, and cannot be
  exercised on CPU. If its capacity assumption is ever violated it reports a clean run
  while dropping. Still open.

## Open questions

1. **FSDP trio-on, QB-on, one rack at d5120 8-of-256.** The only arm that makes the
   EP-vs-FSDP comparison matched. Currently FSDP (QB-off) is 0.46% ahead on tok/s while EP
   pays a QB penalty worth up to 1.44pp. One 120-step job.
2. **Reconcile the trio's cost with Larry Dial's estimate.** He put it at "2 MFU or so";
   measured it is 0.503pp on FSDP and 0.285pp on EP.
3. **Whether the FSDP comparison figures were single- or dual-rack**, and their QB state.
   Both branches gate the trio behind env vars defaulting off, so the code cannot settle
   it.
4. **Series assembly.** Nothing is merged; the ordering constraints above are unapplied.

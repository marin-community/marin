# Grug MoE MFU optimization — NVIDIA-directed, profile-guided (8×B200)

**Goal:** maximize MFU for the grug MoE model on 8×B200 (Schmidt JHU cluster), driven by
NVIDIA's directions, using xprof profiling to guide each step. Deliver two maxed configs:
1. **EP1** — no expert parallelism (pure FSDP/data over 8 GPUs).
2. **EP-max** — maximum expert parallelism (expert_axis_size=8 → mesh `(1,1,8,1)`).

Branch: work off `rav-moe-repro-v1` (validated baseline **17.88%** = d5120 / `sonic_cute` / EP1).
Harness: `experiments/grug/moe/standalone/grug_moe_mfu.py`. Document findings in issue #7012 as we go.

---

## Baseline (established)
- d5120 / `sonic_cute` / EP1 / 8×B200 = **17.88% MFU** (3218 TFLOP/s, 55.9k tok/s, 9.39 s/step).
- Stack: JAX 0.10.1 (CUDA13), stock XLA (no flags), `quack-kernels` 0.5.0, `nvidia-cutlass-dsl` 4.5.2.
- MFU accounting is honest & backend-independent: `mfu = 3·lm_flops_per_token(active experts)·seq·batch / dur / (num_gpus·2.25e15)`. Numerator fixed at config time → sonic↔sonic_cute↔EP comparisons are apples-to-apples (only `dur` changes). Keep `--num-gpus` = real device count (it only scales the peak denominator).

## The NVIDIA levers → our findings
| lever | what it targets | our finding / gating |
|---|---|---|
| `jax.ragged_dot` + `xla_gpu_experimental_use_ragged_dot_fusion=ON` | XLA-cuDNN ragged GEMM (fprop/dgrad ragged, wgrad batched-dense) | Only bites if `RAGGED_DOT_IMPL=xla` (else Triton is used) **and** XLA is built with the cuDNN path. Our pip XLA is unknown → test empirically; NGC 26.06 (JAX 0.10.1, XLA 9b63591) is the drop-in if pip lacks it. Helps `sonic` GEMMs + `sonic_cute`'s 2 wgrad GEMMs (which stay on XLA ragged_dot). |
| TransformerEngine MoE / NCCL_EP (PR #3036) | expert-parallel dispatch/combine, no external dep (≈DeepEP) | Merged → **TE 2.17** (`transformer_engine.jax.ep`: `ep_bootstrap`/`ep_dispatch`/`ep_combine`/`EpHandle`; `ncclEpDispatch/Combine` XLA prims, CUDA-graph). Hopper+ (B200 ok). Not in any NGC yet (26.06 has TE 2.16) → build/install TE 2.17+. New 4th EP backend to add + benchmark vs ring/ragged_a2a/deepep. |
| 8k seq-len limit removal (raises local batch) | larger local batch under FSDP | NVIDIA WIP, not shipped. FSDP reshards activations through a replicated intermediate → peak mem ∝ batch·seq (512×4096 OOMs at 90B). Track; not actionable yet. Our bench is 128×4096. |

## Methodology (rigor)
- **Profile every config**: `jax.profiler.start_trace` at `step==warmup`, `stop_trace` after loop → xplane in `--output-dir`. Analyze op/kernel breakdown (GEMM vs attention vs combine vs collectives vs bubbles). Use it to pick the next lever — do not guess.
- Warm the JAX persistent cache per distinct script/flag set (separate compile from runtime).
- Steady-state median over post-warmup steps (existing). One variable per run. A/B on identical hardware (8×B200, backfill 8-CPU/90G, `--comment=accept_cost`).
- Keep MFU accounting untouched; log FLOPs numerator to prove it's constant across variants.

## Phase 0 — instrumentation + baseline + profiling (no new deps)
Code changes to `grug_moe_mfu.py` (opt-in flags, baseline behavior preserved):
- `--expert-axis-size N` (default 1) → replace literal at `:2098` `compact_grug_mesh(expert_axis_size=N, replica_axis_size=1)`.
- `--profile` → wrap the measured loop (`:2108-2128`) in `jax.profiler.start_trace(out/"xprof")` gated on `step==warmup_steps`, `stop_trace()` after; `StepTraceAnnotation("train", step_num=step)` around `train_step` (`:2111`).
- Document that `XLA_FLAGS` / `RAGGED_DOT_IMPL` must be **exported by the launcher** (before `import jax` at `:19`); add them to the sbatch, not `main()`.
- Small xprof analysis helper (offline): read `xplane.pb`, dump top ops by device time.

Runs (8×B200, profiled): EP1 × {`sonic_cute` (baseline), `sonic` (Triton default), `scatter`}. Deliverable: baseline table + profile breakdown → **the bottleneck** (expected: expert GEMMs dominate; confirm attention/combine/overhead share).

## Phase 1 — EP1-max (profile-guided)
- **Lever A (ragged_dot fusion):** `RAGGED_DOT_IMPL=xla` + `XLA_FLAGS=--xla_gpu_experimental_use_ragged_dot_fusion=true` on `sonic` and on `sonic_cute` (its wgrad GEMMs). Profile; compare vs Triton and vs stock. If our pip XLA doesn't honor it → escalate to NGC 26.06 container (§Env).
- **Lever B (XLA flags sweep):** async collectives, latency hiding, `--xla_gpu_...` autotuning knobs surfaced by the profile bubbles.
- **Lever C (kernel/dtype):** anything the profile flags (combine scatter-add cost in `sonic_cute`, attention share, remat).
- Deliverable: **EP1-max config + MFU + profile evidence**.

## Phase 2 — EP-max (profile-guided)
- Enable EP: `--expert-axis-size 8`, `--moe-implementation {ring,ragged_all_to_all,deepep}`, `--num-experts 64` (64%8=0 → 8 experts/GPU), batch%8. Mesh `(1,1,8,1)`.
- Baseline the 3 existing EP backends at EP {2,4,8}; profile the all-to-all/all-gather overlap with compute (ring all-gathers all tokens → memory ∝ ep; ragged_a2a moves only real tokens → more scalable; deepep intranode NVLink). Watch capacity drops (cf 1.25).
- **Lever (NCCL_EP):** install TE 2.17+, add `ep_nccl` backend wrapping `transformer_engine.jax.ep.ep_dispatch/ep_combine` (mirror `ep_deepep.py`'s shard_map-over-"expert" interface), wire into `grug_moe.py` EP branch. Profile vs deepep/ring. CUDA-graph capture is the expected win (dispatch/combine overlap).
- Deliverable: **EP-max config + MFU + profile evidence** (+ EP-degree sweep curve).

## Phase 3 — synthesis
Two best configs with MFU + profiles + the winning knobs; post consolidated results to #7012.

## Environment strategy
Start **incremental on the current pip venv** (cheap flag test first). Escalate only when a lever needs a newer stack:
- ragged_dot fusion not honored by pip XLA → **NGC 26.06** container (JAX 0.10.1 matches exactly; tuned XLA + TE 2.16). Requires cluster container runtime — check enroot/pyxis/apptainer on SLURM.
- NCCL_EP → **TE 2.17+** via pip/build into the (venv or container) env.
Decision points get documented in #7012 as we hit them.

## Open questions / risks
- Does pip jaxlib 0.10.1 honor `xla_gpu_experimental_use_ragged_dot_fusion` at all? (empirical, Phase 1A)
- Cluster container support for NGC (enroot/pyxis/apptainer)? (SSH check)
- TE 2.17 build against the cluster CUDA13/NCCL — pip wheel vs source build.
- EP correctness: capacity drops differ per backend (deepep silently over/underflows a fixed buffer); validate loss sanity, not just speed.
- Compute budget: each 8×B200 run ≈ a few credits; sweep breadth TBD with user.

## Injection-point reference (verified)
- Mesh/EP: `sharding.py:147` `compact_grug_mesh`; harness `:2098`, `:2101` `set_mesh`.
- sonic GEMMs: `_moe/sonic.py:332,336` `ragged_dot` → haliax `ragged_dot.py:317` XLA path (forced by `RAGGED_DOT_IMPL=xla`).
- sonic_cute: `_moe/sonic_cute.py:51-52` QuACK fwd, `:60,70` XLA wgrad; `quack_moe_cute.py:115,166` `cutlass_call`.
- EP backends: `_moe/{ep_ring,ep_ragged_all_to_all,ep_deepep}.py`; dispatch in `grug_moe.py:201-289`.
- MFU: `flop_utils.py:5-37`; harness `:1624-1649`, `:2097,2116-2124`.
- Profiling: none live today; loop `:2108-2128`, timing `:2110-2114`.

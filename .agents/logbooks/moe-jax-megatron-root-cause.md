# JAX vs Megatron MoE GPU Gap: Research Logbook

## Scope
- Goal: identify the concrete root causes behind the discrepancy between the negative JAX custom-call result in `#3665` and the more positive torch/Megatron results in `#3641` and `#3666`.
- Primary metric(s):
  - matched H100x8 forward/backward wall time
  - phase-level timing deltas where possible
  - attributable delta by hypothesis category
- Constraints:
  - Use CoreWeave Iris H100x8 via `~/llms/cw_ops_guide.md`.
  - Keep the experiment falsifiable: matched comparisons first, wider tuning second.
  - Include a direct JAX/Marin vs Megatron-LM head-to-head.
- Experiment issue: https://github.com/marin-community/marin/issues/3677

## Baseline
- Date: 2026-03-14
- Code refs:
  - `#3641` seal: `moe-deepep-hybrid-ep-seal-20260314`
  - `#3665` seal: `moe-deepep-jax-layout-ffi-h100-matrix-20260314`
  - `#3666` seal: `moe-megatron-qwen-scale-h100-matrix-20260314`
- Baseline numbers:
  - `#3665` fixed-shape H100x8:
    - `current` beat `deepep_layout_ragged_a2a` by about `1.48x` to `1.81x` on distributed cells
    - `deepep_layout_ragged_a2a` stayed roughly tied with `ragged_a2a`
  - `#3641` fixed-shape torch-side H100x8:
    - patched `hybrid_ep` / `hybrid_ep_permute` beat `deep_ep` on the replicated `runs` slice
  - `#3666` Megatron full-layer H100x8:
    - `deepep` won most `128`-expert Qwen-like points
    - `hybridep` won the authoritative `32` and `64` expert reruns

## Initial Hypotheses
- The dominant root cause is kernel coverage: `#3665` only replaced `get_dispatch_layout`, while `#3641` / `#3666` benefited from real dispatch/combine transport kernels.
- The next root cause is benchmark scope: `#3665` timed the sealed `#3633` fixed-shape path, while `#3666` timed a full `MoELayer` with grouped GEMM and different compute/communication balance.
- JAX-specific overheads may matter, but only after the first two causes are controlled for; otherwise they are easy to overstate.
- A direct same-shape torch dispatch/combine benchmark with optional JAX->Torch bridging should help isolate whether the missing gains are mostly in transport kernels or mostly in the broader Megatron methodology.

## Stop Criteria
- Produce a direct JAX/Marin vs Megatron-LM head-to-head on CoreWeave H100x8.
- Produce a root-cause table that identifies which differences are necessary and/or sufficient to explain the gap.
- Update the issue body with a concise conclusion that is more specific than "JAX is slower" or "Megatron is better."

## Experiment Log
### 2026-03-14 21:05 - Kick off the new root-cause thread
- Commands:
  ```bash
  git worktree add -b research/moe-jax-megatron-root-cause \
    /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    moe-megatron-qwen-scale-h100-matrix-20260314

  sed -n '1,220p' /Users/romain/dev/marin/.agents/skills/agent-research/SKILL.md
  sed -n '1,260p' /Users/romain/llms/cw_ops_guide.md
  gh issue view 3641 --repo marin-community/marin --json title,body,state,comments --comments
  gh issue view 3665 --repo marin-community/marin --json title,body,state,comments --comments
  gh issue view 3666 --repo marin-community/marin --json title,body,state,comments --comments
  ```
- Config:
  - branch: `research/moe-jax-megatron-root-cause`
  - worktree: `/Users/romain/marin-wt/moe-jax-megatron-root-cause`
  - CoreWeave kubeconfig: `~/.kube/coreweave-iris`
- Result:
  - Reloaded the `agent-research` workflow and the CoreWeave ops guide.
  - Re-established the three relevant sealed baselines:
    - torch-side DeepEP / Hybrid-EP wins in `#3641`
    - JAX layout-only negative result in `#3665`
    - Megatron full-layer ranking shift in `#3666`
  - Decided to structure the new thread around root-cause separation rather than another one-off benchmark.
- Initial root-cause matrix:
  - `methodology`: dispatch-only vs full `MoELayer`
  - `kernel coverage`: layout-only vs dispatch/combine transport
  - `backend`: JAX `ragged_all_to_all` / current ring EP vs DeepEP / Hybrid-EP
  - `interop`: optional JAX -> Torch bridge cost
  - `timing hygiene`: dummy GEMM, warmup length, fixed inputs, router dtype
- Next action:
  - Create the experiment issue and link it back here.
  - Audit the relevant JAX / torch / Megatron code paths before launching the first H100x8 comparison.

### 2026-03-14 22:00 - Matched same-shape JAX vs Megatron head-to-head
- Commands:
  ```bash
  export KUBECONFIG=~/.kube/coreweave-iris

  uv run python .agents/scripts/deepep_jax_krt_bench.py \
    --repo-ref 4807bb3f3dbfe654977c978f57900a297cc421f5 \
    --task-id jax-megatron-root-cause-jax-forward-20260314-1445 \
    --skip-smoke \
    --shared-expert-dim 0 \
    --bench-pass forward \
    --warmup 5 \
    --iters 20 \
    --topk-list 2,8 \
    --distributions random,runs \
    --kernels current,ragged_a2a,deepep_layout_ragged_a2a \
    --ep-list 1,2,4,8

  uv run python .agents/scripts/megatron_qwen_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --task-id jax-megatron-root-cause-megatron-20260314-1458 \
    --cases marin_3633_topk_2,marin_3633_topk_8 \
    --dispatchers alltoall,deepep,hybridep \
    --warmup-iters 5 \
    --measure-iters 20
  ```
- Result:
  - JAX/Marin fixed-shape forward-only control stayed negative even after removing the shared-expert branch and using longer timing windows.
  - On distributed JAX cells, `current / ragged_a2a` ranged from `1.73x` to `2.48x`.
  - `deepep_layout_ragged_a2a` stayed effectively tied with `ragged_a2a`; most cells were within `~1%`, worst-case about `+4.2%`.
  - Same-shape Megatron `MoELayer` runs were positive on both `topk=2` and `topk=8`:
    - `topk=2`: `deepep` `1.89x` forward / `3.26x` backward faster than `alltoall`; `hybridep` `2.32x` forward / `2.35x` backward faster
    - `topk=8`: `deepep` `2.33x` forward / `3.39x` backward faster than `alltoall`; `hybridep` `1.82x` forward / `2.00x` backward faster
  - So the cross-framework discrepancy reproduces on the same H100x8 shape; it is not just a reporting artifact from unrelated model scales.
- Additional signal:
  - JAX emitted repeated `gemm_fusion_autotuner` slow-kernel warnings on the large grouped expert matmuls for the `topk=8` slices.
  - This looks like an absolute JAX throughput factor, but it is not needed to explain the relative `deepep_layout_ragged_a2a ~= ragged_a2a` result.
- Next action:
  - Measure raw DeepEP dispatch/combine on the same shape and separate:
    - layout cost
    - steady-state transport cost
    - JAX -> Torch bridge cost
    - Torch -> JAX bridge cost

### 2026-03-14 22:20 - Direct DeepEP dispatch/combine isolation with explicit JAX bridge
- Commands:
  ```bash
  export KUBECONFIG=~/.kube/coreweave-iris

  uv run python .agents/scripts/deepep_dispatch_krt_bench.py \
    --worktree /Users/romain/marin-wt/moe-jax-megatron-root-cause \
    --task-id jax-megatron-root-cause-dispatch-matrix-jax-20260314-1517 \
    --topk-list 2,8 \
    --distributions random,runs \
    --input-sources torch,jax \
    --return-to-jax \
    --warmup 5 \
    --iters 20
  ```
- Launcher/debug notes:
  - First dispatch attempt failed because `Buffer.dispatch` was called with `topk_idx` as the second positional argument, which DeepEP interprets as a cached `handle`.
  - Fixed the harness to use keyword arguments for the non-cached path and a real cached `handle` for the steady-state path.
  - Second attempt failed because the PyTorch image did not include JAX.
  - Fixed the launcher to install `jax[cuda12]==0.8.0` only when the matrix includes `input_source=jax`.
- Result:
  - Raw torch-side DeepEP dispatch/combine on the sealed shape is very fast:
    - `random, topk=2`: `full_s=0.000486` (`67.44M tokens/s`)
    - `random, topk=8`: `full_s=0.001062` (`30.85M tokens/s`)
    - `runs, topk=2`: `full_s=0.000913` (`35.90M tokens/s`)
    - `runs, topk=8`: `full_s=0.001298` (`25.25M tokens/s`)
  - Once tensors are already in Torch, the steady-state dispatch/combine cost for JAX-originating tensors is in the same ballpark:
    - `random, topk=2`: JAX/Torch `full_s` ratio `1.02x`
    - `random, topk=8`: JAX/Torch `full_s` ratio `1.36x`
    - `runs, topk=2`: JAX/Torch `full_s` ratio `0.96x`
    - `runs, topk=8`: JAX/Torch `full_s` ratio `0.92x`
  - The bridge itself is the expensive part:
    - `bridge_to_torch_s`: about `85 ms` to `105 ms`
    - `bridge_to_jax_s`: about `2 ms` to `12 ms`
  - Therefore:
    - a direct JAX -> Torch DLPack bridge every training step would dominate the sub-millisecond to low-millisecond transport kernel time
    - but the raw DeepEP transport kernel is not inherently incompatible with JAX-shaped inputs once those tensors are already on the Torch side
- Root-cause update:
  - This confirms that the missing JAX gains are primarily about missing transport kernel coverage, not about DeepEP transport itself being bad on the sealed shape.
  - It also shows that a naive per-step Torch interop path is not a viable substitute for a real JAX custom call, because the bridge cost would erase the gain.

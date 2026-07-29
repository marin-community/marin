# D-12: `all_but_moe` production-shape memory probe

## TL;DR

The requested AOT comparison is blocked at the code-provenance gate. No job was
submitted, so there are no defensible `recompute_all` or `all_but_moe` peak
arena figures and no headroom figure.

The original `remat_oom_probe.py` is recoverable byte-for-byte, but no Git ref
contains the candidate graph, the EP64 implementation, the homogeneous scan,
and the JAX 0.11 production stack together. Running the recovered probe from an
available ref would either reject EP64 or compile a different graph. The
minimum faithful next step is a candidate port plus gradient parity coverage,
followed by the two compile-only arms.

## Preregistered prediction

Before inspecting or attempting a D-12 result, the prediction was:

- `all_but_moe`: about 80 GiB of temporary arena, with a 60-110 GiB plausible
  range;
- `recompute_all`: about 40 GiB;
- falsification: an `all_but_moe` temporary arena above the approximately
  186 GiB of usable GB200 HBM reported by the prior stack;
- operational warning: a result above 150 GiB would count as a technical fit
  with poor margin.

The estimate treats the two recorded irreducible `gu` and `out_dispatch` pins
as roughly 30 GiB together after applying the 0.25 token factor and the 48/26
layer factor. It reserves another roughly 50 GiB for hidden/MLP inputs and
other temporaries. This prediction remains untested.

## Recovered probe

The exact probe used for B200MFU-013 jobs 3264/3265 is present at:

```
/home/marin/.claude/jobs/6fc274da/tmp/remat_oom_probe.py
```

Its SHA-256 is
`a8a1613ad0399b95a393744e6ccb71bf83ed774a26e8ce4850c1ac8cdf2a96fc`.
It lowers and compiles the real `_make_train_step`, reports
`compiled.memory_analysis()`, then scans the optimized HLO text for distinct
tensor shapes of at least 1 GiB. That is the required measurement. The
production-shape parameter changes are mechanical: d5120, 48 layers,
intermediate 1280, 256 experts, top-8, batch 1024, sequence length 4096, and
an expert-axis size of 64.

The probe was never committed. `git log --all -- '*remat_oom_probe.py'` returns
no commit. The retained job directory is the only local copy found.

## Provenance blocker

| Ref | Candidate split and slim residuals | EP64 candidate path | Homogeneous scan | JAX 0.11 production stack |
|---|---|---|---|---|
| `origin/mcwitt/sonic-cute-wgrad@2949be3bb` (#7489 head) | yes | no | model support exists | no |
| `mcwitt/moe-standalone-ep@59e5fe25f` | standalone-harness port | yes, `ring_cute` / `ragged_all_to_all_cute` | not the requested training-step lineage | no |
| `mcwitt/fsdp-drop-metric@eaa408f48` | no | production launcher only | yes, descends from `97b53fe0e` | no, GPU lock is JAX 0.10.1 |
| `research/mcwitt/7407-jax011-verify@761c03d34` | no | no extracted candidate | yes, descends from `97b53fe0e` | yes |
| `origin/main@6ce4a7e68` | no | no extracted candidate | no `SCALE_SCAN_LAYERS` feature | yes |

The #7489 head contains `all_but_moe` and the slim
`sonic_cute._expert_mlp` custom VJP. That VJP is a local-MoE backend. With
`expert_axis_size=64`, `MoEExpertMlp` rejects `sonic_cute`: local
implementations do not provide expert-parallel dispatch/combine. Changing the
probe to `ring` or `ragged_all_to_all` makes it compile an EP graph without
the slim Sonic CuTe residuals, so it no longer tests #7489.

Commit `59e5fe25f` is the validated manual EP port behind B200MFU-029. It routes
the slim residual hints through the standalone benchmark's `_cute` EP
backends. Its `all_but_moe` implementation lives in
`experiments/grug/moe/standalone/grug_moe_mfu.py`, not in the requested
`experiments.grug.moe.model` training step. It also predates the JAX 0.11
production line.

The JAX 0.11 scan line contains neither part of the candidate. A trial
cherry-pick of `01b8e7c92` onto `761c03d34` conflicts across the `Block` API.
The production block now includes SConv branches, hoisted chunk-0 expert
gathers, MTP outputs, and drop telemetry. Resolving that conflict by choosing
either side would change the graph under test. This is the bit-rot condition
called out in the D-12 assignment.

## Results

| Arm | Peak temporary arena | Largest individual buffers | Headroom |
|---|---:|---|---:|
| `recompute_all` | not measured | not measured | not measured |
| `all_but_moe` | not measured | not measured | not measured |

No Iris job ID or submitted command exists. Submitting the baseline alone
would not resolve the candidate blocker and would leave an unusable
one-sided comparison.

## Work required before submission

1. Port `01b8e7c92` and the axis-type-aware residual storage fix
   `868d9d7e4` to the production EP expert-MLP seam. Preserve the validated
   `x_dispatch` re-gather, elementwise `h` reconstruction from `gu`, and
   sharded expert-weight residuals under the `_cute` EP backend.
2. Implement the `all_but_moe` boundary in the current `Block.__call__`
   without dropping SConv, hoisted-gather, MTP, or drop-metric behavior.
3. Add value and full-gradient parity against `recompute_all` on the
   homogeneous scan before using the port for an AOT memory decision.
4. Parameterize only the recovered probe's shape and distributed-array
   construction. Keep `_make_train_step`, `initial_state`,
   `lower(...).compile().memory_analysis()`, and the optimized-HLO census
   unchanged.
5. Run the two compile-only EP64 arms from the same commit with
   `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`,
   `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`,
   and
   `--xla_gpu_experimental_parallel_collective_overlap_limit=4`. Use the
   direct `cw-us-east-08a` route, default priority, user `mwittmann`, and
   `--max-retries 50`. Do not execute a training step.

## Adoption implication

This report does not establish that #7489 fits at the production operating
point. The desk estimate still predicts a fit, but it has no measured
confirmation.

Adoption remains a package deal. Slim residuals alone measured -0.28
percentage points under `recompute_all` because backward pays the re-gathers
without receiving the memory dividend. The residual change must not land
ahead of a validated remat mode.

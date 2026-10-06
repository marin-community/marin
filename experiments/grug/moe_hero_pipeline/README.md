# Hero pipeline variant

`pipeline.py` adapts `moe_pipeline` to the model in [`moe_hero_ep`](../moe_hero_ep/README.md), preserving
latent/shared experts, SConv, absolute local/global attention positions, and QB
router updates. Standard 1F1B is the default. Each process initializes only its
own stage. CPU tests compare sequential-stage hidden states, routing statistics,
losses, and gradients against the unsplit model.

The synthetic runner does not restore or save checkpoints. `--main-hero-recipe`
uses FP32 parameters, BF16 compute and compute-scaled MuonH; it requires
`--optimizer muonh` and an explicit `--processes-per-task` count.
The legacy AdamW default is a bring-up configuration. The historical long-context
recipe explicitly selects BF16 MuonH and both host offloads. Production
continuation and long-run training stability remain unvalidated.

The current optimizer normalizes each layer's routed expert bank together, matching
the stacked EP model. Set `--expert-normalization per_expert` for a comparison;
the shared `GrugMoeMuonHConfig.expert_normalization` selects this choice.
The historical hardware runs used the earlier per-expert behavior; they do not
validate this normalization correction. The current optimizer and frozen-QB
options have CPU numerical/state-transition coverage.

QB bias adapts after each update by default. Set `--qb-bias-mode frozen` to
retain the current pending bias while continuing to train expert and router weights. This controls the QB bias
only; it does not freeze the router weights. Frozen QB has CPU state-transition
coverage but has not been evaluated for SFT or RL training quality.

## Historical validated result

The full 535,477,106,688-parameter, 48-layer model completed ten finite synthetic
updates at sequence length 65,536 on 192 H100s in `cw-rno2a`. The recipe uses
PP24/EP8, one process per eight-GPU worker, batch 384, 48 microbatches, six expert
transport waves, and FA4 (`gpu_fa4_cute`). No FP8 was needed.

| Schedule | Median reported step time | Median MFU | Validation |
| --- | ---: | ---: | --- |
| Standard 1F1B | 174.432 s | 15.523% | Ten finite updates, all 24 workers exited successfully |
| DualPipeV with backward-task waits | 550.704 s | 4.917% | Ten finite updates, all 24 workers exited successfully; extra recomputation/offload and telemetry enabled |

Keep 1F1B as the working recipe. The reported timer excludes some global loss
collection and inter-step overhead. These are systems measurements from fresh
initialization with synthetic data, not training-quality results. The complete
scaling table, negative results, dependency findings, and W&B links are in
[experiment #9277](https://github.com/marin-community/marin/issues/9277).

PP24 gives two transformer layers per stage, reducing the parameters and
optimizer state owned by each worker. EP8 keeps expert communication within
an eight-GPU worker's NVLink fabric. CP1 was the initial working integration;
this layout was chosen to fit the model and then fill the 1F1B pipeline, rather
than through a matched comparison with PP8/EP32/CP4. More pipeline stages also
increase the number of microbatches needed to amortize pipeline fill and drain.
The subsequent PP24/EP4/CP2 diagnostic completed ten updates at batch 96,
68.539 seconds/update and 9.877% MFU. It reduced tokens per GPU and tokens per
update; differing batch sizes and attention-helper versions prevent an isolated
CP comparison. No matched Megatron layout benchmark was performed.

## Historical runtime requirements

The measured runs used JAXPP `46b8443eed01f54688dd24ae451203a504b1cff8` with
local overlays. A stock installation of that pin is insufficient. Required
pieces are:

- Preserve Host/Device memory targets during mesh rebinding
  ([NVIDIA/jaxpp#13](https://github.com/NVIDIA/jaxpp/issues/13)).
- Keep activation D2H copies in the forward partition and preserve pinned-host
  intermediate shardings. This saved 12 GiB/device at eight outstanding
  microbatches in the 24-layer diagnostic.
- Include the Dime2 send-buffer lifetime fix from upstream
  [78458f8](https://github.com/NVIDIA/jaxpp/commit/78458f8b258f0fd467be13b4b8e7056a853e320b).
- Precompile and warm local tasks with disposable inputs before pipeline
  transfers; initialize adjacent communication channels before stepping.
- Park real device state on host during disposable warmup, then restore its
  original values and shardings.

The runner retains `--synchronize-devices-after-step`: two unsynchronized full
m8 trials produced NaNs, while synchronized trials passed. This is a timing
workaround with an unresolved root cause. Do not remove it from the validated
recipe based only on successful small tests.

Use the `pipeline` dependency environment and CUDA toolchain. After applying
the dependency overlays, reproduce the worker configuration with:

```bash
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_CLIENT_MEM_FRACTION=0.91 \
JAX_ENABLE_PGLE=false XLA_FLAGS=--xla_gpu_enable_command_buffer= \
CUDA_MODULE_LOADING=EAGER JAXPP_DISABLE_SCHEDULE_TASK_FUSION=1 \
uv run --no-sync python -m iris.hooks.multigpu_main --nproc 1 --devices-per-proc 8 -- \
  python -u -m experiments.grug.moe_hero_pipeline.pipeline_smoke --full-hero \
    --schedule standard_1f1b --stages 24 --expert-axis-size 8 \
    --microbatches 48 --batch-size 384 --sequence-length 65536 --expert-waves 6 \
    --optimizer muonh --expert-normalization per_expert \
    --offload-opt-state --offload-activations \
    --park-state-during-warmup --synchronize-devices-after-step \
    --compilation-cache /tmp/hero-pp-jax-cache \
    --steps 10 --run-id <unique-run-id>
```

Iris must supply the distributed coordinator and ranks for 24 eight-GPU workers.
Keep ten steps for matched diagnostics: the configured weight-decay schedule
depends on the requested step count. BFC and larger preallocation trials failed;
retain the allocator settings above when reproducing this result.

The measured source base was `96da2a6cdce401187239846957687fdde22cf8a5` plus
local model/runner/dependency overlays. Launch manifests record exact hashes in
`scratch/hero-h100-20260916/`; `full65k-m48-sync24-evidence.json` retains the
baseline evidence. These ignored artifacts are not a published reproduction
package. Subsequent runner cleanup removed per-leaf finite-state diagnostics
and per-step memory dumps; it has not been benchmarked again on hardware.

[PR #9662](https://github.com/marin-community/marin/pull/9662) extends this
foundation with a GB200 launcher, tracked runtime setup, checkpoint/data support
and bounded main-recipe hardware validation.

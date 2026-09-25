# Hero pipeline variant

`pipeline.py` adapts `moe_pipeline` to the model in [`moe_hero_ep`](../moe_hero_ep/README.md), preserving
latent/shared experts, SConv, absolute local/global attention positions, and QB
router updates. Standard 1F1B is the default. Each process initializes only its
own stage. CPU tests compare sequential-stage hidden states, routing statistics,
losses, and gradients against the unsplit model.

The synthetic runner can save and resume complete pipeline checkpoints. Its
AdamW default is a bring-up configuration; the validated long-context recipe
explicitly selects BF16 MuonH and both host offloads. Production continuation
and long-run training stability remain unvalidated.

Set `--checkpoint-root` to restore the latest complete `step-N` checkpoint and
`--checkpoint-every-steps` to save at that interval and on the final step. Keep
`--steps` at the same total update count when restarting: it sets the MuonH
schedule as well as the stopping point. The checkpoint contains model weights,
optimizer state, and pending QB router updates under global layer paths, so a
run may resume with a different pipeline layer split. The checkpoint root must
be shared by every worker. A checkpoint written by the standard Hero FSDP
trainer has a different state tree, including optional master and EMA weights;
loading that format into this pipeline has not been implemented or validated.

## Validated result

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

## Runtime requirements

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
    --optimizer muonh --offload-opt-state --offload-activations \
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

## Review boundaries

The small Muon expert-stack sharding correction and its numerical regression
can be reviewed independently. The Hero stage adapter, runner, and parity tests
form the Marin pipeline change. JAXPP memory-space rebinding and forward-host
residual placement are separate dependency fixes; explicit startup preparation
needs its own lifecycle API. Resolve these dependencies before presenting the
65K command as a stock-install recipe.

DualPipeV split-rematerialization, delayed-cotangent offload, and per-task waits
remain experimental scratch overlays. They are not required by the validated
1F1B recipe and should stay outside its initial PR. Follow the existing CP work
rather than introducing another context-parallel implementation here.

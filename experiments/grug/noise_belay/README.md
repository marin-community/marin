# Noise on Belay candidate 11: partial implementation and measurements

This experiment records a partial transfer of Noise's Boost → Cut recipe to Marin
Belay candidate 11. It includes forward-loss probes, selected parameter-credit
calculations, correctness checks and component timings. **There is no complete
forward-only training step, Noise-trained Belay checkpoint or treatment loss.**
The native backprop control is complete and retained for future comparisons.

The [HTML report](site/index.html) explains the technical background, measured
results, independent skeptical assessment and the proposed tied-drafter experiment.
[Hosted report](https://storage.googleapis.com/marin-public/benfeuer/noise-belay/2026.10.23/index.html),
published with Marin’s `scripts/ops/publish_site.py` as artifact
`sites/benfeuer/noise-belay`, version `2026.10.23`. All repository additions
are under this experiment directory. `runtime.patch` preserves fixes to the
historical runtime without changing shared library code in the PR.

## Recorded setup and baseline

The [Belay issue](https://github.com/marin-community/marin/issues/9451#issuecomment-5945519263)
selects candidate 11. Its recorded configuration is in
[candidate11-reference.json](candidate11-reference.json), pinned to
`8983bec6fdaeb1cd6d0acff041194d2bf53ab679`. The shape audit counts
1,026,488,855 parameters: six layers of width 512, four KDA layers, two MLA
layers, 512 routed experts with top-eight selection, a shared expert of width
768, a bigram embedding, and a 16,384-token vocabulary. Total stored parameters
are about 27× Noise's 37.7M dense model; active computation does not scale by that ratio.

| Native backprop run | Updates | Training wall time | Paloma macro loss |
| --- | ---: | ---: | ---: |
| [Historical seed 0](https://wandb.ai/marin-community/marin_moe/runs/lc1-b174-ft-kda) | 2,698 | 480 s allowance | 2.971509 |
| [Historical seed 1](https://wandb.ai/marin-community/marin_moe/runs/lc1-b174-ft-kda-s1) | 2,698 | 480 s allowance | 2.975003 |
| [RNO2A replication](records/full-control-v4/result.json), seed 0 | 2,116 | 479.876 s | 3.019722 |

The new run used eight H100s at interactive priority and global batches of
128 × 4,096 tokens. It presented 1,109,393,408 token positions, of which
1,109,122,560 had nonzero next-token loss weight. The first native batch matched
recorded token, loss-mask and document-segment hashes. Full native evaluation
scored all 16 Paloma corpora and seven additional corpora using EMA weights.

The original 2,698-step optimizer schedule was retained. RNO2A completed 582
fewer updates, including fewer late decay and EMA updates. Its loss was 0.048213
nats/token higher than historical seed 0. Cluster, compiler settings and runtime
patches also differ; one run cannot isolate the cause of that gap.

The median synchronized native update took 203.36 ms. The training clock includes
27.75 seconds of data waiting. It also includes optimizer execution, synchronization
and local metric recording. Dataset startup (11.24 s), initialization (16.85 s), compilation
(63.88 s), discarded warmup on state copies (3.60 s), checkpointing (32.69 s),
capture screening (21.47 s) and evaluation (85.47 s) are outside that clock.
The successful pod lifecycle was 757 seconds, or 1.6822 GPU-hours; timed training
used 1.0664 GPU-hours. The largest allocator high-water mark through training was
44.21 GB/GPU, including startup and warmup. The checkpoint in
[checkpoint.json](records/full-control-v4/checkpoint.json) holds EMA evaluation
weights, not optimizer state. [Scalar history](records/full-control-v4/metrics.jsonl.gz).

## Component costs and bounds

The first three rows are synchronized medians of three calls after compilation on eight H100s,
with BF16 compute, FP32 master weights, the full training batch and synthetic
inputs at initialization. They do not measure time to a target quality.

| Component | Measured time per batch | What it excludes |
| --- | ---: | --- |
| Native training-loss forward | 61.532 ms | Backward, optimizer, data loading |
| Native loss forward + backward | 156.900 ms | Optimizer, data loading |
| Twelve Boost readouts | 61.226 s | Capture, parameter reconstruction, Cut, merging, optimizer |
| Hidden-state forward, two final calls of a separate capture diagnostic | 55.14–55.39 ms | Vocabulary loss and all training work |
| Forward returning captured states, same diagnostic | 61.50–61.59 ms | Vocabulary loss and all training work |
| Full-scale reconstruction, Cut, future-token credit, merge | Unmeasured | No complete treatment exists |

Evidence: [matched forward/backward](records/gpu-v11/candidate11-forward.json),
[Boost](records/gpu-v10/candidate11-forward.json), and
[capture](records/gpu-v7/candidate11-forward.json).
The capture diagnostic suggests about 6.3 ms overhead in that setting. It is not
a timing for all subsequent credit work. The 203.36 ms real-data optimizer-step
median comes from a different workload; subtracting the synthetic profile from
it does not isolate optimizer cost.

Each Boost readout uses 512 directions and both signs through the depth mixture,
normalization and 16,384-way vocabulary loss. Twelve readouts therefore perform
12,288 logical vocabulary-loss evaluations per token. They reuse clean activations
and head projections; these are not 12,288 complete model forwards. The implementation
tiles 16 directions and 64 tokens. Despite that reuse, its readouts cost **390.22×**
the measured forward-plus-backward operation, or about 301× the separate median
native optimizer step. At most seven complete readout cycles fit in 480 seconds
before any other treatment work.

With this measured probe cost fixed, even making all other treatment operations
free cannot yield a per-update speedup at equal useful progress. This is a result
about the naive implementation, not an impossibility theorem for forward-only learning.

### The narrow plausible route to a speedup

Let F=61.53 ms be the measured target forward and T=156.90 ms its forward-plus-backward
cost. Replacing one ordinary update with m target forwards and extra cost C has the
optimistic component speedup T/(mF+C). With C=0, one forward permits at most 2.55×,
two permit 1.27×, and three already lose. These omit optimizer and full-system costs.

A practical method needs far fewer probes, useful credit at low cost, and enough
lasting progress to pay for parameter reconstruction, optimizer state, synchronization,
rejected candidates and verification. A cheap tied drafter might rank updates that
replace several hero updates. Four target forwards for unchanged/candidate comparisons
on screening and confirmation batches already cost about 246 ms before proposal work.
Parallel candidate execution can improve latency while increasing GPU-hours.

Forecasting an ordinary loss curve does not establish that a proxy can rank the
effects of candidate parameter updates. Test ranking on fresh batches, compare against
momentum extrapolation and step-size search, and check persistence under further
training. Charge all proposal and verification costs. Sparse experts and fixed
attention/head work can limit draft savings even when stored parameter count shrinks.
Memory savings or a better discrete architecture could change the comparison, but
neither benefit has been demonstrated here. No production speedup is established.

Dust's 1,000–10,000× headline compares population efficiency with EGGROLL weight-space
ES, extrapolating beyond a measured population ladder. It is not a measured wall-clock
speedup over backprop. At 10M tokens, Dust with 64 draws has loss 5.397, EGGROLL at
approximately 16k has 6.033, and backprop has 4.989. Matching EGGROLL populations are
fit-dependent extrapolations. See the [paper](https://qlabs.sh/research/dust) and
[Appendix C–D](https://qlabs.sh/research/dust/appendix.html).

## Implementation and validation limits

Implemented components include independent Boost readouts and reconstruction for
head/final-gain/depth-query, shared and routed expert matrices, conditional router
matrices, latent output expansions and MLP causal convolutions. Router tests hold
expert choices and capacity masks fixed. They leave discrete selection changes untested.
Cut, future-token probes, attention internals, several normalization and embedding
updates, latent input credit and trajectory merging remain incomplete.

The selected source recipe is [Noise Boost → Cut](https://github.com/qlabs-eng/noise/tree/5e0d53fd02fd769d573e07af0d382d313c5ac878/train/val).
Its attention and residual structure differ from Belay's; completing the transfer
requires architecture-specific credit paths, especially for KDA recurrence.

Twenty-four CPU component checks pass on the pinned runtime, including independent
finite-loss/native-gradient references, packed-document boundaries and two-device
expert-parallel cases. Tests use native derivatives only as independent references;
the implemented credit calculations use forward probes and linear operations.
The broader historical affected-suite run was not fully green: 2,543 passed,
17 failed, 13 skipped and five expected failures. Twelve failures reproduced in the
untouched pinned checkout; four timeout cases passed at lower test concurrency;
a dense KMA timeout remains unresolved.
The packaging branch’s current-main affected-suite check produced 3,113 passes,
11 failures, six skips and five expected failures. All 11 failures reproduced in
a separate unmodified `3eb121f2ee` checkout. The packaged pinned-runtime checks
passed 24/24; mechanical lint and the strict MkDocs build passed.

The old elementwise GPU tolerance, |a−b| ≤ 1e−5 + 1e−5|b|, was chosen for this port.
With explicit user authorization, the bounded GPU screen instead requires finite
final states, activation error RMS/reference RMS ≤0.005, absolute mean cross-entropy
change ≤0.001 nats/token, and expert-slot disagreement ≤0.001 in every layer.
CPU mathematical tolerances remain unchanged.

The final screen compares native forward with routing inspection against forward
returning the complete activation capture, using final raw training weights (not
EMA) and the first real training batch. Route disagreement is the fraction of
token/selected-slot positions with differing expert identifiers; slot-order
differences count. This trained-state screen passes the aggregate limits: relative RMS 0.003079,
loss change +0.00000143, and worst-layer route disagreement 0.0005064. However, its
maximum hidden-state difference is **12.078125**, and 1.4516% of elements fail the
old tolerance. Passing does not establish equivalent credit estimates or training
trajectories. [Complete diagnostics](records/full-control-v4/candidate11-validation.json).

`runtime.patch` includes the ungated expert shape correction, ordered FP32 local
expert-output accumulation (fixing observed repeatability), shared compiler
boundaries, no-tail configuration fix, selected routing logits, and a size-one
mesh Muon fix with their focused regression tests. Some changes alter floating-point
execution; the replication is not a bitwise reproduction of the historical run.

## Reproduction

The runtime is historical. Do not apply its patch to current `main` or run the
candidate-specific checks against current model APIs. `prepare.py` creates a new,
detached worktree at the recorded revision, applies `runtime.patch`, copies this
experiment, and records source hashes. It refuses to replace an existing directory
and submits no GPU jobs.

```bash
python3 experiments/grug/noise_belay/prepare.py --checkout /tmp/noise-belay-runtime
cd /tmp/noise-belay-runtime
JAX_PLATFORMS=cpu PYTHONPATH=.:experiments/grug/noise_belay uv run --all-packages --no-default-groups --group test python -m pytest -n 2 -m manual experiments/grug/noise_belay/check_candidate11_forward.py experiments/grug/noise_belay/check_forward_credit.py experiments/grug/noise_belay/check_routed_expert_credit.py -q
```

These specific CPU checks carry `manual` because they require that source-pinned
runtime, not because they use a live cluster. Their `check_*.py` filenames keep
them outside automatic current-main test discovery: marker filtering happens
after imports, which already require historical APIs. The explicit file list
and marker above run all 24 checks in the prepared runtime.

For native data/control, prepare a separate destination instead of reusing the
CPU-only worktree:

```bash
python3 experiments/grug/noise_belay/prepare.py --checkout /tmp/noise-belay-control-runtime --tokenizer /path/to/hero-bpe-v16384
```

The native-control command below must run from `/tmp/noise-belay-control-runtime`.
The tokenizer’s two files must match the hashes in [the data audit](records/data-manifest.json).
Training text, token batches, tokenizer assets and model weights are not bundled.
CPU and synthetic checks need no private data. Native-control reproduction requires
Marin storage access to the training cache and evaluation caches listed under
`config.data` in the recorded configuration, plus the matching hero tokenizer
archive from the Marin maintainers. The tokenizer is not provided by this PR.
Ambient `CW_KEY_ID`/`CW_KEY_SECRET` and read permission on those caches are required.
No launcher reads a personal secrets file.

Run the following from the prepared checkout, using a current Marin checkout for
Iris. The measured job used client revision `f41b27c558222fac6404546280339ce37a0dfa72`
with working RNO2A controller access. Set `IRIS_CLIENT_CHECKOUT`, a unique `RUN_NAME`,
and `CHECKPOINT_URI` explicitly. The latter is a new writable RNO2A S3 output
destination for evaluation weights, not an input checkpoint.
This command incurs another eight-H100 run; it is documentation, not an automatic job.

```bash
uv run --project "$IRIS_CLIENT_CHECKOUT" iris --config "$IRIS_CLIENT_CHECKOUT/lib/iris/config/cw-rno2a.yaml" job run --no-wait --enable-extra-resources --gpu H100x8 --priority interactive --cpu 16 --memory 128GB --disk 100GB --timeout 3600 --max-retries 0 --job-name "$RUN_NAME" --sync-package marin-core --extra gpu --extra dedup -e JAX_PLATFORMS cuda,cpu -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.85 -e PYTHONPATH . -e NCCL_DEBUG WARN -e XLA_FLAGS '--xla_allow_excess_precision=false --xla_gpu_enable_triton_gemm=false' -e CW_KEY_ID "$CW_KEY_ID" -e CW_KEY_SECRET "$CW_KEY_SECRET" -- bash -c 'uv pip install --python .venv/bin/python nvidia-cufile==1.15.1.6 nvidia-cusparselt-cu13==0.8.0 nvidia-curand==10.4.0.35 && exec .venv/bin/python experiments/grug/noise_belay/candidate11_control_run.py --training-seconds 480 --output "$IRIS_OUTPUT_DIR" --checkpoint-uri "$1"' _ "$CHECKPOINT_URI"
```

The full-shape synthetic profiles use `candidate11_gpu_probe.py --batch 128
--sequence 4096 --repeats 3 --output /path/result.json`, with either
`--measure-backward` or `--measure-boost` (see its CLI). Use the same prepared
runtime, compiler flags and eight-GPU allocation. Compilation is outside medians.

Historical manifests identify measured sources. The published copies add packaging,
formatting and test-harness cleanup; no new GPU timing is claimed for those edits.
The [resource ledger](records/gpu-resource-summary.json) accounts for all 29 attempts:
8.8397 GPU-hours as a conservative reservation upper bound, including failures and
startup. Jobs use at most eight GPUs each; one cancellation cleanup briefly overlapped
its replacement, giving a conservative peak reservation of 16. All jobs are terminal.

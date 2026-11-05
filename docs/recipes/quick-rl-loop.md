# Quick RL loop

Use this loop to improve task selection and the RL recipe: curate tasks, publish
a fixed selection, train from the pinned SFT model, and compare evaluation before
and after training.

The current entry points are the
[Snowball selection recipe](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/recipes/snowball-100k.json)
and
[`train_quick_loop.py`](https://github.com/marin-community/marin/blob/main/experiments/post_training/inventory/train_quick_loop.py).
The selection defines the 100k training allocation. The launcher defaults to
25 synchronous RLOO-N updates on 40 H100s, starting from Datakit SFT revision
`fe666478454e79fe6027a0bab0388d9abd68f51a` and its matching tokenizer. It uses
128 prompt groups per update, four samples per group, a 24,576-token context
and an 8,192-token response limit.

## Serving profile and reproducibility

The checked-in inventory recipe enables batch-invariant arithmetic, fixes the
evaluation seed to `20261020`, disables prefix caching, and limits each inference
worker to one scheduled sequence. It retains DP8/EP8/TP1/PP1, chunked prefill
with an 8,192-token scheduling budget, and CUDA graphs. The sequence cap also
applies to training generation and reduces concurrency. Training throughput
has not been measured.

Two runtime profiles have reproduced the fixed 100-task panel. They use different
runtime revisions and produce different answers; their scores are not a
controlled comparison of serving policies.

| Profile | Evidence | Exact agreement | Mean reward | Response tokens per pass |
| --- | --- | --- | --- | --- |
| One scheduled sequence; vLLM `39e62869693c` | Six evaluations across four fresh engine startups: four direct SFT and two Megatron initial evaluations | 100/100 tasks | 0.2002 | 469,450 |
| Concurrent scheduling; corrected vLLM `02872a7e5530` candidate, Torch 2.14.1+cu132 | Four fresh direct-SFT evaluation processes across two eight-H100 jobs on 2026-11-05 | 100/100 tasks | 0.1801 | 503,697 |

Agreement covers prompt and response token IDs and text, grades, verifier
outcomes, recorded operational outcomes, and reported aggregates. Each pass
independently attempted and graded all 100 tasks. Missing grades, errors,
nonfinite values, or any compared-field mismatch fail the check; equal mean
scores alone do not pass it. The one-sequence Megatron evaluations used NCCL
`expert_block` weight transfer and exited before any optimizer step.

### Concurrent DP8/EP8 qualification

The four-pass concurrent qualification used the pinned Datakit SFT checkpoint
and tokenizer above, the original 24,576-token context and 8,192-token reply
limit, greedy decoding with seed `20261020`, and the original task-owned graders.
All 400 results were independently graded, verified and finite. There were zero
task-level or aggregate differences. Evaluation logprobs were not requested.
The worker sequence limit was 1,024; prefix caching stayed off, while TRITON
attention, 8,192-token chunked prefill and compilation/CUDA graphs stayed enabled.

This candidate used vLLM `02872a7e55304d288e52fdb906923551b9517501` plus
the [three IEEE FP32 Triton dot corrections](patches/grug-ieee-fp32-20261105.patch)
in `batch_invariant.py`, with
these native TorchInductor settings:

```yaml
generator:
  engine_init_kwargs:
    compilation_config:
      inductor_compile_config:
        deterministic: true
        batch_invariant: false
        combo_kernels: true
        benchmark_combo_kernel: false
```

`batch_invariant: false` here is TorchInductor's separate reduction-policy
option. vLLM's `VLLM_BATCH_INVARIANT=1` remained enabled. The native CUDA-graph
capture-size defaults, including size 1, were retained. A separate short-prompt
diagnostic observed ModelRunnerV2 executing full and piecewise graphs, including
full size 1; the full scored runs did not collect per-request graph traces.

The numerical source bindings were:

- Qualified native wheel: release
  [`marin-vllm-gpu-20261103-02872a7e5530`](https://github.com/marin-community/vllm/releases/tag/marin-vllm-gpu-20261103-02872a7e5530),
  SHA256 `729cffd23294926b5ddbe1a19485b3da8ad8c582b4ea809ed5350779faa67f83`.
- Candidate `batch_invariant.py`: SHA256
  `2a70d9021f784b27fd409e085093b5bba1e1c0c4a659116bbb7063b647301436`.
- Fixed validation parquet:
  `s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k/2026.10.23.2/validation.parquet`,
  SHA256
  `958b38bedb04294c6523b21f26cf3aa7a9af43038750bbe849318fbce1a872b7`.

The completed qualification is output version `2026.11.05.2` under
`s3://marin-us-east-02a/marin/users/dlwh/evals/fixed100-reproducibility/`,
with `job-{1,2}/pass-{0,1}` for the four independently graded passes. Both jobs
finished with task exit 0 and no failures or preemptions. The strict comparison
joined results by frozen canonical task ID, not completion order, with no grade
cache or reuse.

Earlier full-length evaluations on the corrected vLLM candidate with default
compiler policy matched only 91/100 replies across four fresh processes. Two
scores changed in opposite directions while every mean remained 0.20. A later
qualification attempt stopped at 99/100 verified scores after a Daytona snapshot
metadata lookup returned HTTP 502. That failed attempt was preserved. The clean
four-pass qualification above included a bounded retry for read-only snapshot
metadata GET on HTTP 502/503/504; it did not retry grading, creation POSTs, or
failed tasks, or change startup deadlines. The
[lookup retry patch](patches/daytona-snapshot-get-20261105.patch) applies to Marin
`a2f550d966551876535540063017f85cee01473e`; the resulting `machine.py` SHA256 is
`42218ba65d5eb62e1ef951197c3e743271928c7b064086848c9e53687c90258a`.

These results qualify this finite panel and pinned standalone evaluation
configuration. They do not isolate the cause of the earlier nine reply changes,
prove universal determinism, or qualify training placement and weight transfer
under the concurrent profile. The numerical candidate and lookup retry were
validated on separate source branches. This PR retains its one-sequence recipe
and older runtime pin; its exact head has not been run on GPUs. Adopting the
concurrent candidate requires a separate runtime integration and qualification.

## Prior DP4 router measurements

[Xiaoyuan's measurements](https://github.com/marin-community/marin/pull/10760#issuecomment-6553179601)
on Snowball v0.1 (Grug-67B-A2B, update 14 from
[#10305](https://github.com/marin-community/marin/issues/10305)) attributed most
observed batch variation to the FP32 router. At vLLM `e09cfd55`, its direct
`F.linear` stayed on cuBLAS despite batch-invariant mode. Router logits changed
between batches of one, 2–16, and at least 17 tokens. Moving the router to the
batch-invariant IEEE FP32 kernel improved eager agreement from 23/164 to
161/164 replies with fixed-root expert reduction. Compiled execution reached
162/164 before excluding size-1 CUDA graphs. Both partial results fail an exact
reproducibility check.

On four H100s at DP4, the reported combination produced identical greedy
HumanEval+ replies for all 164 prompts in 13 passes across four launches, at
16 and 64 requests in flight, with different engine placements and request
orders. The report does not specify whether reply equality compared text,
token IDs, or both. The combination was:

- `VLLM_BATCH_INVARIANT=1` and the router in the batch-invariant IEEE FP32 kernel.
- Fixed-root expert-output reduction: reduce on one rank, then scatter, using
  the reduction change described in vLLM `01911be3`.
- `TORCHINDUCTOR_DETERMINISTIC=1`, Inductor's `benchmark_combo_kernel` disabled,
  and `cudagraph_capture_sizes` excluding 1.

The same report identified a separate long-prompt issue in FlashAttention 3's
sliding-window path: chunk starts needed 32-token alignment to match unsplit
prefill, and concurrent requests could move the split. Chunked prefill disabled
kept long-prompt scoring bit-identical under load; the report gives no panel
size or pass count for that check. The HumanEval+ panel above
also matched with chunked prefill and prefix caching both disabled or both at
vLLM's defaults; that result does not cover every long-prompt chunk boundary.

These measurements used a different checkpoint, panel and DP4 setup.
Xiaoyuan had not tested vLLM `01911be3` as a complete revision, this PR's
`39e62869693c` pin, or DP8/EP8. The IEEE FP32 router result does not qualify
`01911be3`'s TF32 router path. The DP8/EP8 qualification above used a newer
runtime and a different combination:
TRITON attention and native size-1 CUDA graphs remained enabled. It does not
retroactively qualify the older pins or reproduce every setting in Xiaoyuan's
DP4 experiment. Neither finite panel establishes universal bitwise determinism.

## Make or update tasks

Declare new tasks through the
[dataset-owned curation pipeline](../references/task-curation.md#dataset-owned-invocation).
Pin their source revision, converter and grader dependencies. The training pool
comes from FULL curation's admitted `final/*.parquet` files; QUICK conversions
and SAMPLE panels do not qualify.

Use the [curation campaign procedure](../references/task-curation.md#run-a-campaign)
to review and verify the sources. For example, inside an Iris coordinator with
the pipeline dependencies and review credentials available:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.driver \
  --mode full --source SOURCE_NAME \
  --review-mode chat --model-revision "$REVIEW_MODEL_REVISION" \
  --review-cache "$CURATION_CACHE" \
  --max-workers 64 --coordinator-memory 16g --concurrent-sources 10 \
  --normalized-shards 32 --image-backend gvisor \
  --worker-image "$WORKER_IMAGE" \
  --report-path "$CAMPAIGN/full.json" \
  --selection experiments/post_training/task_curation/recipes/snowball-100k.json
```

Repeat `--source` for every source in the selection recipe, using its invocation
name from
[`runnable_sources()`](https://github.com/marin-community/marin/blob/main/experiments/post_training/task_curation/sources.py).
For example, recipe ID `MarinSkyRL:eurus2_code` is invoked as
`--source eurus2_code`. The CLI names differ from the stable IDs in the recipe.
Set `REVIEW_MODEL_REVISION` to the pinned reviewer, `WORKER_IMAGE` to a
digest-pinned worker image, and `CURATION_CACHE` and `CAMPAIGN` to retained,
region-local storage prefixes. Build the declared
[grader environments](../references/task-curation.md#run-a-campaign) first.
The command prints a plan; add `--run` to execute it. Successful source artifacts
and exact review requests remain cached across campaigns.

## Publish a selection

If a completed FULL campaign already has the required source artifacts, reuse
its `corpus.json` without rerunning curation:

```bash
uv run --with-editable './lib/taskcompendium[pipeline]' python \
  -m experiments.post_training.task_curation.driver \
  --corpus "$CAMPAIGN/corpus.json" \
  --selection experiments/post_training/task_curation/recipes/snowball-100k.json
```

Add `--run` to publish. The printed `task_selection` path is the artifact root
to pass to training. Its `summary.json` records the recipe, source artifacts,
selected counts and evaluation IDs. Check the source and family quotas there
before launching. A source shortfall fails instead of borrowing from another
source. Source or selection changes require a new recipe version; preserve
published versions for comparisons.

The artifact contains `train.parquet`, `validation.parquet` and
`full_validation.parquet`. The fixed validation panel samples 20 tasks per
family from the original 910-task development set. All 910 are excluded from
training by ID, public semantics and exact conversation context. This reused
development set is a hillclimbing signal; use a separate benchmark for final
claims. See [selection semantics](../references/task-curation.md#final-task-selection)
for deduplication and task serialization.

## Train and evaluate

Load the storage, tracking, image-grader and judge credentials required by the
[RL launcher](../references/rl-launching.md#experiment-entry-point) into the
submission environment. Use the published selection path and a fresh run version
to print a three-update plan:

```bash
uv run python -m experiments.post_training.inventory.train_quick_loop \
  --selection-uri "$SELECTION_URI" --version "$RUN_VERSION" --steps 3
```

Set `SELECTION_URI` to the printed artifact root, including its selection name
and version. Add `--run` to submit the graph through the maintained artifact
entry point. After the short run succeeds, use a fresh `RUN_VERSION` and
`--steps 25` for the normal iteration. Each run starts from the pinned SFT model.
The selected data stays cached; training does not curate or resample it.

The launcher binds `train.parquet` and `validation.parquet` from that artifact.
Evaluation attempts the fixed panel before training and after the last update,
and saves task-level verifier outcomes. Under the run artifact's `exports/`,
`dumped_evals/global_step_0_evals/` holds the baseline and
`dumped_evals/global_step_<steps>_evals/` holds the final evaluation. Each contains
per-dataset JSONL responses and `aggregated_results.jsonl`; compare its
`avg_score` and `pass_at_1` metrics alongside verifier coverage. The final update saves a checkpoint;
HF export is opt-in with `--export-hf` and adds a separate export job after training.
Local graders require the privileged container profile; image graders and host judges use their configured backends.

## Compare the iteration

Require `SUCCESS` from the Iris coordinator submitted by `--run` and its training
child, plus successful completion of the Marin training artifact. With `--export-hf`,
also require the export child to succeed. Its `terminal.json` records the run result;
the returned `SkyRLRun` supplies `iris_job_id` and `checkpoint_root` for follow-up inspection.
`hf_model_uri` is populated only when export is requested. Check
optimizer-update counts, finite metrics, initial/final attempted and scored
counts and native checkpoint metadata; check the completed HF export only when requested.
A missing verifier result is an evaluation error, not a wrong answer; report coverage with scores.

Record the source revision, selection identity, initializer, run version and
elapsed submission-to-completion time with each comparison. Hold the validation
panel fixed and change one recipe choice at a time. Reuse the published data for
training-only changes; produce a new selection version when changing tasks or
sampling. Update this page and the checked-in recipe when adopting a better
configuration, and keep trial results in the experiment issue or report.

# Small-model merging playground

Five independently written CPU reference merges are cross-checked against
`marin.merging.arithmetic` on identical tensors. The evaluation runner uses vLLM
on one GPU and retains every completion and score for paired comparisons.

## Models and interpretation

`models.json` pins public model revisions. The main dense arm starts from
Qwen2.5-1.5B-Instruct and trains matched math/code specialists. The limited far
arm transfers Coder-Instruct or Math-Instruct weights into Instruct. Official
model-card `base_model` fields document Qwen2.5-1.5B ancestry; they do not pin the
unpublished historical initializer revision. The
[Coder report](https://arxiv.org/html/2409.12186v3) reports over 5.5T
continued-pretraining tokens, so Coder is a far-donor control.

The [Math description](https://qwenlm.github.io/blog/qwen2.5-math/) documents
Qwen2.5 initialization and over 1T mathematical pretraining tokens.
Math's native `rope_theta=10000` differs from Instruct's `1000000`.
All merges use the Instruct value; Math alone is evaluated at both values to
measure the RoPE effect. The context limit is explicit (`--max-model-len`), and
every candidate in a comparison shares one output-token budget. All candidates
use the pinned general Instruct tokenizer, template and stop IDs; the runner
verifies identical token-to-ID mappings in every downloaded model. These are
comparisons under a common prompting policy, not reproductions of model-card
benchmark scores. Non-weight configuration choices stay fixed apart from the
explicit Math RoPE control.

The MoE family is `allenai/OLMoE-1B-7B-0924` → `-SFT` → `-Instruct`
(DPO from SFT). The [official adaptation commands](https://github.com/allenai/OLMoE#adaptation)
document the chain. Base/SFT/Instruct share architecture and RoPE settings.
Their revisions are recorded for subsequent router and expert-row experiments;
the initial dense sweep does not evaluate OLMoE.

## Arithmetic

For anchor B and donor D_i, each task vector is D_i - B, calculated in FP32.
The result is cast once to the anchor dtype.

| Method | Definition |
| --- | --- |
| average | (1 - sum(c_i)) B + sum(c_i D_i); unit scale and density |
| task_arithmetic | B + scale × sum(c_i (D_i - B)) |
| ties | Retain ceil(density × tensor size) largest absolute entries per donor; elect sign from the sum of weighted vectors; average agreeing vectors by their coefficient mass; apply scale |
| dare_linear | Independently retain each entry with probability density; divide retained vectors by density; task arithmetic |
| dare_ties | DARE followed by TIES sign election and disjoint averaging; extra magnitude trimming uses `ties_density`, which defaults to 1 |

These definitions match Marin, including exact-zero sign votes giving no
update, nonzero agreeing entries only, deterministic SHA256(seed:name:donor)
mask seeds and tensor-local top-k. A tensor's DARE mask does not change when
tensor traversal order changes. Top-k ties follow Torch's selection behavior.
Coefficient mass normalization means uniformly scaling positive TIES
coefficients cancels; change `scale` to change the update amplitude. DARE
rescaling preserves each linear vector's expectation before sign filtering;
it does not make DARE-TIES unbiased. No semantic difference from the production
implementation is intended. Reproducibility across Torch versions is not assumed.

## Run

Submit experiments from the repository root to Iris / CoreWeave. The owner's
2026-10-23 policy prohibits local model training, evaluation, merging and heavy
preparation on karans-linux, including small models. Local work is limited to
light lint/unit tests, UI and metadata checks. Resident voice-notes and Whisper
services stay running. The model commands below are worker commands, executed
inside an Iris job. Use `host-test-budget` only for light local unit tests.

Use a current Iris submission client including secret-file delivery (commit
c7c14a252d or later). The experiment branch predates that API; its older bundled
CLI cannot forward CoreWeave credentials to the current server. Source the
existing operator environment, and let Iris inherit secrets as files. Do not
put credential values on command lines. The worker bootstrap pins secret-aware
rigging from the same revision. Keep `config/coreweave.yaml`'s explicit
`marin-rno2a` / `RNO2A` declaration (added upstream in #10436). `cloud_io`
rejects unrecognized S3 buckets and endpoints before transferring data; without
that declaration a generic S3 client can silently select the host's AWS account.

```bash
set -a
source /path/to/operator.env
set +a
IRIS_BIN=/path/to/current/iris
"$IRIS_BIN" --cluster marin job run --enable-extra-resources \
  --target-cluster cw-rno2a --cpu 8 --memory 32GB --disk 32GB \
  --job-name playground-norms --no-sync --no-wait \
  -e PLAYGROUND_OUTPUT s3://marin-rno2a/scratch/USER/playground/norms \
  -- bash experiments/weight_merging/playground/iris_norms.sh
```

For pruning controls chosen by retained energy, set
`PLAYGROUND_NORM_MODULE=experiments.weight_merging.playground.pruning_response`
on the norm job. It scans241log-spaced densities, selects the nearest90%,70%,50%
squared-L2 retention points and emits matched-c0.5 top-k/DARE recipes. The
metrics include realized retention and FP32/BF16 output norms.

`iris_resume.sh` resumes this pilot's code donor from its complete step50
checkpoint to step100 on one H100. Stage `code-rft/checkpoint-50/` and `rft-data/`
under `PLAYGROUND_INPUT`, set `PLAYGROUND_OUTPUT` and `PLAYGROUND_REVISION`, and
submit with `--gpu H100x1 --cpu 8 --memory 32GB --disk 64GB --no-sync`. The script
relocates the saved best-checkpoint path and records the hardware transition;
bitwise training equivalence across GPUs is not assumed. Resume training uses
Torch2.6.0/CUDA11.8 because Transformers4.57.6 rejects optimizer loading on
Torch2.5.1. Transformers, bitsandbytes0.45.5 and Liger0.6.4 stay fixed. Math's
100steps use the old stack; code's final50use the new stack. This is a known
asymmetry in the 100-step trajectories, recorded in resume provenance. Both
selected donor exports use the best step-50 weights from the original stack;
the resumed code step-100 weights are excluded by validation selection.
Set `PLAYGROUND_EVAL_ONLY=1` to measure the step-50 checkpoint's validation
loss on the resume stack without optimizer updates; this publishes only
`evaluation.json`. Compare it with the stored validation loss before gating
when a hardware or library transition changes the observed loss.

`iris_generate.sh` runs each candidate separately on one H100, with
cap8192/context12288 and a fresh-versus-reloaded 20-item parity check before
each candidate. Set `PLAYGROUND_PANEL` to the frozen panel object's S3 URL,
`PLAYGROUND_RECIPES` to the recipes object, `PLAYGROUND_REFERENCE` to the
original run's metadata object, and `PLAYGROUND_OUTPUT` / `PLAYGROUND_REVISION`.
The generation Python files must match the reference's recorded hashes exactly.
For an old source bundle, copy only this shell launcher and the nested `remote/`
sidecar into that bundle; retain the original top-level Python files. Submit with
`--gpu H100x1 --cpu 8 --memory 32GB --disk 64GB --no-sync`.
Set `PLAYGROUND_RECOVERED_RUN` to an immutable recovered run prefix to import
complete candidates. `PLAYGROUND_PREFLIGHT=1` verifies the source and recovery
on a CPU worker before allocating a GPU. Every candidate publishes raw outputs
and provenance with read-back hashes, then writes `candidates/NAME/done.json`
last. Retries validate those objects and skip completed candidates. The final
combined run is under `aggregate/`; a partial aggregate contains only completed
candidates. Keep one writer per phase prefix.
Set `PLAYGROUND_FAMILY=far` to download the pinned Instruct/Coder/Math family
for the small far-donor negative-control arm, using `recipes --arm far`.
Set `PLAYGROUND_BASE_CONTROL=1` to run the base model's eight-shot GSM8K
control after the curve in the same job.

`iris_rft_gates.sh` evaluates Instruct, math-RFT and code-RFT under both BF16
and FP32 serving with the same full panel and decoding policy. Set
`PLAYGROUND_MATH_RFT` and `PLAYGROUND_CODE_RFT` to source prefixes containing
`final-fp32/` and `training-metadata.json`, plus `PLAYGROUND_PANEL`,
`PLAYGROUND_OUTPUT` and `PLAYGROUND_REVISION`. Submit on one H100 with
32GB RAM/64GB disk. Registering donor hashes precedes generation; scoring is
separate for each precision. Admission uses paired own-domain improvement.

`iris_precision.sh` compares the original math SFT donor's task arithmetic
`B+c(D-B)` at c=0.1 and 0.25 under BF16 and FP32 serving, with a fresh Instruct
control for each precision. Set `PLAYGROUND_MATH_SFT` to its source prefix,
plus `PLAYGROUND_PANEL`, `PLAYGROUND_OUTPUT` and `PLAYGROUND_REVISION`; use
the same H100 resources as the donor gate job. Code scoring stays deferred.

CW GPU pods have containerd rather than Docker. Use `sweep --defer-code-scoring`
there: MBPP rows have null correctness and an explicit pending flag. Generation
never executes their Python. `report` refuses pending scores. Run `iris_score.sh`
on an Iris GCP Docker worker with `--container-profile container_profile_docker_access`
to fill in the scores using the unchanged pinned Python image and resource
limits. `PLAYGROUND_PARITY_ITEMS=20` checks the first20fixed MBPP IDs on all
baseline greedy rows against their existing scores; any difference fails the
job. Normal scoring writes a separate scored run, retaining the raw generation
records. Each scored observation is saved immediately under
`PLAYGROUND_OUTPUT/item-grades/items/`; retry with the same output prefix to
reuse it. The cache key covers the completion, item and tests, decode identity,
image and scorer source hashes. Direct `rescore` calls require `--cache PREFIX`.
Infrastructure errors abort scoring and preserve completed grades. Keep large
inputs/results in S3/GCS rather than local scratch.

Use isolated CUDA environments because the repository defaults to CPU Torch.
The tested GPU driver is 535.309.01; CUDA 12.1 is compatible with it.

```bash
uv venv /tmp/merge-eval --python 3.12
uv --no-config pip install --python /tmp/merge-eval/bin/python \
  torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv --no-config pip install --python /tmp/merge-eval/bin/python \
  vllm==0.6.6.post1 transformers==4.48.3 datasets==3.2.0 pyarrow==18.1.0 numpy==1.26.4
uv run --no-project --python /tmp/merge-eval/bin/python \
  python -m experiments.weight_merging.playground.prepare --output /tmp/merge-run --panel full
uv run --no-project --python /tmp/merge-eval/bin/python \
  python -m experiments.weight_merging.playground.recipes --arm far --output /tmp/far-recipes.json

docker pull python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
env VLLM_ATTENTION_BACKEND=XFORMERS \
  uv run --no-project --python /tmp/merge-eval/bin/python \
  python -m experiments.weight_merging.playground.sweep --run /tmp/merge-run \
  --candidates /tmp/far-recipes.json --max-new-tokens 2048
```

`prepare --panel smoke` freezes 12 GSM8K and 8 MBPP IDs for pipeline checks.
The full panel includes all 1319 GSM8K and 257 sanitized MBPP test items.
`panel.json` stores exact IDs, content and pinned dataset revisions;
`model-paths.json` and `model-sources.json` record source locations and revisions.
Downloads and the Docker pull should complete before borrowing the GPU.

The original 256-token sweep was stopped after its baseline passes because
Instruct truncated 32.83% of greedy GSM8K and 8.95% of greedy MBPP outputs.
Set one output cap for the full comparison using the longest-writing donor;
The 2048 command above is a starting configuration. Run the length probe below
before the main comparison; raise the shared cap if the longest donor still
truncates more than 3% of either dataset. Re-evaluate all baselines at the chosen
cap, and freeze it before any grid candidate.

### Train matched near donors

The main arm uses two full fine-tunes of the pinned Instruct model. Both use
8000 training and 256 validation examples, maximum training length 1024,
1000 steps, LR 1e-5, cosine decay with 5% warmup, effective batch 8, zero weight
decay, gradient clipping 1 and seed 290. Math uses GSM8K-train plus MetaMathQA;
code uses Python Magicoder-OSS-Instruct. Equal example counts do not match token
counts. Test-question exact/13-word overlaps and duplicate source questions are
excluded; train/validation source questions are disjoint. Examples exceeding
the length cap are excluded instead of truncating their answers.

```bash
uv venv /tmp/merge-train --python 3.12
uv --no-config pip install --python /tmp/merge-train/bin/python \
  torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv --no-config pip install --python /tmp/merge-train/bin/python \
  transformers==4.57.6 datasets==4.8.5 accelerate==1.12.0 \
  bitsandbytes==0.45.5 liger-kernel==0.6.4 wandb

uv run --no-project --python /tmp/merge-train/bin/python \
  python -m experiments.weight_merging.playground.training_data \
  --parent PARENT_SNAPSHOT --panel /tmp/merge-run/panel.json --output /tmp/sft-data

env \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
  WANDB_PROJECT=weight-merging-playground \
  uv run --no-project --python /tmp/merge-train/bin/python \
  python -m experiments.weight_merging.playground.train \
  --parent PARENT_SNAPSHOT --data /tmp/sft-data --domain math --output /tmp/math-sft
```

Replace `PARENT_SNAPSHOT` with the pinned Instruct snapshot path from
`model-paths.json`. Run the same command with `--domain code --output /tmp/code-sft`.
For a single training reservation, put both commands in a shell script under
one Iris worker job. Use `--resume CHECKPOINT` after a saved checkpoint.

Training keeps FP32 master parameters and saves FP32 donors under `final-fp32/`.
It uses BF16 autocast, TF32 for residual FP32 matrix multiplications, gradient
checkpointing, paged 8-bit Adam and Liger fused linear cross-entropy. Only the
linear-loss fusion is enabled; the attention/norm/MLP implementations are fixed.
Validation loss selects among 250-step checkpoints; test scores do not select
training checkpoints. Retention keeps only the latest resumable checkpoint, plus
weights from an older validation-best checkpoint until final export. After export,
only the latest checkpoint and `final-fp32` remain. Older optimizer states are
removed; intermediate checkpoints are not an analysis archive. Training metadata records the configuration, parent path,
data/source hashes, versions and selected checkpoint. Offline W&B and local
Trainer state retain training metrics.

Create a separate near-arm run, then register every planned donor before the
first evaluation. Source manifests are immutable once outputs exist. Keep the
far and near arms in separate run directories.

Register the FP32 donors and generate the main recipe file:

```bash
uv run --no-project --python /tmp/merge-eval/bin/python \
  python -m experiments.weight_merging.playground.prepare \
  --output /tmp/near-run --panel full --models anchor instruct
for domain in math code; do
  uv run --no-project --python /tmp/merge-train/bin/python \
    python -m experiments.weight_merging.playground.register --run /tmp/near-run \
    --name "${domain}_sft" --path "/tmp/${domain}-sft/final-fp32" \
    --training-metadata "/tmp/${domain}-sft/training-metadata.json"
done
uv run --no-project --python /tmp/merge-eval/bin/python \
  python -m experiments.weight_merging.playground.recipes --arm near --output /tmp/near-recipes.json

env VLLM_ATTENTION_BACKEND=XFORMERS \
  uv run --no-project --python /tmp/merge-eval/bin/python \
  python -m experiments.weight_merging.playground.sweep --run /tmp/near-run \
  --candidates /tmp/near-recipes.json --only instruct math_sft code_sft \
  --max-new-tokens CHOSEN_CAP
```

Registration writes checkpoint paths, weight/config hashes and training metadata
hashes into the two run manifests. The baseline command above is the donor gate:
require a strictly larger greedy correct count than Instruct on GSM8K for math
and MBPP for code. Report sampled accuracy and paired wins/losses as well; a
positive point estimate alone does not establish statistical significance.
Checkpoint selection still uses validation loss, and all test comparisons are
exploratory. After both gates pass, rerun without `--only` to execute the grid.
Resume retains the same source, recipes and evaluation settings.

Measure model-wide relative distance `||donor - Instruct|| / ||Instruct||` with
FP64 reductions before the grid. The helper also reports how many FP32 update
entries disappear when the donor is rounded to BF16, and final BF16 cast survival
and error at task-vector coefficients 0.1, 0.25, 0.5 and 1.0:

```bash
uv run --extra pipeline python \
  -m experiments.weight_merging.playground.distances \
  --anchor PARENT_SNAPSHOT --donor /tmp/math-sft/final-fp32 --output /tmp/math-distance.json
```

`EphAsad/Aristaeus` @ `f6e918c83cda9aa8ec57ab8d57e9a557b8190b00` is an
additional math/reasoning donor, trained with full BF16 SFT on OpenThoughts3 and
Stratos. Download that revision, register it as `aristaeus` with `--repo` and
`--revision`, and add `{"name": "aristaeus", "source": "aristaeus"}` to a candidate
file. It must pass the same GSM8K gate as the self-trained math donor. It does
not replace either matched training arm.

Use its pinned local snapshot and the general Instruct tokenizer for the length
probe. This samples 128 GSM8K and 32 MBPP items with fixed seed 290:

```bash
env VLLM_ATTENTION_BACKEND=XFORMERS \
  uv run --no-project --python /tmp/merge-eval/bin/python \
  python -m experiments.weight_merging.playground.controls --mode length_probe \
  --model ARISTAEUS_SNAPSHOT --tokenizer PARENT_SNAPSHOT \
  --panel /tmp/merge-run/panel.json --max-new-tokens 4096 --output /tmp/length-probe.json
```

Select `CHOSEN_CAP` from these lengths, with at least 2048 tokens. Verify the
full donor baseline's truncation after the probe; raise the cap and rerun the
baselines if necessary. Set `--max-model-len` high enough to fit the prompt and
output budget. The control helper also supports `--mode base_fewshot`, using
eight fixed GSM8K training examples and plain Question/Answer prompting on all
test items. Pass the untuned base snapshot as both `--model` and `--tokenizer`.

### Rejection-sampled self-distillation pilot

The first human/MetaMath math SFT donor regressed from 943 to 844 greedy GSM8K
correct answers at an 8192-token cap. Its 231 paired losses comprise 230 explicit
wrong numeric final fields and one truncated loop; answer extraction is not the
cause. Both training and evaluation use the same final-marker instruction.

The replacement pilot samples the pinned Instruct parent on train-only prompts,
retaining correct, nontruncated continuations. Freeze 1024 GSM8K training prompts
and MBPP full training prompts after test-question overlap checks:

```bash
python -m experiments.weight_merging.playground.rft_data prepare \
  --panel /tmp/near-run/panel.json --output /tmp/rft-tasks.json

env VLLM_ATTENTION_BACKEND=XFORMERS \
  /tmp/merge-eval/bin/python -m experiments.weight_merging.playground.rft_sample \
  --tasks /tmp/rft-tasks.json --parent PARENT_SNAPSHOT --output /tmp/rft-samples.json

/tmp/merge-train/bin/python -m experiments.weight_merging.playground.rft_data build \
  --tasks /tmp/rft-tasks.json --samples /tmp/rft-samples.json \
  --parent PARENT_SNAPSHOT --output /tmp/rft-data
```

Sampling uses k=4, temperature 0.7, top-p 0.95, cap 1024 and seed 290 plus task
index. Math requires a correct marked answer. Code must execute successfully in
the Docker scorer; verification also includes any challenge assertions absent
from the prompt. Visible tests match the evaluation prompting policy. These
checks establish consistency with the available assertions, not code correctness
on every possible input.

The builder splits source questions before selecting 512 training and 128
validation continuations per domain, uses the same assistant-loss encoding and
1024-token training limit, and stops if either domain has insufficient yield.
Fine-tune both with `train --steps 100 --learning-rate 5e-6 --eval-steps 50`,
otherwise retaining the matched training settings. Save FP32 donors. Evaluate
both FP32 and BF16 serving against a parent served in the same precision. Report
greedy paired W/L and exact two-sided sign-test p-values. For admission, average
each item's paired score differences across greedy and sampled seeds 17/29,
then bootstrap those item means with 5,000 resamples (seed 290). Require a
positive mean gain in both serving precisions and a 95% interval excluding zero
in at least one. Pooled sign tests are descriptive because the three decodes
of one item are correlated. All comparisons are exploratory; these intervals
and p-values do not correct for recipe selection.

### Aristaeus dose response

The real-donor question is whether partial transfer preserves Aristaeus's math
gain while suppressing its unclosed thinking on code tasks. Eleven points include
the parent and donor, three convex averages at c=0.25/0.5/0.75, and three points
each for density-0.3 magnitude top-k pruning and DARE-linear. With one donor,
TIES reduces to magnitude pruning: this experiment measures no sign election.
Single-donor DARE-TIES would duplicate DARE-linear at matched update size.

```bash
/tmp/merge-train/bin/python \
  -m experiments.weight_merging.playground.dose_response \
  --anchor PARENT_SNAPSHOT --donor ARISTAEUS_SNAPSHOT \
  --output /tmp/aristaeus-dose-recipes.json --metrics /tmp/aristaeus-dose-norms.json

env VLLM_ATTENTION_BACKEND=XFORMERS \
  /tmp/merge-eval/bin/python -m experiments.weight_merging.playground.sweep \
  --run /tmp/near-run --candidates /tmp/aristaeus-dose-recipes.json \
  --max-new-tokens 8192 --max-model-len 12288 --reload-check-every 5

python -m experiments.weight_merging.playground.report --run /tmp/near-run \
  --candidates /tmp/aristaeus-dose-recipes.json --output /tmp/aristaeus-dose-report.md
```

Register Aristaeus before starting the run, as above. The norm helper uses actual
seed-42 masks and tensor-local magnitude retention. It scales each sparse update
to match c times the full donor update's model-wide L2 norm before serving. It
also measures the resulting FP32 and BF16 update norms, so serving casts remain
visible. The report includes greedy accuracy, unclosed `<think>` rates and
truncation and mean generated tokens per dataset, paired W/L with sign tests,
sampled scores by seed, and bootstrap intervals for per-item mean gains.
Its companion JSON retains per-seed lengths, counts and denominators.
`iris_generate.sh` defaults to BF16 serving; set `PLAYGROUND_DTYPE=float32` for
FP32 merging and serving end to end. Use a fresh Instruct control at each dtype;
this comparison changes both weight casting and activation arithmetic.
To isolate final weight rounding at fixed FP32 compute, add
`"output_precision": "bfloat16_rounded"` to a candidate and serve it with
`--dtype float32`. This rounds the completed FP32 merge once to BF16, then
upcasts those exact values to FP32 before loading. It differs from
`donor_precision`, which rounds donors before task-vector arithmetic.

### Format audit

Original scoring extracts the longest closed Python fence, or the full output
when no fence exists. Preserve that primary score. A separate diagnostic prefers
the last closed Python fence, otherwise the last syntactically complete
contiguous unfenced code region. It chooses by syntax before executing one
candidate; no test result selects among drafts or edits their code.

```bash
python -m experiments.weight_merging.playground.format_audit \
  --run /tmp/near-run --candidate aristaeus --output /tmp/aristaeus-format-audit.json
```

At cap8192, Aristaeus scored16/257 on MBPP under original extraction and94/257
under this diagnostic (78 recoveries, including45 truncated drafts, no lost
passes). Instruct remained157/257 under both policies. Draft salvage measures
code present inside output, including unfinished thoughts; it is not a completed
answer score. Aristaeus's GSM8K986/1319 vs Instruct943/1319 has paired W/L166/123,
exact two-sided p=0.01335 before correcting for exploratory comparisons.

### Recipes and precision

The near grid has 32 candidates: three single-model baselines, two convex
averages, two task-arithmetic scales, 24 density/scale configurations across
TIES, DARE-linear and DARE-TIES, plus a BF16-rounded-donor TIES control.
Densities are 0.1/0.3/0.5/0.7 and scales are 0.5/1. Donor coefficients are
0.5/0.5 except one convex average at 0.25/0.25. Merge construction seed is 42.

The far arm has 14 controls: Instruct, Coder, Math under native and anchor RoPE,
and five Instruct→specialist transfers for each of Coder and Math. The transfers
are convex weights 0.1/0.3/0.5, TIES at density 0.3 and scale 0.3, and DARE-linear
at density 0.3 and coefficient 0.3. Large pretrained-specialist deltas motivate
this limited control arm. Weight distance alone does not prove loss-basin
separation.

The runner accepts explicit JSON candidate lists. Each entry names a source,
optional donor list, recipe, RoPE value and donor-precision policy. All floating
source tensors are upcast to FP32 before arithmetic. FP32-trained donors keep
their saved precision; only the merged serving tensor is cast to BF16 by vLLM.
The paired TIES control rounds donor weights to BF16 before upcasting and
computing deltas. Existing public BF16 checkpoints already contain that rounding.

For a serving-precision control, register the same source manifest in a separate
run and pass `sweep --dtype float32` with the same candidates, panel, cap and
context limit as the BF16 run. Include Instruct itself as a control: FP32 serving
changes activation arithmetic as well as preserving small merged weight updates.
This comparison does not isolate weight rounding from activation precision.

### Evaluation and artifacts

Every candidate uses greedy decoding and sampled seeds 17/29 (temperature 0.7,
top-p 0.95). Each request adds its fixed panel index to its seed. The output
records both seeds. Sampled accuracy pools two attempts per item and is not
pass@2. Inspect `finish_reason` and per-dataset truncation before comparing scores.

Before evaluation, fresh Instruct and an in-place reload must produce identical
greedy token IDs on the fixed 20-item panel. The check repeats every 16 candidates.
vLLM's architecture-aware loader covers packed QKV and gate/up tensors; a coverage
assertion rejects missing parameters. Prefix caching is disabled between models.
A changed RoPE value creates a fresh engine. Tensor streaming avoids persistent
merged checkpoint copies. This reload path is tied to vLLM 0.6.6.post1.

MBPP code runs in network-disabled, read-only Docker containers as an unprivileged
user, with memory/CPU/process limits and a wall timeout. Docker infrastructure
failure aborts evaluation. All MBPP test assertions appear in the prompt and are
executed. GSM8K extracts the final marked/boxed number, falling back to the final
numeric token.

`outputs.jsonl` records completion text, tokens, decoding settings and correctness
for each `(candidate, item_id, mode, seed)`. `summary.json` stores correct counts
and paired wins/losses against Instruct. `run-metadata.json` records sources,
recipes, versions, source hashes and the panel hash. Resume rejects changed
source/settings or partial candidate output and skips complete candidates.
`--only instruct code_sft` selects named candidates from a recipe file.

Check CPU arithmetic, scoring and assistant-loss masking:

```bash
host-test-budget uv run --extra pipeline --group test pytest -n 0 \
  experiments/weight_merging/playground
```

Recipe selection on these test panels is exploratory. Independent held-out
confirmation is needed before a generalization claim. OLMoE router/expert-row
interventions remain a later experiment.

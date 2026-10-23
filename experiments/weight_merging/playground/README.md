# Small-model merging playground

Five independently written CPU reference merges are cross-checked against
`marin.merging.arithmetic` on identical tensors. The evaluation runner uses vLLM
on one GPU and retains every completion and score for paired comparisons.

## Models and interpretation

`models.json` pins public model revisions. The first dense experiment anchors
on Qwen2.5-1.5B and merges Qwen2.5-1.5B-Instruct with
Qwen2.5-Coder-1.5B-Instruct. Their model-card `base_model` fields document the
Qwen2.5-1.5B ancestry; this does not pin the unpublished historical initializer
revision. Coder includes extensive continued pretraining, so its vector from
the public base is a far-donor test, rather than a small SFT update.
The [Coder report](https://arxiv.org/html/2409.12186v3) reports over 5.5T
pretraining tokens.

The second experiment adds Qwen2.5-Math-1.5B-Instruct. The
[Math description](https://qwenlm.github.io/blog/qwen2.5-math/) documents
Qwen2.5 initialization and over 1T mathematical pretraining tokens.
Math's native `rope_theta=10000` differs from the anchor's `1000000`.
All merges use the anchor value; Math is evaluated alone at both values to
measure the configuration effect. Context is capped at 4096 throughout. All
candidates use the pinned general Instruct tokenizer, template and stop IDs;
the runner verifies identical token-to-ID mappings in every downloaded model.
These are comparisons under a common prompting policy, not reproductions of
the model-card benchmark scores. Non-weight configuration choices stay fixed.

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

Run from the repository root. `gpu-borrow` and `host-test-budget` are local
helpers on `karans-linux`: they reserve the GPU around the resident LLM and
limit CPU test resources. Whisper remains running and uses about 2.24 GiB of
VRAM. On another host, use that host's resource controls.

Use isolated CUDA environments because the repository defaults to CPU Torch.
The tested GPU driver is 535.309.01; CUDA 12.1 is compatible with it.

```bash
uv venv /tmp/merge-eval --python 3.12
uv pip install --python /tmp/merge-eval/bin/python \
  torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python /tmp/merge-eval/bin/python \
  vllm==0.6.6.post1 transformers==4.48.3 datasets==4.8.5 pyarrow==18.1.0
uv run --no-project --python /tmp/merge-eval/bin/python \
  python -m experiments.weight_merging.playground.prepare --output /tmp/merge-run --panel full
uv run --no-project --python /tmp/merge-eval/bin/python \
  python -m experiments.weight_merging.playground.recipes --arm far --output /tmp/far-recipes.json

docker pull python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f
gpu-borrow "weight-merging playground" -- env VLLM_ATTENTION_BACKEND=XFORMERS \
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
2048 is the current minimum, pending the Aristaeus length check.

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
uv pip install --python /tmp/merge-train/bin/python \
  torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python /tmp/merge-train/bin/python \
  transformers==4.57.6 datasets==4.8.5 accelerate==1.12.0 \
  bitsandbytes==0.45.5 liger-kernel==0.6.4 wandb

uv run --no-project --python /tmp/merge-train/bin/python \
  python -m experiments.weight_merging.playground.training_data \
  --parent PARENT_SNAPSHOT --panel /tmp/merge-run/panel.json --output /tmp/sft-data

gpu-borrow "weight-merging playground" -- env \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline \
  WANDB_PROJECT=weight-merging-playground \
  uv run --no-project --python /tmp/merge-train/bin/python \
  python -m experiments.weight_merging.playground.train \
  --parent PARENT_SNAPSHOT --data /tmp/sft-data --domain math --output /tmp/math-sft
```

Replace `PARENT_SNAPSHOT` with the pinned Instruct snapshot path from
`model-paths.json`. Run the same command with `--domain code --output /tmp/code-sft`.
For a single training reservation, put both commands in a shell script under
one `gpu-borrow` call. Use `--resume CHECKPOINT` after a saved checkpoint.

Training keeps FP32 master parameters and saves FP32 donors under `final-fp32/`.
It uses BF16 autocast, TF32 for residual FP32 matrix multiplications, gradient
checkpointing, paged 8-bit Adam and Liger fused linear cross-entropy. Only the
linear-loss fusion is enabled; the attention/norm/MLP implementations are fixed.
Validation loss selects among 250-step checkpoints; test scores do not select
training checkpoints. Training metadata records the configuration, parent path,
data/source hashes, versions and selected checkpoint. Offline W&B and local
Trainer state retain training metrics.

After training, add `math_sft` and `code_sft` entries to both run source files.
Use absolute `final-fp32/` paths; record the training metadata and weight-file
SHA256 hashes in `model-sources.json`. Generate main-arm recipes with
`recipes --arm near`. Evaluate each specialist against Instruct and inspect its
weight distance before running the grid. Include a donor only after it improves
its own task under the same evaluation policy. `EphAsad/Aristaeus` is an
additional public full-SFT candidate, subject to its own gate.

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

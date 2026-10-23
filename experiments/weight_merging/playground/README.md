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

`gpu-borrow` and `host-test-budget` are machine-local helpers on
`karans-linux`. The former stops and restores the resident LLM around GPU work;
the latter limits CPU test resources and prevents overlapping heavy test jobs.
On another machine, use its GPU reservation and CPU resource controls.

Use an isolated CUDA environment: the repository's default Torch build is CPU.
Run these commands from the repository root. This environment was chosen for
the local driver 535 and RTX 3090; it uses CUDA
12.1. The runner's in-place reload is tied to vLLM 0.6.6.post1's worker API.

```bash
uv venv /tmp/merge-playground --python 3.12
uv pip install --python /tmp/merge-playground/bin/python \
  torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python /tmp/merge-playground/bin/python \
  vllm==0.6.6.post1 transformers==4.48.3 datasets==4.8.5 pyarrow==18.1.0
uv run --no-project --python /tmp/merge-playground/bin/python \
  python -m experiments.weight_merging.playground.prepare --output /tmp/merge-run --panel full

docker pull python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f

gpu-borrow "weight-merging playground" -- env VLLM_ATTENTION_BACKEND=XFORMERS \
  uv run --no-project --python /tmp/merge-playground/bin/python \
  python -m experiments.weight_merging.playground.sweep --run /tmp/merge-run
```

`--panel smoke` freezes 12 GSM8K and 8 MBPP IDs for pipeline verification.
The full panel includes all 1319 GSM8K test and 257 sanitized MBPP test items;
`panel.json` records exact IDs, content and dataset revisions. Downloads happen
before borrowing the GPU. Add `--models olmoe_anchor olmoe_sft olmoe_instruct`
to prepare the MoE checkpoints instead of the dense family.

The sweep has 35 candidates: three dense baselines, two convex averages, two
task-arithmetic scales, 24 density/scale configurations across TIES,
DARE-linear and DARE-TIES, two Math configuration controls and two Math merges.
Densities are 0.1/0.3/0.5/0.7, scales are 0.5/1; donor coefficients are 0.5/0.5
except the 0.25/0.25 convex average, which leaves 0.5 anchor mass.
The Math pair uses convex average at coefficients 0.5/0.5 and TIES at density
0.3, coefficients 0.5/0.5, scale 1. Every DARE merge uses construction seed 42,
separate from the decoding seeds. DARE-TIES keeps `ties_density=1`; its swept
`density` controls Bernoulli retention. In TIES and DARE-TIES, `scale` multiplies
the final sign-filtered average update before it is added to the anchor.
Each candidate uses greedy decoding and two sampled seeds (17, 29;
temperature 0.7, top-p 0.95). Each request adds its fixed panel index to the
sampling seed. The output records both seeds. Generation allows 256 new tokens;
inspect `finish_reason` for truncation before interpreting scores.

Before the sweep, the runner compares fresh Instruct greedy tokens against
an in-place reload on the 20-item smoke panel and fails if any differ.
It repeats this check every 16 candidates. Reload checks include packed QKV and
gate/up projection coverage through vLLM's architecture-aware loader. Prefix
caching is disabled to avoid reusing previous-model KV entries. Math's native
RoPE control constructs a fresh engine. Merging streams CPU tensors into one
resident GPU model, avoiding persistent merged checkpoint copies.

MBPP requires Docker and this image:
`python@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f`.
Generated code runs in read-only, network-disabled containers as an unprivileged
user, with memory/CPU/process limits and a wall timeout. Docker infrastructure
failure aborts evaluation. All official MBPP test assertions are provided in
the prompt and executed. GSM8K compares the marked final number, falling back
to the final numeric token when no marker or boxed answer appears.

`outputs.jsonl` stores completion text, token IDs, decoding settings and score
for each `(candidate, item_id, mode, seed)`. `summary.json` records correct
counts, plus paired wins/losses against Instruct. Sampled rows are two attempts
per item; their pooled accuracy is not pass@2. `run-metadata.json` records
versions, source hashes, models, recipes and panel hash. Resume skips completed
candidates and rejects changed source/settings or partial candidate output.
Use `--only instruct coder average-c0.5` for a selected run.

Check arithmetic without a GPU:

```bash
host-test-budget uv run --extra pipeline --group test pytest -n 0 \
  experiments/weight_merging/playground/test_arithmetic.py \
  experiments/weight_merging/playground/test_evaluation.py
```

The later controlled experiment should train small math/code specialists from
one Qwen2.5-0.5B checkpoint on training splits, then compare the same methods
with the far-donor results. Full test panels support exploratory comparisons;
selecting a recipe on them does not establish held-out generalization.

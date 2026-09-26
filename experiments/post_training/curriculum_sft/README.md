# Curriculum SFT

`generation.py` builds verified single-turn reasoning data for one capability of the pinned
`TASK_CURRICULUM` artifact in two GLM 5.3 steps:

1. `generate_curriculum_problems` sends one request per problem. Each request names a sampling
   facet and a difficulty target (MATH levels 3–5, AMC 12 or early AIME); requests cycle through
   every facet and then every difficulty. GLM returns a problem and a short LaTeX reference answer.
   The step rejects truncated or malformed responses, duplicate problem text, and answers that
   math-verify cannot parse. It writes `problems/` audit Parquet, exact `raw-responses.jsonl`, and
   a manifest.
2. `solve_curriculum_problems` samples several independent GLM solutions per accepted problem
   without the reference answer. A solution counts as correct when its last `\boxed{}` answer is
   math-verify-equivalent to the reference. The step keeps the first correct solutions, up to
   `solutions_per_problem`, that have non-empty reasoning and fit `max_solution_chars`. It writes
   `solutions/` audit Parquet with every sample's grade, `chat/` training Parquet, exact responses,
   and a manifest.

Agreement between the problem author and a blind solver is the only verification. It filters
ambiguous and mis-keyed problems, but a shared mistake still passes.

Each `chat/` row is one user problem and one assistant turn. The assistant's `content` is GLM's
final solution; its `reasoning_content` is GLM's reasoning. Rows set
`chat_template_kwargs.enable_thinking = true`, as Datakit does for examples with reasoning, so the
Marin chat template renders the `Reasoning: /think` system header and a
`<|start_think|>…<|end_think|>` block. Snowball's inference template selects `/think` when a
request does not set `enable_thinking`, so evaluations prompt the model in the same mode. Training
rows without the header or a think block teach the model to answer without thinking.

## Where each stage runs

The GLM relay (`DEFAULT_GLM_RELAY_JOB` in `experiments/post_training/glm.py`) is registered on the
marin Iris hub and is not reachable from `cw-*` clusters. Submit generation to the hub with
`GLM_BULK_TOKEN` set, and write it under the trial's `ttl=30d` source prefix:

```bash
uv run iris --cluster=marin job run --no-wait --job-name curriculum-math-generate-<date> \
  -e MARIN_PREFIX s3://marin-us-east-02a/tmp/ttl=30d/curriculum-math-20260924 \
  -e GLM_BULK_TOKEN "$GLM_BULK_TOKEN" \
  -- uv run python experiments/post_training/curriculum_sft/math_trial.py \
  --stage generate --version <SOURCE_VERSION> --run
```

Training and evaluation run on `cw-rno2a` with `MARIN_PREFIX` set to the `ttl=7d` trial prefix.
`math_trial.py` adopts the solved chat artifacts from the source prefix at `SOURCE_VERSION`:

```bash
uv run iris --cluster=cw-rno2a job run --no-wait --job-name curriculum-math-sft-<version> \
  -e MARIN_PREFIX s3://marin-us-east-02a/tmp/ttl=7d/curriculum-math-20260924 \
  -- uv run python experiments/post_training/curriculum_sft/math_trial.py \
  --stage train --learning-rate 1e-5 --warmup 1 --version <version> --run
```

Run `--stage after` with the same options to evaluate the trained model. `--learning-rate 0`
exercises the conversion, training, export, and serving path without changing the weights.

## Training and evaluation

The prepared artifacts feed `sft_step` through `ArtifactDatasetSpec`. The shared launcher
tokenizes the chat, masks user turns from the loss, and trains `SnowballLMHeadModel` from a cached,
weights-only Levanter conversion of the pinned Hugging Face checkpoint. The conversion is a separate
artifact dependency with a fixed version, so SFT retries and task variants reuse it without loading
the 39 HF shards again. Conversations stay separate (`pack=False`) because Snowball does not
consume packed-document attention masks. The launcher saves a sharded bfloat16 Hugging Face
checkpoint at the final training step under the training artifact's `hf/` directory.

`math_trial.py` declares a bounded loop: deterministic OlympiadBench and Math500 on the pinned
September 20 HF model, three algebra capabilities, four Snowball updates, and the same evaluations
after training. The deterministic OlympiadBench variant uses Minerva/SymPy equivalence without an
LLM judge; its scores are not comparable to historical judge-backed OlympiadBench runs. Generation
and HF conversion use fixed versions; the SFT and its re-evaluation use the CLI `--version`, so a
failed SFT can be retried without regenerating data or reconverting the model. `--stage full` binds
both evaluations and training in one graph after generation has completed.

Adam without warmup at a learning rate of 5e-5 damaged the model within two updates: Levanter's
training loss rose from 0.79 to 1.75 and Math500 fell to 14/500. Use a smaller learning rate with
warmup. The standard Levanter Adam optimizer does not reproduce Will's special-token learning-rate
or frozen-router-bias changes; it starts a fresh optimizer from the published September 20
weights.

# Curriculum SFT

`generation.py` builds verified single-turn reasoning data for one capability of the pinned
`TASK_CURRICULUM` artifact in two GLM 5.3 steps:

1. `generate_curriculum_problems` sends one request per problem. Each request names one content
   target, one difficulty target (MATH levels 3–5, AMC 12 or early AIME), and one answer form
   (a specific value, every solution, a count, an extremal value, or a sum or product); requests
   cycle through every combination. Without a required answer form, GLM phrases most equation
   problems as "find the sum of all real solutions". Content targets are the capability's
   sampling facets followed by its `includes` entries, because many catalog capabilities declare
   few or no facets. GLM returns a problem and a short LaTeX reference answer.
   The step rejects truncated or malformed responses, duplicate problem text, and answers that
   math-verify cannot parse. It writes `problems/` audit Parquet, exact `raw-responses.jsonl`, and
   a manifest.
2. `solve_curriculum_problems` samples several independent GLM solutions per accepted problem
   without the reference answer. A solution counts as correct when its last `\boxed{}` answer is
   math-verify-equivalent to the reference. The step keeps the first correct solutions, up to
   `solutions_per_problem`, that have non-empty reasoning and whose rows, rendered with the Marin
   chat template and the model's tokenizer, fit `max_sequence_tokens`. Unpacked SFT rejects longer
   rows, and a character cap is not a safe proxy: Unicode-heavy reasoning tokenizes at under two
   characters per token. It writes `solutions/` audit Parquet with every sample's grade, `chat/`
   training Parquet, exact responses, and a manifest.

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

The GLM relay (`DEFAULT_GLM_RELAY_JOB` in `experiments/post_training/glm.py`) is an Iris job on the
marin hub whose task runs on `cw-us-east-08a`, and its registered endpoint is reachable only from
that cluster: generation jobs on hub (GCP) workers time out connecting to it. Submit generation
through the hub and federate it to `cw-us-east-08a`, which also injects CoreWeave object-storage
credentials. Write it under the trial's `ttl=30d` source prefix:

```bash
uv run iris --cluster=marin job run --no-wait --target-cluster cw-us-east-08a \
  --enable-extra-resources --cpu 2 --memory 16GB --disk 32GB \
  --job-name curriculum-math-generate-<date> \
  -e MARIN_PREFIX s3://marin-us-east-02a/tmp/ttl=30d/curriculum-math-20260924 \
  -e GLM_BULK_TOKEN "$GLM_BULK_TOKEN" \
  -- uv run python experiments/post_training/curriculum_sft/math_trial.py \
  --stage generate --version <SOLUTIONS_VERSION> --run
```

Problem generation and solving use the fixed `PROBLEMS_VERSION` and `SOLUTIONS_VERSION`; the
experiment CLI still requires `--version`, which the generation graph does not use. A completed
step with a changed recipe serves its cached output, so bump `SOLUTIONS_VERSION` after changing
only the solve settings, and both versions after changing problem generation.

Training and evaluation run on `cw-rno2a` with `MARIN_PREFIX` set to the `ttl=7d` trial prefix.
`math_trial.py` adopts the solved chat artifacts from the source prefix at `SOLUTIONS_VERSION`:

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
training loss rose from 0.79 to 1.75 and Math500 fell to 14/500. With one warmup step, four updates
on the 912 verified `/think` rows gave these Math500 scores:

| Learning rate | Math500 | Think blocks opened / closed |
|---|---|---|
| 0 (control) | 332/500 | 284 / 255 |
| 1e-6 | 315/500 | 268 / 238 |
| 1e-5 | 182/500 | 387 / 53 |

At 1e-5 the model imitates GLM's terse reasoning and usually ends its reply inside the think block,
without `<|end_think|>` or a final solution, although the training rows close every think block
and train on `<|end_think|>`. At 1e-6 behavior matches the control within sampling variation. The standard Levanter Adam optimizer does not reproduce Will's special-token learning-rate
or frozen-router-bias changes; it starts a fresh optimizer from the published September 20
weights.

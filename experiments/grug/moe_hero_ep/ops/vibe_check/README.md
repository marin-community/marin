# Hero checkpoint completions

[Latest report](https://storage.googleapis.com/marin-public/rav/hero-completions/latest/index.html)
· [Actions workflow](../../../../../.github/workflows/marin-hero-completions.yaml)

## Token probabilities

Each prompt in [prompts.json](prompts.json) has an `expected` reference continuation.
Open-ended references are examples of valid answers. Other answers can also be correct.
The `north-america` reference includes Central America and the Caribbean.
The `presidents` reference lists US presidents from 1900 through 1999.

Each completion records `token_scores` for generated tokens and `expected_scores`
for the reference continuation. Each score contains the token ID, decoded text,
natural-log probability, and the five most likely tokens with their log probabilities.
The probabilities are `softmax(raw_logits)` across the full vocabulary, before temperature scaling or token filtering.
Generation still uses the configured temperature and seed.

Expected scores use the prompt followed by the preceding expected tokens.
This is teacher forcing: the model scores the reference even when generation selects a different answer.
The tokenizer encodes the prompt with special tokens and the expected text separately without special tokens.
Scoring adds no EOS token and includes no prompt-token probabilities.
The prompt and expected completion together must fit in the context, or the job fails before model loading.
Expected completions are not truncated at `max_new_tokens`.
One causal model pass scores each batch of references. The vocabulary projection processes one position at a time.

The [hero completion report](https://storage.googleapis.com/marin-public/rav/hero-completions/latest/index.html)
colors generated and expected tokens by probability.
Hover, focus, or tap a token to see its probability and the five most likely tokens.
The report also shows total log probability and the geometric mean of token probabilities for each continuation separately.
Generated scores include EOS when generation selects it. Expected scores include only the reference text.
These values describe the exact token sequence. They do not measure answer correctness.
Longer answers usually have lower total probabilities, so compare the same reference across checkpoints.
An EOS token or a byte fragment with no decoded text appears as a dot.
A Unicode character belongs to the token that completes its byte sequence.
Candidate text also uses the preceding tokens, so a candidate can complete a partial Unicode character.

Results without scores show plain text and a message that probabilities were not recorded.
The `hero-native-v2-logprobs` release creates new requests for retained permanent checkpoints during the next workflow invocation.
Previous results stay fixed. New data becomes available after these requests complete and a later workflow invocation publishes the report.
See [Operation](#operation) for checkpoint selection and workflow commands.

## Operation

Run discovery, report publication, and the status summary:

```bash
gh workflow run marin-hero-completions.yaml --ref main
```

Read the Actions summary for Iris job links, queue state, and errors.
A successful submission does not mean that sampling started or completed.

To submit all unfinished checkpoints at production priority in one invocation:

```bash
gh workflow run marin-hero-completions.yaml --ref main -f submission=all -f priority=production
```

`submission=all` submits one job for each unfinished request in the current
discovery set. Discovery uses the run list and step limits in
[`CHECKPOINT_RUNS`](config.py) and includes only retained permanent checkpoints.
It skips completed results, active jobs, and exhausted retries.
It uses the current prompt bank and excludes requests for previous prompt banks.
Each job needs 64 GPUs. Iris starts jobs as resources become available.

`submission=next` is the default for manual and scheduled invocations. It waits
until no sampling jobs are active, then submits at most one request.
The workflow saves the selected priority for the discovered requests and their
retries. Hourly invocations continue retries at the saved priority.
Future checkpoints use batch priority unless an invocation sets their priority.
An active job keeps its original priority. The next attempt uses the saved priority.
The three-attempt limit still applies.

The CLI also accepts `reconcile --submission all --priority production` from a clean checkout.
Use Actions for the shared sample store because its concurrency group
serializes submissions. The priority choices are `batch`, `interactive`, and
`production`. Manual workflow runs default to `batch`. Scheduled runs preserve
saved priorities. Iris checks the caller's permission for the selected priority.
Omit the CLI's `--priority` option to preserve saved priorities.

The submission invocation publishes results already available at that time.
Later workflow invocations publish results from the new jobs to the latest
report. Submission does not wait for all GPU jobs to finish.

For a status query without job submission, run from the repository root with
CoreWeave storage credentials and Iris authentication:

```bash
uv run --no-sync python -m experiments.grug.moe_hero_ep.ops.vibe_check status
```

## Recovery

Correct the sampler or access error before another attempt.
Requests retain their attempt counts after a fix. Retries use the original source commit.

For exhausted requests, change `release` in [config.py](config.py) through a PR.
This creates new requests for all retained checkpoints, including prior successes.

# Iterate on Snowball RL

The [iteration recipe](https://github.com/marin-community/marin/blob/main/experiments/post_training/snowball_inventory_loop.py)
trains a pinned initializer for 25 optimizer updates, commits its resumable
checkpoint and HF export, then evaluates that export. Evaluation failures leave
completed training reusable. The recipe retains the full 100,000-task training
pool, the existing optimizer settings, 8,192 reply tokens and a 24,576-token
context. A two-to-four-hour iteration is a target to measure, not a validated
runtime guarantee.

Choose `sft`, `drdoom` or `smartie`. Each choice binds the corresponding exact
model source and tokenizer revision. Supply the immutable model manifest identity
and the SHA256 of the active tokenizer chat template as well as the model URI.
The baseline preflight checks those identities before inference. A different
initializer starts with fresh optimizer state in its own artifact root; resuming
an existing artifact version resumes that artifact's checkpoints.

## Freeze the small development panel

Start from the current-semantics 910-row panel. Convert the historical source
with the maintained [panel adapter](https://github.com/marin-community/marin/blob/main/experiments/post_training/snowball_inventory_validation.py)
(`python -m experiments.post_training.snowball_inventory_validation --help` lists
its source, training-pool, hash and pinned-tokenizer inputs). Derive the 8k outer
runtime limits with `prepare_reference_panel` in the
[evaluation worker](https://github.com/marin-community/marin/blob/main/experiments/post_training/snowball_inventory_evaluate_runtime.py).
Retain the resulting source hashes and exact CPU admission report. Historical
exports using the retired token-prefix harness need conversion before this step.
This command reads local files and writes a new local artifact:

```sh
uv run python -m experiments.post_training.snowball_inventory_development \
  --source full-panel/validation.parquet --source-sha256 FULL_PANEL_SHA256 \
  --output fast-panel
```

The fast panel selects 20 rows from each of the five families, independently of
initializer or score. A versioned hash ranks source contract and task IDs within
each family, making selection independent of file order. The output preserves
selected records, schema, original graders and runtime limits. Its provenance
records ordered pairing identity, source hash and output hash. Freeze it once and
reuse the same artifact across iterations. It is a development subset of an
already reused panel; it cannot establish held-out generalization.

## Build and run an iteration

Publish and verify immutable training and panel artifacts using
[fsutil verified-copy and hash](../references/fsutil.md). The full panel is used
by training CPU admission to check task contracts and loader coverage. The
training job has no inline or periodic evaluation callbacks; the small panel is
the independent scoring input after export. Print the exact graph before
launching:

```sh
uv run python -m experiments.post_training.snowball_inventory_loop \
  --version YYYY.MM.DD.N --initializer sft --max-steps 25 \
  --model-uri MODEL_URI --model-manifest-identity MODEL_MANIFEST_IDENTITY \
  --tokenizer-template-sha256 TEMPLATE_SHA256 \
  --pool-uri POOL_URI --pool-version POOL_VERSION --pool-sha256 POOL_SHA256 \
  --validation-uri FULL_PANEL_URI --validation-version FULL_PANEL_VERSION \
  --validation-sha256 FULL_PANEL_SHA256 \
  --panel fast --panel-uri FAST_PANEL_URI --panel-version FAST_PANEL_VERSION \
  --panel-sha256 FAST_PANEL_SHA256 \
  --judge-base-url JUDGE_BASE_URL --judge-model JUDGE_MODEL
```

Add `--run` to the same command for the normal artifact-main submission route.
Set `DAYTONA_RL_API_KEY`, `DAYTONA_EVAL_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`
and `JUDGE_API_KEY` as required by the selected backends; the
[RL submission route](../references/rl-launching.md) forwards these to the
coordinator. Use a fresh immutable
version when changing training inputs, settings or runtime source. Choose another
positive multiple of 25 with `--max-steps` for a longer experiment.

The graph first resolves an initializer baseline. A successful baseline is
cached by checkpoint content/provenance, tokenizer revision and active template,
panel content and grader contracts, judge configuration, actual evaluation
recipe, recovery limits, scoring source-code bytes and nested runtime source/lock. An
unrelated training budget change does not evict it. Report cold-cache baseline
cost separately from warm-cache iteration cost.

For periodic broader checks, pass `--panel full` with the original full panel's
URI, version and SHA256 as the scoring inputs. Baseline and final scoring within
an iteration always use the same panel, fixed family denominators and decoding
protocol. Compare per-family scores and coverage alongside the aggregate; the
fast and full panels have different denominators.

Independent scoring persists answers before grading. Unscored saved answers get
at most one bounded original-grader recovery attempt, with no new inference.
Incomplete coverage produces a durable diagnostic report and fails the scoring
stage; it is never cached as a successful baseline. Inspect the recorded cause
before retrying a failed stage. Completed training and export remain available,
and existing artifact caching avoids repeating optimizer updates.

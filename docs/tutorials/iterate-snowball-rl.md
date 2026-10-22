# Iterate on Snowball RL

The [iteration recipe](https://github.com/marin-community/marin/blob/main/experiments/post_training/snowball_inventory_loop.py)
trains a pinned initializer for 25 optimizer updates, commits its resumable
checkpoint and HF export, then evaluates that export. The full 100,000-task
training pool, existing optimizer settings, 8,192 reply tokens and 24,576-token
context stay fixed. Two to four hours is an iteration target to measure; this
recipe has not established that runtime.

## Prepare the inputs

This guide consumes already frozen inputs. Before starting, obtain:

- A current-runtime 100,000-row training artifact containing `train.parquet`.
- A current-runtime 910-row development artifact containing `validation.parquet`,
  with the approved 8k runtime limits and no training overlap.
- The chosen initializer's immutable HF model artifact, including
  `.marinskyrl-model-manifest.json`, and its pinned tokenizer metadata.
- The source hashes, artifact manifests and exact CPU admission evidence for
  those inputs.

The maintained [curriculum export instructions](https://github.com/marin-community/marin/blob/main/experiments/post_training/snowball_curriculum/README.md)
and [panel adapter](https://github.com/marin-community/marin/blob/main/experiments/post_training/snowball_inventory_validation.py)
own data preparation. Historical token-prefix exports require conversion and
fresh admission; they do not satisfy these prerequisites. Local conversion and
tokenizer checks do not establish served-model or grader transport. The final
composed launch still runs its own CPU preflight before GPU work.

Choose `sft`, `drdoom` or `smartie`. The recipe's
[`INITIALIZERS` mapping](https://github.com/marin-community/marin/blob/main/experiments/post_training/snowball_inventory_compare.py)
pins each model source and tokenizer revision. `MODEL_URI` below names the
complete HF artifact directory for that choice. Copy its small manifest locally
as `model-metadata/.marinskyrl-model-manifest.json`; do not copy weights merely
to obtain these identities. This CPU-only command reads the manifest and loads
the selected pinned tokenizer through the same constructor used by admission:

```sh
uv run --project vendor/skyrl --frozen --extra cpu --no-group dev python - <<'PY'
import hashlib
import json
from pathlib import Path
from skyrl_train.tokenizer import create_tokenizer

repository = "PINNED_TOKENIZER_REPOSITORY"
revision = "PINNED_TOKENIZER_REVISION"
manifest = json.loads(Path("model-metadata/.marinskyrl-model-manifest.json").read_text())
assert (manifest["model_id"], manifest["revision"]) == (repository, revision)
tokenizer = create_tokenizer(repository, disable_fast_tokenizer=False, revision=revision)
print("MODEL_MANIFEST_IDENTITY=" + manifest["identity"])
print("TEMPLATE_SHA256=" + hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest())
PY
```

Copy the repository and exact revision from the chosen `INITIALIZERS` entry;
use that same initializer in the launch. The current recipe has
no custom chat-template override. CPU admission verifies the actual manifest and
active template again, including when it reuses a cached baseline.

## Freeze and publish the small development panel

With the full panel available locally as `full-panel/validation.parquet`, freeze
a new local artifact:

```sh
uv run python -m experiments.post_training.snowball_inventory_development \
  --source full-panel/validation.parquet --source-sha256 FULL_PANEL_SHA256 \
  --output fast-panel
```

The fast panel selects 20 rows from each of five families using a versioned hash
of source contract and task IDs. Selection is independent of initializer, score
and input file order. It preserves selected records, schema, graders and runtime
limits. `fast-panel/provenance.json` records the source and output Parquet hashes
and ordered pairing identity. Freeze once and reuse this 100-row artifact across
iterations. It is a reused development subset, not a held-out final test.

Publish complete artifact directories, including `artifact-manifest.json`,
provenance and payload files, using
[fsutil verified-copy and hash](../references/fsutil.md). Publish the training,
full-panel and fast-panel directories separately; their URI arguments name those
directories. The `--pool-sha256` value hashes the bytes of `train.parquet`;
`--validation-sha256` hashes the full `validation.parquet`;
`--panel-sha256` hashes the selected scoring `validation.parquet`. These are
payload hashes, separate from the artifact manifest hashes retained in your
publication receipts. `--model-manifest-identity` is the manifest's `identity`
field, including its `sha256:` prefix, rather than the manifest file's byte hash.

## Build and run an iteration

The full panel is used by training CPU admission for contract and loader checks.
Training has no inline evaluation callbacks. Independent scoring uses the small
panel after export. Print the exact graph first:

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

Add `--run` to this same command to submit through the normal artifact main.
Set `DAYTONA_RL_API_KEY`, `DAYTONA_EVAL_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`
and `JUDGE_API_KEY` as required by the selected backends; the
[RL submission route](../references/rl-launching.md) forwards them to the
coordinator. Record the printed coordinator job ID and follow its child jobs.
Submission returning is not training or scoring completion.

The graph resolves an initializer baseline, trains, then scores the committed
export. Successful baselines are cached by checkpoint content/provenance,
tokenizer revision and template, panel and grader contracts, judge configuration,
evaluation recipe, recovery limits, scoring source-code bytes and nested runtime
source/lock. Changing only the training budget does not evict that baseline.
Choose another positive multiple of 25 with `--max-steps` for a longer experiment.
Use a fresh immutable version when changing inputs, settings or runtime source.

For periodic broader checks, replace `--panel fast` with `--panel full` and
replace all three `--panel-uri`, `--panel-version`, `--panel-sha256` values with
the full panel's values. Keep the `--validation-*` arguments unchanged. Baseline
and final scoring use the same selected panel, fixed family denominators and
decoding protocol. Compare per-family scores and coverage alongside the
aggregate; fast and full panels have different denominators.

## Read results and resume failed scoring

The printed graph identifies the training and scoring artifact names and
versions under `MARIN_PREFIX`. In the training artifact directory,
`resolved-launch.yaml` records the launch and `terminal.json` records terminal
model/export metadata. The artifact's `.artifact.json` result contains the exact
`hf_model_uri`, `checkpoint_root`, `global_step` and worker `iris_job_id`.
The scoring artifact contains `report.json` with paired per-family scores,
coverage and recovery accounting. Its `preflight.json` and `admitted-request.json`
retain admission evidence; completed saved answers live under its
`exports/saved_answers` directory.

Record wall time from coordinator acceptance through final scoring completion,
including queueing, baseline, training, export and scoring. Use the child job
timestamps and optimizer telemetry to separate stage costs. Report baseline
cache misses separately from warm-cache iterations.

Scoring persists answers before grading. Eligible unscored answers get one
bounded original-grader recovery attempt without new inference. Incomplete
coverage writes a diagnostic report and fails scoring; it is not cached as a
successful baseline. After inspecting the report, if generation completed and
only scoring failed, rerun the identical launch command with `--run` and the
**same** `--version YYYY.MM.DD.N`. Completed training/export and generation are
reused. An exhausted recovery budget or incomplete generation requires diagnosis;
it does not authorize another inference run. Do not change the scoring contract
under the same version.

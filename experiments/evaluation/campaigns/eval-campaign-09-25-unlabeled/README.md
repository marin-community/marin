# Unlabeled evaluation campaign

This directory contains every version-controlled input for the campaign: model configs, Evalchemy
configs, Harbor configs, the co-hosted judge config, model-specific Harbor policy values, validation,
and launch commands. The launcher records the vendored Evalchemy runtime's source hash
and Harbor's upstream revision and runtime descriptor. It checks the independent Harbor
lockfile before submission.

The launcher leaves `EvalRunRecord.version` unset. Its checked-in inputs, config digests, and durable
records support reproduction and review; they do not attest that a run conforms to an accepted eval
policy. Policy conformance requires the independently trusted evidence and verifier described in
[issue #9458](https://github.com/marin-community/marin/issues/9458). The preflight and comparison work
in [PR #9461](https://github.com/marin-community/marin/pull/9461) must consume that trusted decision
before presenting a run as policy-conformant.

SWE-bench uses Harbor's native mini-SWE 2.1.0 adapter.
Its 32,768-token budget includes the rendered prompt and the requested reply. Before each
completion, the adapter counts the prompt, including tools and the generation prefix, through
the serving model's `/tokenize` endpoint. It caps the reply to the remaining context and the
model's output limit. When no reply fits, the harness exits through its normal limit handling
so Harbor can verify the existing patch. It does not truncate or summarize the conversation.
The model limits and Harbor runtime pin are recorded in the launch snapshot;
results from the previous adapter have a different policy digest and cannot be resumed under
this configuration.

Harbor's native SOTOPIA agent enforces the declared `model_info.max_input_tokens` budget for
the evaluated model at each request. It counts the rendered prompt with the serving endpoint's
tokenizer and caps the reply to the remaining context. Counterpart and evaluator calls use
their independently configured models and are not subject to this evaluated-model budget.
The verifier's default reward is the arithmetic mean of all seven normalized SOTOPIA
dimensions. It also records the evaluated agent's emitted character count. Run
`analysis/sotopia_length_control/rank.py` over the complete cohort to obtain the reportable
length-controlled overall score. Goal achievement remains an explicit non-default dataset
generation mode.

Evalchemy's 30-minute timeout starts separately for each request after it acquires a client
concurrency slot. A transport attempt may use that full window; retrying does not extend it.
`evalchemy_num_concurrent` in a model config limits simultaneous requests to that model's endpoint.
K2 is limited to four after its 16-request runs saturated serving and produced mostly timed-out
generations. Effective concurrency is recorded in each run's Evalchemy provenance.

Pass `--version LABEL` only when a submitter-controlled cohort label is useful. The launcher forwards
the label to Marin unchanged. It remains provenance metadata and does not alter the conformance
boundary above.

From a clean checkout of the campaign's pinned Marin commit:

```bash
cd experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled
./validate-campaign.sh
./launch-campaign.sh --model Qwen-Qwen3.6-35B-A3B
./launch-campaign.sh --submit --model Qwen-Qwen3.6-35B-A3B
```

The first launch command is a dry run. `--suite nonagentic` and `--suite agentic` select one full
mechanism. Repeat `--evalchemy NAME` or `--harbor NAME` to select individual configs; pair a selector
with its corresponding suite to avoid launching the other mechanism. `--wait` waits for durable
records and requires CoreWeave object-store credentials.

The launcher writes generated effective configs beneath
`experiments/evaluation/campaigns/.launch-staging/` and local provenance snapshots beneath
`experiments/evaluation/campaigns/.launch-snapshots/`. Both paths are ignored by Git. Durable
`record.json` files contain the effective model and evaluator configuration, repository revision,
and Harbor policy digest.

Credentials are operational inputs. Export `HF_TOKEN`, `TOGETHER_API_KEY`, and CoreWeave object-store
credentials, or use the configured Google Secret Manager access. Set `EVAL_CAMPAIGN_SECRETS_ENV` only
when reading Hugging Face or Together credentials from a local environment file.

## Sharded SWE-bench recovery

`analysis/swebench_recovery/shard.py` partitions the remaining tasks from a
combined recovery audit. Stop the source Iris root and confirm it is terminal,
then run `combine.py --manifest OLD_MANIFEST.json --recovery-results OLD_RESULTS_URL
--output AUDIT.json` before preparing replacement jobs. Supply manual reviews
with `--reviews REVIEWS.json` when required.
Retained tasks must have audited numeric rewards; infrastructure-affected and
unfinished tasks are assigned to exactly one new shard.
When a numeric verifier reward follows an agent startup failure, attach a
hash-bound `exclude_infrastructure_failure` manual review. The recovery audit
excludes that trial despite the reward; it does not apply the decision to other
trials with the same exception type.

Run `shard.py --audit AUDIT.json --base-config CONFIG.yaml --source-root ROOT
--count 4 --output DIRECTORY`. It writes a retained-task manifest, a combined
recovery config, and four `swebench-shard-N.yaml` configs. Check these files into
the campaign worktree before launching. Pass one shard YAML as the Harbor config
and the original canonical model-serving config as the model config for each job.
The combined recovery config is audit input, not an additional launch. Only
task selection differs between the shard configs.

After launch, record each canonical config path and its exact results URL in a
JSON list: `[{"config": "PATH.yaml", "results": "s3://.../results"}, ...]`.
Run `combine.py --manifest DIRECTORY/manifest.json --shards SHARDS.json
--output AUDIT.json`, adding `--reviews REVIEWS.json` for hash-bound manual
error classifications. The audit rejects overlapping assignments, tasks omitted
from the shard plan, benchmark-config drift, and runtime mismatches. Assigned
tasks without results remain pending. The audit records each shard's effective
config and counts valid scored tasks across all shards and retained evidence
against the original 500-task denominator.
Do not seal a shard independently or count an unfinished shard as a model zero.

# Unlabeled evaluation campaign

This directory contains every version-controlled input for the campaign: model configs, Evalchemy
configs, Harbor configs, the co-hosted judge config, model-specific Harbor policy values, validation,
and launch commands. The launcher records the Marin revision and vendored Evalchemy
runtime's source hash. It checks the independent Harbor lockfile before submission.

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
The model limits and dataset pins are recorded in the launch snapshot;
results from the previous adapter have a different policy digest and cannot be resumed under
this configuration.

Harbor's native SOTOPIA agent enforces the declared `model_info.max_input_tokens` budget for
the evaluated model at each request. It counts the rendered prompt with the serving endpoint's
tokenizer and caps the reply to the remaining context. Counterpart and evaluator calls use
their independently configured models and are not subject to this evaluated-model budget.
The verifier's default reward is the arithmetic mean of all seven normalized SOTOPIA
dimensions. It also records the evaluated agent's emitted character count. Harbor applies
length correction at job completion using the campaign's `sotopia_cohort` in
`harbor-configs/sotopia-hard.yaml`: the versioned `policy-10197-v1` cohort of 13 baseline
archives plus the evaluated model’s current run. SOTOPIA jobs require an explicit cohort
with a `version` and a `models` list containing `model` and `results_path` for each baseline. A rerun of a baseline model
replaces its previous source in the fit. Harbor preserves raw rewards and saves the cohort sources, input digest, fit parameters,
and adjusted episodes in the job result. Marin reads the corrected metric from that result. Goal achievement remains an explicit
non-default dataset generation mode.

`finalization.yaml` selects MiniMax for native FinanceBench and OlympiadBench grading.
Each judged benchmark starts the pinned 262,144-token TP8 MiniMax service on eight H100s
in the candidate's cluster before evaluation and releases it afterward. Evalchemy grades
answers once and preserves judge verdicts with the samples. Judge requests use constrained
labels and escalate empty final answers through 128, 512, and 2048 output tokens.

These stages finish before the run is recorded as successful. Missing cohort evidence
fails the Harbor job and records the run as `failed`. Evalchemy preserves checkpoints for
interrupted judge requests.
Changing SOTOPIA baselines does not invalidate FinanceBench or OlympiadBench results.
The launch snapshot includes the finalization configuration, judge YAML, and Harbor YAML
containing the campaign cohort. Harbor saves the resolved baseline cohort with the correction
evidence in its job result. `record.json` records the FinanceBench and OlympiadBench judge
configuration in `eval.postprocessing`. The generic evaluation CLI
accepts `--policy-finalization-config finalization.yaml`; `launch-campaign.sh` supplies the
absolute path to this YAML by default.

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

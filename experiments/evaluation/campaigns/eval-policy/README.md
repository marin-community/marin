# Verified evaluation campaign

This directory contains every version-controlled input for the campaign: model configs, Evalchemy
configs, Harbor configs, the co-hosted judge config, model-specific Harbor policy values, validation,
and launch commands. The launcher records the vendored Evalchemy runtime's source hash
and Harbor's upstream revision and runtime descriptor. It checks the independent Harbor
lockfile before submission.

The launcher defaults to `eval-policy-2026-10-22-verified`. Marin checks each selected benchmark
against the single checked-in [profile](../../../../lib/marin/src/marin/evaluation/policies/verified.json),
including its source settings and vendored runtime. Subset runs are allowed. EvalDash uses the same
checks for this cohort and keeps historical cohorts available without a verified marker.
These checks establish configuration agreement, not the independent attestation described in
[issue #9458](https://github.com/marin-community/marin/issues/9458).

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

Pass `--version LABEL` to use a different cohort for experiments. Retired `eval-policy-*` labels
cannot be used for new launches. Capped smoke runs require an experimental label, for example
`--limit 1 --version smoke`.

From a clean checkout of the campaign's pinned Marin commit:

```bash
cd experiments/evaluation/campaigns/eval-policy
./validate-campaign.sh
./launch-campaign.sh --model Qwen-Qwen3.6-35B-A3B
./launch-campaign.sh --submit --model Qwen-Qwen3.6-35B-A3B
```

The first launch command is a dry run. `--suite nonagentic` and `--suite agentic` select one full
mechanism. Repeat `--evalchemy NAME` or `--harbor NAME` to select individual configs; pair a selector
with its corresponding suite to avoid launching the other mechanism. `--wait` waits for durable
records and requires CoreWeave object-store credentials.

Benchmarks with the same model config share a server within each suite. The launcher
submits a separate group for a benchmark-specific model override, such as Tau³'s larger
context window. Each group runs up to four evaluations concurrently. BixBench alone uses
the hosted MiniMax judge; the other evaluations can run while that judge starts.

The launcher writes generated effective configs beneath
`experiments/evaluation/campaigns/.launch-staging/` and local provenance snapshots beneath
`experiments/evaluation/campaigns/.launch-snapshots/`. Both paths are ignored by Git. Durable
`record.json` files contain the effective model and evaluator configuration, repository revision,
and Harbor policy digest.

Credentials are operational inputs. Export `HF_TOKEN`, `TOGETHER_API_KEY`, and CoreWeave object-store
credentials, or use the configured Google Secret Manager access. Set `EVAL_CAMPAIGN_SECRETS_ENV` only
when reading Hugging Face or Together credentials from a local environment file.

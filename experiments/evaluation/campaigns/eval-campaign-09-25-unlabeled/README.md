# Unlabeled evaluation campaign

This directory contains the campaign's model configs, benchmark selections in
`evaluations.yaml`, validation and launch commands.
The launcher records the Marin revision, resolved protocol choices and execution settings.
Curation publishes immutable TaskSpec corpora before candidate inference.

The agentic evaluations keep the names of the campaign's earlier Harbor runs, under which Evaldash lists
them. Their curated protocols in `experiments/evaluation/benchmarks/agentic.py` pin the same packages,
agents and limits:

| Evaluation | Benchmark and protocol | Agent | Trials | Input / reply tokens | Attempts at once |
|---|---|---|---|---|---|
| `tb2-recovery` | `terminal-bench-2.1`, `campaign-2026-09-25` | Terminus 2 | 3 | 65,536 / 16,384 | 16 |
| `ot-tblite-recovery` | `ot-tblite`, `campaign-2026-09-25` | Terminus 2 | 3 | 32,768 / 8,192 | 16 |
| `swebench-verified` | `swebench-verified`, `mini-swe-agent` | mini-swe-agent 2.1.0 | 1 | 32,768 / 8,192 | 16 |
| `ds-1000-local` | `ds-1000`, `terminus-2` | Terminus 2 | 1 | 32,768 / 8,192 | 32 |
| `bfclparity-pi` | `bfcl-parity`, `pi` | Pi 0.87.0 | 1 | 32,768 / 16,384 | 32 |
| `bixbench-pi` | `bixbench`, `pi` | Pi 0.87.1 | 1 | 65,536 / 16,384 | 16 |

Each gives the agent 1,800 seconds and the verifier 300, whatever the package sets. BixBench machines
have 10,240 MB of storage. The Pi protocols install their Pi version into each task image.
An agent's context is the smaller of its input limit and the served window. Pi's BixBench profile keeps
a 32,768-token compaction reserve, which a 32,768-token window cannot hold, so `campaign.yaml` selects
the `pi-32k` protocol, with a 16,384-token reserve, for the models served with that window.

Launch policy evaluations from the Marin revisions pinned in the
[eval policy](https://github.com/marin-community/marin/issues/10197#codebase-pins), not from `main`.
This directory on `main` can differ from the pinned configs; for example, its SWE-bench budget is
32,768 tokens, while the policy runs SWE-bench at 131,072. Launches with `--policy-finalization-config` must
write remote records directly under `gs://marin-eval-metadata/evals` or
`s3://marin-us-east-02a/marin/evals`, the roots EvalDash catalogs, and may resume only a
`<root>/<run_id>/results` directory there. EvalDash finds only `<root>/<run_id>/record.json`, so
nested result files such as replay outputs under `evals/ablations/` are not EvalDash runs.

The launcher leaves `EvalRunRecord.version` unset. Its checked-in inputs, config digests, and durable
records support reproduction and review; they do not attest that a run conforms to an accepted eval
policy. Policy conformance requires the independently trusted evidence and verifier described in
[issue #9458](https://github.com/marin-community/marin/issues/9458). The preflight and comparison work
in [PR #9461](https://github.com/marin-community/marin/pull/9461) must consume that trusted decision
before presenting a run as policy-conformant.

SWE-bench runs mini-swe-agent 2.1.0 as Harbor ran it.
Its 32,768-token budget includes the rendered prompt and the requested reply. Before each
completion, the agent counts the prompt, including tools and the generation prefix, through
the serving model's `/tokenize` endpoint. It caps the reply to the remaining context and the
model's output limit. When no reply fits, the episode ends normally and the existing patch is
graded. It does not truncate or summarize the conversation. A model config's
`agent.agent_kwargs.model_format` selects how the agent formats the model's history; K2 uses
`k2_horizon`.

`finalization.yaml` selects MiniMax for FinanceBench, OlympiadBench and BixBench grading.
Each judged benchmark starts the pinned 262,144-token TP8 MiniMax service on eight H100s
in the candidate's cluster before evaluation and releases it afterward. The TaskSpec driver grades
answers once and preserves judge verdicts with the samples. FinanceBench and OlympiadBench judge
requests use constrained labels and escalate empty final answers through 128, 512, and 2048 output
tokens; BixBench sends its package's structured 1,024-token request, as Harbor's verifier did.

These stages finish before the run is recorded as successful. Jobs preserve raw attempts for
resume and finish as `artifact_failed` if fewer than 90% of selected trials are scored.
To resume a launch, repeat it for its one model and suite with `--run-id` set to the group ID it
printed; finished attempts are reused and infrastructure failures are retried.
The launch snapshot includes the finalization configuration and judge YAML. `record.json` records the effective FinanceBench, OlympiadBench and BixBench judge
identity in `evaluation.curated.settings.judge`. The generic evaluation CLI
accepts `--policy-finalization-config finalization.yaml`; `launch-campaign.sh` supplies the
absolute path to this YAML by default.

`evaluation_num_concurrent` in a model config limits simultaneous requests to that model's endpoint,
and an evaluation's `run.concurrency` limits its attempts at once; the smaller applies.
K2 is limited to four after its 16-request runs saturated serving and produced mostly timed-out
generations. Effective concurrency is recorded in `evaluation.curated.settings.concurrency`.

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

The first launch command is a dry run. `--suite nonagentic` and `--suite agentic` select one suite;
the two suites launch as separate jobs. Repeat `--eval NAME` to select individual evaluations.
`--wait` waits for durable records and requires CoreWeave object-store credentials.

The launcher writes generated effective configs beneath
`experiments/evaluation/campaigns/.launch-staging/` and local provenance snapshots beneath
`experiments/evaluation/campaigns/.launch-snapshots/`. Both paths are ignored by Git. Durable
`record.json` files contain the effective model and evaluator configuration and repository revision.

Credentials are operational inputs. Export `HF_TOKEN`, `TOGETHER_API_KEY`, and CoreWeave object-store
credentials, or use the configured Google Secret Manager access. Set `EVAL_CAMPAIGN_SECRETS_ENV` only
when reading Hugging Face or Together credentials from a local environment file.
The agentic suites run their task machines on Daytona.

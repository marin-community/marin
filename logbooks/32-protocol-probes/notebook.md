---
title: "Agent protocol probes on Qwen3-0.6B and Snowball v0.1"
source_repository: "marin-community/marin"
task_issue: "https://github.com/marin-community/marin-logbooks/issues/32"
related_issues:
  - "https://github.com/marin-community/marin/issues/10393"
  - "https://github.com/marin-community/marin/pull/10399"
  - "https://github.com/marin-community/marin/pull/10932"
  - "https://github.com/marin-community/marin/pull/11049"
  - "https://github.com/marin-community/marin/pull/11050"
research_branch: "codex/32-protocol-probes"
marin_base_revision: "f24a46889a8b3a16858a32d25ad5565532c4f456"
archive_template_revision: "f54b3aa0e6ee533e988d2b97e9db0d1523f934dd"
status: complete
---

# Agent protocol probes on Qwen3-0.6B and Snowball v0.1

## Question and decision criteria

- Goal and why it matters: marin-community/marin#10393 asks for structural
  evals that separate agent protocol failures from wrong answers. #10399 grades
  four checks (required-file submission, tool execution, thinking disabled, URL
  loading) into `diagnostics["protocol"]`; #10932 runs 32 probe tasks, 8 per
  check, as a validation source in the quick RL loop. These runs check that the
  probes and labels work on real vLLM serving before the quick loop relies on
  them, and record baseline rates for two models.
- Baseline and alternatives: no prior protocol measurement exists. Qwen3-0.6B
  is the cheapest model that runs the full path; Snowball v0.1 is the current
  agentic checkpoint.
- Evidence needed to choose: every probe row produces the label its trace
  shows, and a correct answer is not lost to serving or grading bugs.
- Constraints: one node per run; thinking turned off on every request so the
  thinking check applies to all 32 rows.

## Source

The branch base `f24a46889a` is the reviewed #10932 tip with both launchers:

- `experiments/post_training/inventory/protocol_probes_smoke.py`: Qwen3-0.6B on
  one H100, trains one dry step so the trainer evaluates the probes before and
  after it.
- `experiments/post_training/inventory/protocol_probes_eval.py`: a Snowball
  checkpoint served by vLLM on one 8xH100 node through SkyRL's `generate`
  entrypoint, with no policy loaded.

The recorded runs used earlier revisions of the same PR before its rebase and
cleanup. Both are parents of this branch's merge commit `608bf14d0c`, which
keeps the `f24a46889a` tree:

- `7de9f12c88`: Qwen3-0.6B run `probes4-dev`.
- `7b81a5e760`: Snowball v0.1 run `probes-eval4-dev`.

Between `7b81a5e760` and `f24a46889a`, `taskcompendium/protocol.py` and
`taskcompendium/examples/protocol_probes.py` are unchanged. The later revision
reworks how RolloutEngine passes recorded turns to grading and removes tests.

## Entries

### 2026-11-04 00:17-01:10 UTC - Qwen3-0.6B smoke on one H100

- Question: do the 32 probes run end to end through the quick loop's path?
- Commands, from the run revision:
  `uv run python -m experiments.post_training.inventory.protocol_probes_smoke --version probes<N>-dev --run`.
- Runs, Iris coordinators under `/dml/users-dml-checkpoints-protocol-probes-smoke-*`:
  - `probes-dev` (`…-69622d00`): all 24 tool, URL and file probes failed with
    HTTP 400 and no recorded message. Led to keeping the server's error message
    with a failed rollout.
  - `probes-dev` (`…-11bd2e81`): refused, because run 1 had sealed that version
    label.
  - `probes3-dev` (`…-e7e66ac3`): the recorded 400 was vLLM's
    `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`.
    The thinking probes were not assessed, because the worker merged
    `enable_thinking: false` into the request after the harness recorded it.
    Fixed by enabling the Hermes tool parser and recording the settings the
    transport applied per turn.
  - `probes4-dev` (`…-76782a11`, revision `7de9f12c88`): the measured run.
- Result (`probes4-dev`, before training, 32 rows): average score 0.19.

  | Check | Labels |
  | --- | --- |
  | tool execution (8) | `no_invocation` 8 |
  | URL loading (8) | `not_attempted` 8 |
  | required file (8) | `missing_file` 8 |
  | thinking disabled (32) | `respected` 12, `reasoning_emitted` 20 |

- Evidence:
  `s3://marin-us-east-08a/marin/users/dml/checkpoints/protocol-probes-smoke/probes4-dev/exports/dumped_evals/global_step_0_evals/protocol-probes.jsonl`;
  per-row extract in `analysis/traces.json` under `qwen3-0.6b`.
- Interpretation: Qwen3-0.6B never calls a tool; it answers in prose, often
  inventing a value. The labels match the traces. The model is useful only as a
  plumbing check.

### 2026-11-04 18:02-20:18 UTC - Snowball v0.1 on one 8xH100 node

- Question: what are the protocol rates of a real agentic checkpoint?
- Model: `open-athena/Marin-Snowball-67B-A2B-v0.1` at revision
  `adc0078a19a4383bc998fcdb43867b3360baf5f3`, served by vLLM with the Hermes
  tool parser, broadcast weight sync, pipeline parallel 1.
- Commands, from the run revision:
  `uv run python -m experiments.post_training.inventory.protocol_probes_eval --version probes-eval<N>-dev --run`.
- Runs, Iris coordinators under `/dml/users-dml-checkpoints-protocol-probes-eval-snowball-v0.1-*`:
  - `probes-eval-dev` (`…-b25b389e`): failed loading the probe rows, built under
    TaskSpec schema 0.28 while main read 0.30. Fixed by putting the schema
    version in the probe artifact's identity.
  - `probes-eval2-dev` (`…-e46698d9`) and `probes-eval3-dev` (`…-143ff52a`):
    average score 0.19. Seven exact-nonce answers graded as
    `<nonce><|eot_id|>`: Snowball's tokenizer EOS is `<|end_of_text|>`, its
    template ends turns with `<|eot_id|>`, and the client stripped only the
    EOS. Fixed in `7b81a5e760` by stripping whichever special token the server
    stopped at.
  - `probes-eval4-dev` (`…-67e5d2f3`, revision `7b81a5e760`): the measured run,
    8 minutes once the model was staged.
- Result (`probes-eval4-dev`, 32 rows): average score 0.53.

  | Check | Labels | Correct answers |
  | --- | --- | --- |
  | tool execution (8) | `executed` 7, `not_executed` 1 | 3 |
  | URL loading (8) | `retrieved` 8 | 4 |
  | required file (8) | `submitted` 2, `missing_file` 6 | 2 |
  | thinking disabled (32) | `respected` 27, `reasoning_emitted` 5 | |

- Evidence:
  `s3://marin-us-east-08a/marin/users/dml/checkpoints/protocol-probes-eval/snowball-v0.1/probes-eval4-dev/exports/dumped_evals/eval_only/protocol-probes.jsonl`;
  per-row extract in `analysis/traces.json` under `snowball-v0.1`.
- Interpretation:
  - Snowball calls the tool on nearly every tool and URL probe. The wrong
    answers there wrap the nonce in prose ("The token is …"), which the
    exact-match grader rejects as the instruction asks.
  - The file misses are mostly a malformed call: Snowball writes
    `<tool_call name="shell">` with a JSON body in its text and follows it with
    an observation it wrote itself. The Hermes parser does not accept that
    markup, so no command ran and no file was written.
  - An earlier comment on #10393 described these traces as fabricated tool
    output; the corrected reading is on the issue
    (marin-community/marin#10393, issuecomment-6641107802).

### 2026-11-06 19:30 UTC - Evaluate the probes on checkpoints, outside the quick loop

- Change of direction: the probes measure a model and its serving setup, not
  RL progress, so they no longer run inside the quick RL loop or add a grading
  layer. On the file, tool and URL probes the nonce makes the ordinary reward
  the protocol result, so the per-check labels in `diagnostics["protocol"]`
  were dropped; failure modes are classified offline from the dumped
  `messages` and `response_token_ids`. Rewarding protocol compliance directly
  was considered and rejected for now: outcome reward already covers the file,
  tool and URL checks, and per-action rewards invite gaming.
- Marin contribution PRs: marin-community/marin#11049 supersedes #10399 and
  #10932 with the library pieces only (probe tasks, `fetch-v1` provider,
  per-dataset turn and task-machine config in the SkyRL task spec, `messages`
  in eval dumps). marin-community/marin#11050 carries the end-of-turn fix with
  a regression test. The model-server error capture from #10399 was dropped.
- Source: this branch merges the #11049 head
  `8f94ccc9188e2b3755acb8b2ffaccce46f065d83`, which is based on Marin `main`
  `6b9428b21b916b97e55fc3851d09e3fce079c84c`. The tree matches #11049 apart from
  this directory and the launcher; the earlier `protocol_probes_eval.py`,
  `protocol_probes_smoke.py`, the grading layer and the quick-loop wiring are
  removed. The recorded runs above keep their exact source revisions.
- Launcher: `experiments/post_training/inventory/protocol_probes.py` publishes
  the probe dataset and evaluates it against one checkpoint on one 8xH100 node
  through SkyRL's evaluation-only `generate` entrypoint, starting from the
  quick loop's `training_recipe()`, which stays unchanged. It enables vLLM's
  Hermes tool parser and turns thinking off unless `--thinking` is passed.
  `--model` is `datakit-sft` (the quick loop's starting checkpoint, adopted from
  its existing copy) or `snowball-v0.1`; other 67B-A2B checkpoints in HF format
  need a `MODELS` entry. The default cluster is `cw-us-east-02a`, where the SFT
  weights live.

  ```bash
  uv run python -m experiments.post_training.inventory.protocol_probes \
    --model snowball-v0.1 --version "<label>-dev" --run
  ```

- Validation: the graph builds for both models without submission; no run
  has been launched from this revision yet.

### 2026-11-06 20:30 UTC - Drop the thinking probe after #11049 review

- Why: review of #11049 found the `thinking_disabled` probe's exact-match
  reward independent of whether the model reasoned, since the transport moves
  reasoning out of the reply. The thinking-off check already covers every row
  through the launch setting, as in the 2026-11-04 run (27 of 32 rows).
- Change: merged #11049 at 78a04d1846. The probe set is now three checks, 24
  rows under `PROBE_ARTIFACT_VERSION = "2026.11.06.1"`. `PROBE_TURNS` and
  `locked_grader_environment` moved into the launcher. #11049 also unified
  the task-machine rule (`task_machine_reason`) that `task_machine_by_dataset`
  uses.
- Validation: the launcher imports and builds 24 probe rows with thinking off;
  no run launched.

### 2026-11-06 22:30 UTC - Pretrained Snowball on one 8xH100 node

- Question: how much of Snowball v0.1's protocol behavior comes from
  post-training? The baseline is the pretrained checkpoint the SFT stages start
  from.
- Model: the June pretrain cooldown export
  `s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/d819cbc63780bd86/`
  (`--model snowball-base`). It is served with Snowball v0.1's tokenizer and
  chat template, so it sees the same tool prompts. The vocabularies match by
  id; v0.1 only names reserved tokens 128005 and 128011 `<tool_call>` and
  `</tool_call>`. The branch includes the #11050 end-of-turn fix (eb858b6060).
- Command, from 3c51814b51:
  `uv run python -m experiments.post_training.inventory.protocol_probes --model snowball-base --version probes-base1-dev --run`.
  Coordinator
  `/dml/users-dml-checkpoints-protocol-probes-eval-snowball-base-probes-base1-dev-coordinator-3d0b7893`
  on `cw-us-east-02a`. It waited over an hour in the Kueue queue for a free
  node, then ran in minutes.
- Dump:
  `s3://marin-us-east-02a/marin/users/dml/checkpoints/protocol-probes-eval/snowball-base/probes-base1-dev/exports/dumped_evals/eval_only/protocol-probes.jsonl`.
- Result (24 rows, thinking off): average score 0.00, no tool messages, no
  reasoning tokens.

  | Check | Snowball v0.1 (2026-11-04) | Pretrained base |
  | --- | --- | --- |
  | tool executed | 7 of 8 | 0 of 8 |
  | page retrieved | 8 of 8 | 0 of 8 |
  | answer file submitted | 2 of 8 | 0 of 8 |

  The first turns, up to `<|eot_id|>`, show why. File probes narrate the call
  ("I'll use the shell tool to write the file.") without making it. Tool probes
  echo the template's system tools header. Seven of eight URL probes start with
  an empty turn. No reply contains `<tool_call>`.
- Caveat: all 24 replies hit the 1024-token length stop. The base export's
  generation config ends sequences only at `<|end_of_text|>`, so vLLM did not
  stop at `<|eot_id|>`, and each reply continued with invented chat turns.
  The first turns already contain no tool call, so stopping at `<|eot_id|>`
  would not change the rates.
- Conclusion: the pretrained model follows none of the three protocols. The
  tool and URL behavior of Snowball v0.1 comes from post-training.

## Final decision

- Selected approach: evaluate the probes against checkpoints with the launcher
  in `experiments/post_training/inventory/protocol_probes.py`, outside the
  quick RL loop. On the file, tool and URL probes the reward is the protocol
  result; classify failures offline from the dumped `messages` and
  `response_token_ids`. This replaces the earlier decision to grade per-check
  labels into `diagnostics["protocol"]` (#10399) and run the probes in the quick
  loop (#10932); see the 2026-11-06 entry.
- Marin contribution PRs: marin-community/marin#11049 (probe tasks, `fetch-v1`
  provider, per-dataset SkyRL task-spec config, `messages` in eval dumps) and
  marin-community/marin#11050 (end-of-turn fix). Both are open drafts and
  supersede #10399 and #10932. The launcher stays on this branch.
- Reproduction: check out this branch and run
  `uv run python -m experiments.post_training.inventory.protocol_probes --model <model> --version <label>-dev --run`.
  The 2026-11-04 runs used the earlier launchers at their recorded revisions.
- Remaining uncertainty: 8 variants per check gives coarse rates, and the tool
  and URL variants share one prompt. The launcher has not been run yet; the
  next run should confirm that it reproduces the 2026-11-04 Snowball rates.

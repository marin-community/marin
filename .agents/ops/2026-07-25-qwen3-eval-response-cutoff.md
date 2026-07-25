---
date: 2026-07-25
system: marin-eval
severity: degraded
resolution: mitigated
pr: none
issue: https://github.com/marin-community/marin/issues/6865
---

# TL;DR

- Qwen3-32B evaluation requests reached the model through `iris.oa.dev`, but
  interactive responses could end after thinking without visible answer text.
- Thinking-enabled probes exhausted 32- and 256-token output budgets with
  `finish_reason=length` and no final content. Disabling thinking returned
  `391` in four tokens for the same multiplication prompt.
- Marin's remote inference client also discarded the controller-minted
  federated capability URL and rebuilt a child-cluster URL.
- Remote inference now uses the controller's capability URL, and Marin Serve
  exposes the maximum output-token budget in its main toolbar.
- Qwen3-32B requires 8 host CPUs and 128 GiB of host memory for the tested
  H100x2 serving configuration.
- The final TB2-lite record was `failed`: `bn-fit-modify` completed with reward
  0, while `adaptive-rejection-sampler` timed out after 900 seconds. Harbor
  persisted both trials, two trajectories, `samples_harbor.parquet`, and the
  aggregate result.

# Original problem report

The Qwen3-32B evaluation needed an end-to-end validation after Iris capability
URLs moved to `iris.oa.dev`. During manual use, the Marin Serve UI appeared to
cut responses off after the model finished thinking.

# Investigation path

1. The first H100x2 attempt used the inherited 64 GiB host-memory request. The
   serving process was killed after loading 47 of 707 weight shards.
2. A 128 GiB retry loaded all 707 shards. vLLM reported 30.59 GiB of model
   memory per GPU and registered the inference endpoint.
3. A child-cluster token paired with `iris.oa.dev` returned
   `authentication required`. A child-cluster URL paired with the child token
   returned `Forbidden` because the child ingress was IP-restricted.
4. Iris PR #7627 made the parent controller relay child capability URLs through
   `iris.oa.dev`.
5. Marin still used `RemoteInferenceConfig.capability_origin` to reconstruct
   the URL. `lib/marin/src/marin/inference/iris.py` ignored
   `MintEndpointTokenResponse.capability_url`, sending external evaluators back
   to the child origin.
6. A local controller-side fix used the minted capability URL. Daytona then
   drove concurrent Terminal-Bench requests through `iris.oa.dev`; vLLM
   returned HTTP 200 and sustained up to 198 generated tokens/s without a
   401 or 403.
7. Direct requests isolated the UI symptom. Thinking-enabled requests used all
   32 or 256 completion tokens and returned `finish_reason=length` with no
   answer. `enable_thinking=false` returned `391` with
   `finish_reason=stop`.
8. Harbor finished with 0 of 2 trials solved. `bn-fit-modify` completed with
   reward 0 and no exception. `adaptive-rejection-sampler` returned
   `AgentTimeoutError` after 900 seconds. The run record correctly used
   `status=failed`; 41 S3 objects included both trial results, both
   trajectories, normalized samples, and `harbor_result.json`.

# User course corrections

- The user confirmed that the JWT deployment had been repaired and directed
  the validation to rebase onto main before changing authentication code.
- The user required the job to request only the CPU and memory needed. The
  successful serving shape used 8 CPU and 128 GiB instead of the inherited
  48 CPU and 1 TiB defaults.
- After reproducing the thinking-only response, the user chose an explicit
  max-token control in Marin Serve instead of changing Qwen's thinking mode.

# Root cause

Two boundaries caused separate failures. Marin's remote inference integration
treated the configured capability origin as authoritative even after Iris
started returning the complete parent-relayed URL. The resulting URL bypassed
the public parent and reached a restricted child ingress.

The apparent response cutoff was normal OpenAI-compatible completion behavior:
Qwen3-32B spent the entire `max_tokens` budget on its hidden reasoning stream.
vLLM returned `finish_reason=length` without final answer content. The UI did
not make the active output budget obvious because it lived inside a collapsed
sampling panel.

# Fix

`lib/marin/src/marin/inference/iris.py` now prefers the capability URL returned
by the controller. The configured origin remains a fallback for controllers
that cannot construct a public URL.

`lib/marin/src/marin/inference/dashboard/src/App.vue` keeps the max-token input
visible in the shared chat and completion toolbar. Temperature and top-p remain
in the secondary sampling panel.

`experiments/evaluation/serve/models/Qwen/Qwen3-32B.yaml` declares 8 CPU and
128 GiB of host memory for H100x2 serving.

# How OPS.md could have shortened this

Add a task-probing note to `lib/iris/OPS.md` under Task Operations: compare the
mint response's capability URL with the URL consumed by the external client,
and probe both thinking-enabled and thinking-disabled requests while recording
`finish_reason` and completion-token usage. This separates relay failures from
normal output-budget exhaustion without printing the capability token.

# Artifacts

- Evaluation job: `/loom/eval-20260725-163141-qwen3-32b-5858`
- Evaluation record:
  `s3://marin-us-east-02a/marin/eval-metadata/runs/20260725-163141-qwen3-32b-tb2-lite-5113/record.json`
- Iris relay PR: https://github.com/marin-community/marin/pull/7627
- Tracking issue: https://github.com/marin-community/marin/issues/6865

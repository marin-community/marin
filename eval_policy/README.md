# Verified evaluation cohorts

The launcher applies the [September 16](https://github.com/marin-community/marin/issues/9193) or [September 24](https://github.com/marin-community/marin/issues/9409) policy under `eval-policy-2026-09-16-verified` or `eval-policy-2026-09-24-verified`. Each benchmark gets its own H100 serving group, sized by the model YAML's `resource_hint.gpu.H100` value. Harbor always executes the vendored source, so new Harbor runs fail admission under these historical labels. Reproduce historical Harbor runs from their recorded Marin checkout.

Run from the repository root:

```bash
uv run python -m eval_policy.launch eval-policy-2026-09-24-verified \
  --model-config /absolute/path/to/model.yaml \
  --evals math500,gsm8k-0shot \
  --federated-cluster cw-rno2a
```

Use `--evals` to select Evalchemy benchmarks on the current checkout. `--dry-run` validates and prints the launches without submitting them. Submissions do not wait for results; if one fails, earlier jobs remain active. AIME24 launches ten repeats per policy.

September 24 `mmlu-pro`, `gpqa-diamond`, `cruxeval`, `ifbench`, `mrcr`, and `nupa` require `--artifact-dir` pointing to a checkout of the [campaign artifact](https://huggingface.co/datasets/open-athena/marin-eval-policy-2026-09-24/tree/0ba77c019caa88de9583bb0a9d55f66e50aa0f01) at revision `0ba77c019caa88de9583bb0a9d55f66e50aa0f01`. The example above needs no artifact checkout. Historical full-policy launches also require that artifact for Harbor configs and a prepared SOTOPIA-hard dataset inside the Marin workspace; the launcher does not prepare it.

Verified cohorts fix parsed benchmark configs and evaluator identities. Historical commit metadata remains in the admission rules so existing records keep their original eligibility. September 24 chat benchmarks set `enable_thinking` explicitly; September 16 keeps the model YAML setting. EvalDash separates model configurations by source-YAML fingerprint.

Each policy submission creates a fresh run ID for every benchmark and AIME24 repeat. An Iris retry
reuses that run ID and its per-eval StepSpec state, so it skips steps that already reached `SUCCESS`.
To publish a changed Harbor or Evalchemy config or runtime as a verified cohort, add a new version
to the policy maps, source digests, and runtime admission rules before launching. Preserve the
existing labels' admission rules for saved records.

Pi benchmarks require an explicit `thinking_format` in the model YAML's `agent.agent_kwargs` or the Harbor policy's agent kwargs. Choose `chat-template` or `qwen-chat-template` to match the model's request format; this does not select thinking on or off. Preflight constructs Pi to reject missing or invalid provider settings before submission.

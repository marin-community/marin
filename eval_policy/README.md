# Verified evaluation cohorts

The [September 16](https://github.com/marin-community/marin/issues/9193) and [September 24](https://github.com/marin-community/marin/issues/9409) policies use the labels `eval-policy-2026-09-16-verified` and `eval-policy-2026-09-24-verified`. Evalchemy benchmarks still require their original external evaluator commits and a checkout predating vendoring. Harbor benchmarks now use the current vendored runtime and are admitted without the former `21e0ea6a` Harbor pin. Their runtime provenance comes from the Marin artifact checkout; a shared cohort label no longer establishes identical Harbor runtime behavior across checkouts.

Use a separate checkout at [Marin revision `205558e1a7fc`](https://github.com/marin-community/marin/commit/205558e1a7fc3dd705b1132691798b09dc1f254c) to reproduce the original evaluator versions; it supports both cohorts and their original pins. Historical records remain readable.

The launcher gives each benchmark its own H100 serving group, sized by the model YAML's `resource_hint.gpu.H100` value. The following commands apply to that historical checkout.

Run from the repository root:

```bash
uv run python -m eval_policy.launch eval-policy-2026-09-24-verified \
  --model-config /absolute/path/to/model.yaml \
  --evals math500,gsm8k-0shot \
  --federated-cluster cw-rno2a
```

Use `--evals` to select Evalchemy benchmarks on that historical checkout. `--dry-run` validates and prints the launches without submitting them. Submissions do not wait for results; if one fails, earlier jobs remain active. AIME24 launches ten repeats per policy.

September 24 `mmlu-pro`, `gpqa-diamond`, `cruxeval`, `ifbench`, `mrcr`, and `nupa` require `--artifact-dir` pointing to a checkout of the [campaign artifact](https://huggingface.co/datasets/open-athena/marin-eval-policy-2026-09-24/tree/0ba77c019caa88de9583bb0a9d55f66e50aa0f01) at revision `0ba77c019caa88de9583bb0a9d55f66e50aa0f01`. The example above needs no artifact checkout. Historical full-policy launches also require that artifact for Harbor configs and a prepared SOTOPIA-hard dataset inside the Marin workspace; the launcher does not prepare it.

Verified cohorts fix parsed benchmark configs. Evalchemy retains its historical runtime admission rules; Harbor records use Marin artifact provenance without a separate runtime gate. September 24 chat benchmarks set `enable_thinking` explicitly; September 16 keeps the model YAML setting. EvalDash separates model configurations by source-YAML fingerprint.

Each policy submission creates a fresh run ID for every benchmark and AIME24 repeat. An Iris retry
reuses that run ID and its per-eval StepSpec state, so it skips steps that already reached `SUCCESS`.
To publish a changed benchmark config as a verified cohort, add a new version to the policy maps
and source digests before launching. Evalchemy runtime changes also require updated runtime
admission rules. Harbor runtime changes are recorded through the Marin artifact checkout.

Pi benchmarks require an explicit `thinking_format` in the model YAML's `agent.agent_kwargs` or the Harbor policy's agent kwargs. Choose `chat-template` or `qwen-chat-template` to match the model's request format; this does not select thinking on or off. Preflight constructs Pi to reject missing or invalid provider settings before submission.

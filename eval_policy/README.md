# Verified evaluation cohorts

The launcher runs any subset of the [September 16](https://github.com/marin-community/marin/issues/9193) or [September 24](https://github.com/marin-community/marin/issues/9409) policy under `eval-policy-2026-09-16-verified` or `eval-policy-2026-09-24-verified`. Each benchmark gets its own H100 serving group, sized by the model YAML's `resource_hint.gpu.H100` value.

Run from the repository root:

```bash
uv run python -m eval_policy.launch eval-policy-2026-09-24-verified \
  --model-config /absolute/path/to/model.yaml \
  --evals math500,gsm8k-0shot \
  --federated-cluster cw-rno2a
```

Omit `--evals` to run the full policy. `--dry-run` validates and prints the launches without submitting them. Submissions do not wait for results; if one fails, earlier jobs remain active. AIME24 launches ten repeats per policy.

September 24 `mmlu-pro`, `gpqa-diamond`, `cruxeval`, `ifbench`, `mrcr`, `nupa`, and all Harbor benchmarks require `--artifact-dir` pointing to a checkout of the [campaign artifact](https://huggingface.co/datasets/open-athena/marin-eval-policy-2026-09-24/tree/0ba77c019caa88de9583bb0a9d55f66e50aa0f01) at revision `0ba77c019caa88de9583bb0a9d55f66e50aa0f01`. The example above needs no artifact checkout. A full September 24 launch does. SOTOPIA-hard also requires `--sotopia-dataset-dir` pointing to a prepared dataset inside the Marin workspace; the launcher does not prepare it.

Verified cohorts pin parsed benchmark configs and evaluator commits. Changing either requires a new cohort label. September 24 chat benchmarks set `enable_thinking` explicitly; September 16 keeps the model YAML setting. EvalDash separates model configurations by source-YAML fingerprint.

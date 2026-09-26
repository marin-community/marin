# Verified evaluation cohorts

The launcher accepts a subset of either [September 16 policy](https://github.com/marin-community/marin/issues/9193) or [September 24 policy](https://github.com/marin-community/marin/issues/9409). Each benchmark launch gets its own H100 serve group, sized by the model YAML's `resource_hint.gpu.H100` value. September 16 AIME24 submits ten seeded launches; September 24 AIME24 submits ten launches without an outer seed.

```bash
uv run python -m eval_policy.launch eval-policy-2026-09-24-verified \
  --model-config /absolute/path/to/model.yaml \
  --evals math500,gsm8k-0shot \
  --federated-cluster cw-rno2a
```

Omit `--evals` to submit the full policy. Submissions use `--no-wait`, so the script reports each Iris group after submission without waiting for its result. A failed submission stops the script; earlier groups remain active. No benchmark is required merely because another benchmark was selected.

Pass `--dry-run` to check every selected benchmark and serving slice without submitting jobs.

Some September 24 configs come from the [campaign artifact](https://huggingface.co/datasets/open-athena/marin-eval-policy-2026-09-24/tree/0ba77c019caa88de9583bb0a9d55f66e50aa0f01). Supply a checkout of revision `0ba77c019caa88de9583bb0a9d55f66e50aa0f01` with `--artifact-dir`. SOTOPIA-hard also needs `--sotopia-dataset-dir` pointing to its prepared dataset inside the Marin workspace; other subsets do not need that dataset. FinanceBench uses the checked-in config because the artifact config omits the judge required by the current launcher.

The new cohort labels end in `-verified`. Historical `eval-policy-updated` and `eval-policy-2026-09-24` runs keep their original labels and remain available in EvalDash. Historical and all-cohort comparisons carry an asterisk because their settings are not verified as comparable. Each verified contract pins its parsed benchmark YAML and evaluator commits: both current cohorts use Evalchemy `c131e5ab84d3014490549ec06d45deb13b5673c2` and Harbor `21e0ea6a0cc1a0b617aebd86988ea93e1795f84a`. These commits differ from the historical commands in the issues. A runtime or policy-config change requires a new verified cohort version; existing cohorts keep their pins.

The model catalog YAML must declare an `H100` GPU count and a matching serving topology. The record stores its normalized source configuration and effective serving configuration. EvalDash appends a source-configuration fingerprint to the model name in comparisons, so a serving or sampling change cannot reuse the same model row. September 24 requires an explicit `enable_thinking` value on each chat benchmark. September 16 keeps the model YAML's original setting.

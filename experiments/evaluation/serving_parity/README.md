# Serving parity calibration

This tooling compares Marin's serving composition on H100 and single-host TPU
(v6e-8 or v5p-8), with an additional multi-host v4-32 arm. It runs the vendored
Evalchemy client on the frozen GSM8K and BoolQ subsets in `manifest.json` and
collects echoed teacher-forced scores across the 2048-token attention window.
No scheduled workflow is enabled; fresh calibration, reference review, final
thresholds, wall time and cost review are required before activation.

Every candidate uses the adopted dependencies from
`marin.external_dependencies`, without a layer schedule override. Before serving,
the isolated TPU bootstrap builds the actual engine configuration and fails on
a schedule mismatch. It derives the expected layer schedule from the fork's
`grug_moe_layer_types` helper. The runtime trace must match that schedule and the
expected four KV-cache groups before evaluation starts.

## Run a bounded calibration

The launcher defaults to a dry run. Select `--hardware h100`, `v6e-8`, `v5p-8`,
or `v4-32`, a pinned regional checkpoint via `--weights`, and a fresh `--output`
prefix. Add `--submit` only for an allocation. Every arm has a one-hour runtime,
one-hour scheduling deadline and zero retries. H100 uses the fixed HF revision
and writes S3; TPU reads its verified regional GCS copy and writes to that same
region. v5p-8 uses one host, four physical chips and TP4 in us-east5; v6e-8 uses
TP8 there; v4-32 uses four hosts and TP16 in us-central2. Multi-host allocation
and Ray lifecycle are provided by `marin.inference.iris_tpu_vllm`.

The stock v4 tile selector may not support this geometry. The optional
`--diagnostic-v4-tiles` flag instruments a private environment with experimental
tiles, records the original/patched source hashes and marks the result
**ineligible for shipped-stack PASS**. It never changes the dependency pin or
model schedule. Such a diagnostic cannot serve as a gate reference either.
Unmodified H100/v6e/v5p arms can establish the shipped-stack calibration while
a supported v4 tile selector is developed and adopted separately.

The launcher archives source/job identity, hardware profile, pins, preflight,
trace/kernel patch hashes, serial/concurrent teacher responses, evaluator wire
payloads, sample coverage and per-item results. Instrumentation and environment
changes are scoped to the arm and restored on exit. Endpoint capabilities and
credentials must not appear in command arguments or persisted metadata.

The teacher request uses `max_tokens=0`, but the pinned vLLM may internally
report one completion token. Acceptance requires exactly the frozen prompt's
encoded token IDs and echo length, not a zero usage counter. An extra sampled
suffix is rejected. BoolQ's actual client uses one unscored generated token;
GSM8K uses greedy generation with a 2048-token cap. Changed per-item length
finishes fail separately from accuracy.

## Compare complete, reviewed artifacts

```sh
python -m experiments.evaluation.serving_parity.compare \
  --arm h100=H100_PREFIX --arm v6e-8=V6E_PREFIX --arm v5p-8=V5P_PREFIX \
  --reference REVIEWED_IMMUTABLE_PREFIX --output comparison.json
```

Require at least three distinct prefixes with distinct recorded hardware
identities, and labels matching those identities. Missing results, experimental
arithmetic patches and missing provenance cannot produce PASS. Model revision,
frozen dataset manifest and evaluator-source identity must match; TPU peers must
also use the same adopted pin. Every arm is compared with the reviewed frozen
reference and every other arm.

A reference requires explicit review. Never replace it automatically after a
failed run. Initial thresholds are mean absolute logprob difference 0.025 pooled
and 0.04 per position band, top-1 disagreement 0.03, and absolute combined-SEM
accuracy z at most 2. They remain provisional until fresh calibration. Historical
fixed-versus-H100 controls and local protocol tests do not establish current
accelerator parity. Run history belongs in the tracking issue, not these durable
instructions or a squash-merge description.

The proposed nightly time is 13:30 UTC, after the upstream checks. Scheduling
remains disabled until reviewed calibration evidence and cost/timing approval.

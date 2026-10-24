# Serving parity calibration

This canary checks the Marin serving composition across H100, single-host TPU,
and multi-host TPU. It uses the existing vendored Evalchemy client and campaign
YAMLs, with frozen 100-document GSM8K and BoolQ manifests. A separate echoed
teacher-forced request crosses the 2048-token attention window.

The candidate configuration must not inject `layer_types`: its CPU preflight
checks the configuration that Marin actually ships. Until the relevant fixes
reach the shipped pin, report that preflight as `KNOWN_FAIL`. The temporary
`--workaround-pin` arms explicitly use qualified tpu-inference ae387b8 and the
layer schedule override. They measure cross-hardware parity, **not shipped-pin
correctness**. Main's a8fc5c9 also lacks the routing, storage and head-padding
stack; an override alone is insufficient. Remove workaround arms after adoption.

The v4 arm applies a hash-guarded tile adaptation in its private environment.
The patch is not a production tpu-inference change. Multi-host launch depends on
Marin PR #10535. Every arm has a one-hour runtime and one-hour scheduling bound,
zero retries, and a fresh artifact prefix. CoreWeave writes S3; TPU writes GCS.
Never put credentials or endpoint capabilities in command arguments or reports.

## Calibration

Run `python -m experiments.evaluation.serving_parity.launch --help` for the
per-arm launcher; it defaults to dry run. Use `--hardware h100`, `v6e-8`, or
`v4-32`, a pinned checkpoint location, and a new `--output` prefix. TPU
workaround arms require the explicit `--workaround-pin` flag. Add `--submit`
only for an actual allocation. H100 uses the fixed HF model revision; TPU
uses its already-verified regional GCS copy.

The launcher gates the actual TPU layer trace and four KV groups before running
work. It archives serial/concurrent teacher responses, actual evaluator HTTP
payloads, full coverage and per-item results. BoolQ must emit echo requests with
one generated token; the scored continuation must not include a generation
suffix. GSM8K must be greedy with a 2048 output cap. Changed per-item length
finishes fail separately from accuracy.

Compare three completed arm prefixes with:

```sh
python -m experiments.evaluation.serving_parity.compare \
  --arm h100=H100_PREFIX --arm v6e=V6E_PREFIX --arm v4=V4_PREFIX \
  --reference REVIEWED_IMMUTABLE_PREFIX --output comparison.json
```

References require explicit review; a failed or missing run must never replace
them automatically. Initial thresholds are mean absolute logprob difference
0.025 pooled / 0.04 per position band, top-1 disagreement 0.03, and absolute
combined-SEM accuracy z at most 2. These are provisional until fresh calibration.
Historical fixed-versus-H100 item303 passes; the original global-layer-windowing
bug fails pooled and post-2048 band checks. Local mock-client runs validate the
protocol and artifact plumbing, not model quality.

The planned schedule is 13:30 UTC, after upstream Evalchemy's 07:00 UTC and
`tpu-inference`'s 11:00 UTC canaries. Upstream checks fork HEAD; this check belongs
in Marin because launcher, adopted pins, registry, policy configuration and
client behavior jointly determine the stack users run. Workflow activation and
reference freezing follow calibration; no scheduled workflow is enabled yet.

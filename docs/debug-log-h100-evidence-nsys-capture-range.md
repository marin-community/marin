# H100 evidence Nsight capture-range repair

## Initial status

The fourth reviewed H100 evidence attempt used source
`43237d5ea8a68c814d7b4d2356365fffe8fe765a` and immutable image
`ghcr.io/marin-community/iris-task-h100-evidence:dbbd9e4fe53e8ec7ad2c8d409dbaa0351ac064ff@sha256:945f44cca0aa44be922c9d806e7b8e6b98915ed22323cca26ca89f23bf3a4e19`.
The source capsule authenticated, the frozen runtime and H100/tool preflight
passed, and all generated candidate compilation and authoritative loaded-image
SASS topology validation completed. Nsight Systems 2026.1.3 then rejected
`--stop-on-range-end=true` before launching the first case worker. The task
failed once after 71.45 seconds; failure retries were disabled and no relaunch
occurred.

The sealed negative artifact is
`lib/tile_lifetime/benchmarks/artifacts/h100_contract_map_evidence_fourth_launch_failure_43237d_v0/`.

## Hypothesis

`--stop-on-range-end` was deprecated and is absent from the pinned 2026.1.3
CLI. Its replacement is `--capture-range-end=stop`. Combined with
`--capture-range=cudaProfilerApi`, `stop` ends collection at
`cudaProfilerStop`, ignores later capture ranges, and lets the target process
continue. That preserves the runner's source-ordered schedule, which must
finish and serialize results after the bounded profiler range closes.

The repository's independent Iris Nsight wrapper already uses this exact pair.
The [NVIDIA Nsight Systems 2026.1 user
guide](https://docs.nvidia.com/nsight-systems/2026.1/UserGuide/index.html)
documents `none`, `stop`, `stop-shutdown`, and repeat policies for
`--capture-range-end`, and specifically distinguishes `stop` as the policy
that leaves the target running.

## Changes

- Replace the removed runner option with `--capture-range-end=stop`.
- Validate the pinned image's own `nsys profile --help` at build time with a
  bounded stdlib parser. The parser requires one unambiguous
  `--capture-range-end` declaration and one possible-values list containing
  exactly one `stop` value; it rejects the obsolete option.
- Exercise the production subprocess boundary in a runner regression and add
  parser mutation tests for missing, duplicate, malformed, and obsolete help.

## Results

- The new runner regression failed against the frozen obsolete command, then
  passed after the replacement.
- Focused runner, image-policy, and help-parser suites passed: 103 tests.
- The full tile-lifetime suite passed 949 tests. Its sole failure is the
  pre-existing ignored raw snapshot
  `stateful_scan_generated_h100/raw/mutation_per_key_r1_b1_t64_h32_k128_v128_bv32.stdout.log`,
  which is named by that artifact's `SHA256SUMS` but absent from a normal Git
  checkout.
- Pyrefly, Ruff, license, syntax, merge, structured-file, whitespace, and
  Markdown checks passed. Black requested formatting for two changed tests;
  the files were formatted before the final pre-commit rerun.

No image build, workflow dispatch, GPU query, evidence launch, retry, or
relaunch was performed during this repair. Independent source review remains
required before an image build.

## Future work

Build and publish a new immutable evidence image only after source review, then
submit a separately reviewed single H100 attempt. A successful image build is
necessary to execute the parser against the exact packaged 2026.1.3 help text;
local source tests use a representative excerpt of that closed option block.

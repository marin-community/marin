# Manual inference trials

Resolve a serving preset, run one fresh Iris deployment and replay pinned AgentX
data. GPU work runs independently of GitHub Actions and dashboards.
The [inference-perfmax skill](../../../.agents/skills/inference-perfmax/SKILL.md)
owns campaign choices and the adaptive experiment ladder.

```sh
uv run python -m experiments.evaluation.knotmeter.cli plan \
  --profile snowball-h100 --campaign snowball-throughput --stage probe --output /tmp/plan.json
uv run python -m experiments.evaluation.knotmeter.cli run \
  --profile snowball-h100 --campaign snowball-throughput --stage probe
```

`--profile` accepts a preset name or YAML path. Use `hero-gb200` for Hero/Mochi.
The presets request eight H100s on one host or sixteen GB200s on four hosts.
Hero uses retention zero and 256 GB per host from the tested DP16/C64 point.
A matched DP32 trial found no NUMA gain; binding stays off by default.
The pinned task image also lacks `numactl`. The campaign's higher-throughput
DP32/C128 point doubles the allocation and remains an explicit campaign choice.
See the [measured tradeoffs](https://github.com/marin-community/marin-logbooks/issues/57).
The preset uses Marin's selected runtime. These measurements used `0ac9dcb157`;
rerun the checks when that runtime changes.
Inspect live capacity through `use-iris` and coordinate resources before running.
For an additional allowed-host check, pass `--placement-uri` with a JSON document
such as `{"nodes":["NODE1","NODE2","NODE3","NODE4"]}`. Choose spare nodes from live
provider and Iris inventory immediately before submission, following
[Iris operations](../../../lib/iris/OPS.md#coreweave-gpu-operations). Record the
cluster, verification time and inventory digest beside that list. Check protected
training priorities and recheck delayed admissions. This guard checks admitted
hosts against the list after allocation; it does not constrain or reserve placement.

`--stage probe|screen|confirm` is required for both `plan` and `run`. Each trial
checks correctness and runs one selected stage, then stops. The agent decides
whether evidence warrants another trial.

| Stage | Measured seconds | Additional warmup/lane | Concurrency | Unsafe override |
| --- | ---: | ---: | --- | --- |
| Probe | 180 | 1 | min(C, 8) | yes |
| Screen | 900 | 10 | C | no |
| Confirm | 3,600 | 10 | selected C | no |

All stages use the full pinned 393-session corpus. Startup, correctness, native
warmup and drain do not count toward measured time. C is the profile's target
`session_concurrency`, overridable with `--session-concurrency`. Measurement
precedence is explicit flags, then stage overrides, then the shared recipe.
Use `--duration`, `--warmup-requests-per-lane`, `--concurrency` and
`--unsafe-override`/`--no-unsafe-override` to override a stage. `--concurrency`
sets the effective load directly, including overriding Probe's cap. The plan
records requested overrides, target C and effective settings. Durations below
900 seconds require the unsafe override.

Profile precedence is named flags, then `--set`, then the preset. Dotted
`--set FIELD=YAML` changes profile fields; unknown fields fail before submission.
For example:

```sh
uv run python -m experiments.evaluation.knotmeter.cli run \
  --profile snowball-h100 --campaign snowball-throughput --stage screen \
  --session-concurrency 64 --priority interactive \
  --set serving.max_num_batched_tokens=4096
```

Replace whole lists with `--set 'checks=[...]'`, or edit a copy of the preset
for longer check commands. Dotted paths address mappings; list indices are unsupported.

The runner resolves Marin's selected serving wheel once. `--vllm-release TAG`
uses another published release. `--vllm-manifest URI` accepts a saved trial/build descriptor
without changing `config/external/vllm/gpu.toml`. Reuse the fork's native build
paths. For Python-only changes, its
[precompiled build](https://github.com/marin-community/vllm/blob/0ac9dcb157ee86ccb447bc717259159c41ad456a/setup.py#L939-L951)
can reuse a compatible base wheel. Verify the reused native payload hashes and
cold-install the actual candidate wheel for validation. For native code changes,
use the [Docker wheel target](https://github.com/marin-community/vllm/blob/main/infra/release/README.md#gpu-build-configuration).
Supply the resulting `VllmGpuRelease` JSON: `release_tag`, `source_commit`,
`version`, `torch_backend`, `torch_version` and architecture-specific `wheels`
with `architecture`, `sm_targets`, `url` and `sha256`. Worker-accessible wheel
URLs must remain available for recovery. Optional `vllm.repo` and
`vllm.revision` record the source override; the revision must match the manifest.
The `vllm_release` object in `plan --output /tmp/plan.json` is a complete example.
This command does not refresh a fork, publish a release or introduce a build system.
Private `s3://` and `gs://` wheel URLs are copied to task-local storage and
verified against the manifest before installation. The native artifact retains
a receipt with the durable URI and hash; large wheel bytes stay outside the
trial archive.
`quantization` is the native vLLM selector. When that selector is a configuration
path such as `online`, set `weight_precision: fp8` to label the actual weight
format in upstream exports and reports. Keep the exact selective configuration
in `serving.extra_args` and verify its loaded methods with a serving check.

[recipe.yaml](recipe.yaml) supplies the shared harness, corpus and processor
pins. All stages retain upstream replay, cache busting and native warmup.
Data preparation runs on CPU before GPU submission. Probe and Screen cannot
produce accepted recommendations. Neither can a shortened Confirm or any unsafe
override. `--measure-failed-checks` retains failed checks and permits diagnostic
traffic at the selected stage. It preserves that stage's recipe and native validity.
Reports label performance with failed checks as diagnostic; those failures still
block recommendations.

Both presets use the [shared correctness suite](../inference_correctness/README.md).
It compares saved Levanter scores, engine batches, cache states and decode versus
recomputation. The Git fixture pins the manifest and a separate numerical policy.
The runner verifies references on CPU, then checks the actual server before
AgentX. It retains `check-N.json` and raw completion responses. Both the CLI exit and report
must pass. Missing or failed required checks stop ordinary measurements;
`--measure-failed-checks` permits diagnostic traffic. The cache is reset before native warmup.

Additional check commands receive `KNOTMETER_SERVER_URL` (the worker's `/v1` URL),
`KNOTMETER_MODEL`, `KNOTMETER_MODEL_REVISION`, `KNOTMETER_RUNTIME_SOURCE`,
`KNOTMETER_PROFILE` and `KNOTMETER_CHECK_OUTPUT`. An empty command stays pending.
External numerical checks declare `checkpoint_revision` and `runtime_source`;
a mismatched pass remains regression evidence. `kind: serving` identifies checks
that cannot replace numerical evidence. Recommendations require all required
checks, a numerical pass for this deployment and a valid Confirm window. Passing
these serving checks does not qualify RL learning or trainer probability ratios.

Outputs live at `MARIN_PREFIX/knotmeter/campaigns/CAMPAIGN/`. The selected
cluster's configured prefix supplies the default; `--marin-prefix` is an explicit
job override. Each invocation creates a new trial directory. Native retries
keep distinct attempt IDs, host records, exports and logs. Prepared AgentX workload
data uses a checksum-addressed archive; `plan.json` and `result.json` retain recovery inputs.
Keep large disposable intermediates in existing TTL scratch storage.

The leader uploads a snapshot after AgentX exits successfully and before stopping
followers or its server. A final snapshot records shutdown and capture failures.
Each receipt points to a checksum-addressed archive. A failed final upload leaves
the earlier receipt readable. The GPU job's failure still blocks a recommendation.
An abrupt kill before the first upload can leave no durable exports.

Plans, results and attempt records use `schema_version: 1`; older results and
attempts may have no version field. `measured_unix` records AgentX's zero exit.
Shutdown timestamps and `capture_phase` distinguish later work. Timestamps are
Unix seconds, `gpu_seconds` counts physical GPU-seconds, and latency is seconds.
`measurement.validity` holds native admission, local protocol checks, command
completion and final request counts by phase. Numerical reports remain separate.
`cost_complete` describes coverage; allocation entries label measured, estimated
or unknown usage. Dashboard code can read these records without parsing Markdown.

The terminal prints the CPU coordinator's Iris job ID, trial URI and recovery
commands. Its GPU child job is that ID followed by `/serve`. `run` waits
by default. It also captures pod allocation times, including setup before
Python starts, through the selected cluster's existing `kubectl` configuration.
Synchronous `run` exits nonzero after retaining the report if its coordinator
or GPU job failed. `recover` also exits nonzero for a failed GPU result, including
rejected submission. A completed diagnostic can exit zero with failed numerical
checks; inspect validity and check states before selecting a recommendation.
Capture records contain only timing and identity fields. If a pod expires, a retained scheduling
time plus Iris's terminal observation gives an estimated end time. Missing
timestamps remain unknown. Keep `run` attached for complete allocation tracking.
`--no-wait` returns after submission; detached runs may lose required timestamps
after Pod cleanup. Retain known usage and label estimates and unknowns. Incomplete
cost alone does not block recommendations. `status` reports current
state and does not continue allocation tracking. Interrupting the local wait leaves Iris running.
Cost is allocated GPU-hours: physical GPUs times allocated seconds, divided by
3,600, summed across hosts and attempts. It includes failed attempts.

```sh
uv run python -m experiments.evaluation.knotmeter.cli status --trial-uri TRIAL_URI
uv run python -m experiments.evaluation.knotmeter.cli recover --trial-uri TRIAL_URI
uv run python -m experiments.evaluation.knotmeter.cli cancel --trial-uri TRIAL_URI
uv run python -m experiments.evaluation.knotmeter.cli report --campaign-uri CAMPAIGN_URI
```

Recover from the original source revision recorded in `plan.json`. Recovery
collects terminal work and never submits GPUs. After coordinator loss, inspect
its `/serve` child and cancel or wait for it before recovery. Check provider
cleanup before retrying. Artifact uploads can finish after Iris first reports a
failed gang. If outputs are unavailable, rerun `recover` after capture finishes;
it rechecks durable receipts and does not allocate GPUs.
CPU coordinator preemption can stop the GPU child. Recovery retains that failed
trial's evidence and cost; it does not resume its measurement window.
The report prints the same Markdown saved as `summary.md`, including all attempts'
GPU cost. Trials without a result remain visible with unknown cost. Interrupted
partial exports retain attempt costs without qualifying a measurement.
One invocation measures one window; repeatability has not been measured. Use
separate fresh-server trials when noise could change a decision. Native multi-run
reuses the server and skips additional warmup after the first by default.
`report --recommend TRIAL_ID` requires a valid Confirm protocol, unchanged gang,
and passed required checks. Cost completeness is reported separately. Every physical host must retain the same Iris
attempt ID throughout profiling. The command writes `recommended.yaml` with a
frozen runtime. Add `--baseline TRIAL_ID` to show changes within a comparison
group; diagnostics with different controls remain separate rows.
Supplying only `--baseline` preserves the recorded recommendation.
A comparison fixes checkpoint, tokenizer, workload, AgentX protocol, GPU type,
prepared data and numerical criteria. Each full policy identity and applicability
remains in the check artifacts. Separate BF16 and FP8 policies can compare when
their thresholds and probes match. Older records without a criteria hash still
require the same full policy identity. Concurrency, GPU count, runtime and serving
settings may vary; throughput is reported per GPU.
The pinned processor uses `tp` as the total GPU count for normalization.
Its `is_multinode=false` selects a single serving pool even when it spans hosts.
Read the resolved profile and deployment checks for actual vLLM topology:
For example, the Hero preset's `tp=16` aggregate comes from TP1/DP16 across four hosts.
Keep experimental code and discarded approaches in the
[research archive](../../../docs/dev-guide/research-engineering.md).

Every command supports `--help`. The focused CPU compatibility test runs the real
pinned upstream tools against local HTTP; its GPU counts are normalization inputs:

```sh
uv run pytest -n 0 -m integration \
  experiments/evaluation/knotmeter/test_workload.py::test_pinned_tools_prepare_restore_replay_and_normalize_physical_gpus
```

# Check an inference endpoint

`TestSpec.run(endpoint)` returns a typed `CheckReport` with four results:
reference agreement, batch dependence, cache reuse and decode/recomputation.
Checking keeps observations in memory. Callers allocate through Iris, start
`vllm serve`, connect the endpoint, save the report and clean up serving.

Use a dedicated Marin vLLM endpoint with `VLLM_SERVER_DEV_MODE=1`, prefix caching,
compilation and CUDA graphs enabled. Checks reset its KV cache and send
concurrent requests to engine 0. The build must expose `/server_info`, including
`prefix_cache_retention_interval`. The HTTP adapter discovers the default model
through `/v1/models`; a model override belongs to `VllmEndpoint`, not the checks.

## Inputs

The checked-in [specs](https://github.com/marin-community/marin/tree/main/experiments/evaluation/inference_correctness/specs)
are `snowball-v0.1.1-vllm-h100.json` and
`hero-sft-step400407-vllm-gb200.json`. They retain the two BF16 targets and
provisional limits from [#11320](https://github.com/marin-community/marin/pull/11320).

`ArtifactRef` pins a URI and SHA256 digest. A spec points to a shared immutable
reference manifest containing the checkpoint/tokenizer identity, exact token
histories and saved Levanter scores. Execution variants can share that reference
without copying it or generating new goldens.

`TestSpec` contains the existing Iris `ResourceSpec`, `replicas`, Marin's
`ServeBackend`, `engine_kwargs` for `InferenceModelConfig`, and named
`tolerances` using `ErrorLimits`. Resources are per Iris task. The Snowball
binding requests one task with eight H100s; Hero requests four tasks with four
GB200s each. Multi-host startup remains with the caller. Placement and serving
settings describe a candidate; they do not establish its qualification.

`tolerance_version`, `provisional`, `rationale` and `evidence` retain the bounds'
identity and justification. Dtype belongs in `engine_kwargs`. The reference
owns checkpoint, tokenizer and EOS identity; `make_model_config()` carries those
pins into Marin's existing model configuration.

## Python: allocation, serving, checking and saving

This one-task Snowball example uses the existing Iris client and native vLLM
launcher. It runs from the repository root with access to the selected cluster
and artifact storage. Choose a new output directory for each run.

```python
import asyncio
from pathlib import Path
from uuid import uuid4

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from marin.inference.vllm_server import IsolatedCudaVllm, VllmEnvironment, VllmType
from marin.testing.inference.endpoint import VllmEndpoint
from marin.testing.inference.endpoint_checks import read_test_spec
from marin.testing.inference.results import CheckStatus
from rigging.filesystem.storage_path import prefix_join
from rigging.timing import Duration


def run_check(spec_path: str, output: str) -> None:
    spec = read_test_spec(spec_path)
    runtime = IsolatedCudaVllm(source=VllmType.MARIN_FORK)
    with VllmEnvironment(
        model=spec.make_model_config(),
        launcher=runtime,
        extra_args=["--enable-expert-parallel", "--enable-prefix-caching",
                    "--enable-prompt-tokens-details", "--max-logprobs", "50"],
    ) as server:
        async def compare():
            async with VllmEndpoint(server.server_url) as endpoint:
                endpoint.metadata["runtime_source"] = runtime.release.source_commit
                return await spec.run(endpoint)
        report = asyncio.run(compare())
    report.save(output)
    if report.status != CheckStatus.PASSED:
        raise RuntimeError(f"Correctness checks: {report.status}")


spec_path = "experiments/evaluation/inference_correctness/specs/snowball-v0.1.1-vllm-h100.json"
spec = read_test_spec(spec_path)
run_id = f"correctness-{uuid4().hex[:12]}"
output = prefix_join("s3://marin-us-east-02a/marin/inference/correctness/runs", run_id)
timeout = 10800
with connect_controller(cluster_name="cw-us-east-02a") as connection:
    with IrisClient.remote(connection.url, workspace=Path.cwd(), credentials=connection.credentials) as client:
        job = client.submit(
            Entrypoint.from_callable(run_check, spec_path, output),
            name=run_id, resources=spec.resources, replicas=spec.replicas,
            environment=EnvironmentSpec(sync_packages=["marin-core"], extras=["cpu"],
                                        env_vars={"VLLM_SERVER_DEV_MODE": "1"}),
            priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
            timeout=Duration.from_seconds(timeout),
        )
        try:
            job.wait(timeout=timeout)
        except BaseException:
            job.cancel()
            raise
```

The four public functions also return `CheckResult` independently. Within the
same endpoint context, the expanded form of `spec.run(endpoint)` is:

```python
from marin.testing.inference.endpoint_checks import check_reference, check_batch, check_cache, check_decode
from marin.testing.inference.results import CheckReport

report = CheckReport(
    reference=await check_reference(endpoint, spec),
    batch=await check_batch(endpoint, spec),
    cache=await check_cache(endpoint, spec),
    decode=await check_decode(endpoint, spec),
)
```

Runtime provenance is run metadata. The adapter records the observed `/version`;
callers can add their full runtime source commit to `endpoint.metadata` as above.
That label is not an expected-version argument or a numerical acceptance limit.
Checkpoint, tokenizer, dtype, hardware, EOS and cache configuration are still
checked against the immutable reference and execution settings.

## CLI and pytest

Both adapters call `spec.run(endpoint)`. Only `passed` exits successfully.
`--output` and `--correctness-output` request saving; omitting them leaves results
in memory. Output paths are artifact directories or storage URIs.

```sh
uv run python -m marin.testing.inference.check_endpoint \
  --spec experiments/evaluation/inference_correctness/specs/snowball-v0.1.1-vllm-h100.json \
  --url http://HOST:PORT/v1 --output correctness

uv run pytest --confcutdir=tests/cluster/vllm \
  tests/cluster/vllm/test_inference_correctness.py -o addopts= \
  --endpoint-url http://HOST:PORT/v1 \
  --correctness-spec experiments/evaluation/inference_correctness/specs/snowball-v0.1.1-vllm-h100.json \
  --correctness-output correctness

uv run python -m marin.testing.inference.check_endpoint \
  --spec experiments/evaluation/inference_correctness/specs/snowball-v0.1.1-vllm-h100.json --lookup
```

`--lookup` verifies saved artifact hashes, identities, histories and score
coverage without a server or GPUs. Only `found` exits successfully. `--model`
(`--correctness-model` in pytest) optionally overrides default-model discovery;
`--runtime-source` optionally records caller provenance. Neither is required.

`--confcutdir` excludes parent cluster deployment fixtures. `-o addopts=` clears
the default marker exclusion and 600-second session cap for this manual test.
Neither adapter deploys a server or imports Knotmeter or AgentX.

## Reports, units and versioning

`CheckReport` schema version 2 has `reference`, `batch`, `cache` and `decode`
fields of type `CheckResult`. Its derived `status` is `failed` if any check
fails, `passed` if all four pass, and otherwise `incomplete`. Missing reference
artifacts, batch overlap or required cache reuse cannot pass. Contradictory
checkpoint identities, non-finite scores and demonstrated corrupt state fail.

Each result retains its `test_spec`, `run_metadata`, `differences`, `coverage`,
`observations`, diagnostic `details`, HTTP `requests` and elapsed `seconds`.
Results returned by individual checks carry the same context as suite results.
The raw completion body, including an HTTP failure, is retained as `body_gzip`;
its SHA256 refers to the original uncompressed bytes. Artifact JSON encodes
those compressed bytes as base64. No request files are written while checking.

| Field | Meaning and units |
| --- | --- |
| `differences[*].positions` | Number of matching next-token positions compared. |
| `statistics.maximum`, `p99`, `p90` | Maximum and percentiles of per-position upper bounds on absolute probability error, in [0, 1]. |
| `statistics.fraction_above_0075` | Fraction of positions whose upper bound is strictly greater than 0.075. |
| `limits` | Corresponding `ErrorLimits`, using the same units. |
| `coverage[*]` | Required histories, observed overlap or cache hits; `not_required` is limited to unreachable periodic decode reuse. |
| `details.identity` | Observed checkpoint, tokenizer, device, dtype and cache configuration checks. |
| `run_metadata` | Caller provenance, observed server version, and Unix start/finish times in seconds. |
| `seconds` | Wall-clock seconds; request duration includes response capture. |

Raw distribution rows retain `max_probability_error_lower_bound` and
`max_probability_error_upper_bound`. Their `same_top_token` diagnostic compares
the first listed token IDs; it does not compare emitted tokens under tied scores.

`cache.same_execution` compares prepared branches and matched repeats with the
same execution geometry. Its retained bounds require exact equality.
`cache.changed_prefill` compares fresh, prefix-seeded and decode-created
histories under different prefill splits, with the existing nonzero limits.
These comparisons remain separate in the spec and report. Cache lengths,
boundaries and checkpoint offsets are token counts.
`details.retention_interval` is a token interval: zero disables periodic
decode snapshots; `null` records dense aligned snapshots.

Retention zero permits initial recomputation of decode-created history. It
still requires actual reuse after preparing checkpoints and exact branch
controls after sibling requests. Replayed generated history must reuse more
than the original prompt. Exact controls protect persisted state; they cannot
detect every initially wrong state.

`report.save(output)` delegates to `marin.execution.artifact.write_artifact`,
which writes `output/.artifact.json`. Reload with
`read_artifact(output, CheckReport)`. The record retains specification and limit
versions, provisional labels, measured values and raw observations. Version 1
JSON reports remain in their original storage; this API does not rewrite or
silently load them. Reference manifests remain at their existing schema version
and byte identities. The test-spec schema is independently versioned at 1.

Reference comparisons use saved top-64 and serving top-50 scores. Batch/cache
comparisons use serving top-50 lists. Decode compares emitted-token probability
error with recomputation and retains logprob error, in natural-log units (nats),
as a diagnostic. A pass does not establish full-vocabulary KL, RL trainer
agreement or task quality.

The [retained BF16 evidence](https://github.com/marin-community/marin-logbooks/blob/a858ad5f339e9d3a9d0550c12c614e7d3cbb82c0/logbooks/57-inference-perfmax/cache-retention-handoff.md)
includes zero/positive retention for both targets and 19 exact state comparisons
per run on vLLM `0ac9dcb157ee86ccb447bc717259159c41ad456a`. The API refactor reuses
that evidence for unchanged numerical checks. It does not qualify a newer
runtime or a changed serving configuration. Both policies remain provisional;
performance and runtime adoption require separate evidence. The existing
Snowball step42150 gate and bounds are unchanged.

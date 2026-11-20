# Check an inference endpoint

Use `marin.testing.inference.endpoint_checks.check_endpoint` to check a dedicated
`vllm serve` endpoint against saved reference scores. The caller owns deployment
and cleanup. Enable `VLLM_SERVER_DEV_MODE=1`, prefix caching, compilation and CUDA
graphs. The checker reads server settings, resets its cache and sends overlapping
requests to engine 0.

Use a Marin vLLM build that reports its source commit in `/version` and exposes
`prefix_cache_retention_interval` in `/server_info`.

The JSON bindings in
[`experiments/evaluation/inference_correctness/fixtures`](https://github.com/marin-community/marin/tree/main/experiments/evaluation/inference_correctness/fixtures)
pin the checkpoint, tokenizer, reference manifest and acceptance policy:

- `snowball-v0.1.1.json`: Snowball v0.1.1, BF16 on H100.
- `hero-sft-step400407.json`: Hero/Mochi SFT step400407, BF16 on GB200.

Supply the served model name and the full vLLM source commit from its runtime
manifest. A matching runtime identity is checked against `/version`; checkpoint,
tokenizer, dtype, hardware and cache settings are checked against `/server_info`
and checkpoint metadata.

## Python

```python
import asyncio
from pathlib import Path

from marin.testing.inference.endpoint_checks import check_endpoint
from marin.testing.inference.goldens import read_fixture

fixture = read_fixture(
    "experiments/evaluation/inference_correctness/fixtures/snowball-v0.1.1.json"
)
report = asyncio.run(
    check_endpoint(
        base_url="http://HOST:PORT/v1",
        model="SERVED_MODEL",
        fixture=fixture,
        runtime_source="FULL_VLLM_SOURCE_COMMIT",
        output=Path("correctness.json"),
    )
)
assert report["status"] == "passed", report["checks"]
```

The return value equals the retained JSON report. It contains the model and
runtime identities, manifest and policy hashes, per-check outcomes, numerical
statistics and coverage failures. Raw HTTP responses are retained in
`correctness-requests/`, including failed responses. A missing reference returns
`pending`; contradictory identity or scores fail. Only `passed` accepts a run.

## CLI and pytest

Both adapters call the same Python checker and return an unsuccessful exit
status unless all required checks pass. Run these commands from the repository
root; replace the endpoint, model and runtime commit:

```sh
uv run python -m marin.testing.inference.check_endpoint \
  --fixture experiments/evaluation/inference_correctness/fixtures/snowball-v0.1.1.json \
  --url http://HOST:PORT/v1 --model SERVED_MODEL \
  --runtime-source FULL_VLLM_SOURCE_COMMIT --output correctness.json

uv run pytest --confcutdir=tests/cluster/vllm \
  tests/cluster/vllm/test_inference_correctness.py -o addopts= \
  --endpoint-url http://HOST:PORT/v1 \
  --correctness-fixture experiments/evaluation/inference_correctness/fixtures/snowball-v0.1.1.json \
  --correctness-model SERVED_MODEL --runtime-source FULL_VLLM_SOURCE_COMMIT \
  --correctness-output correctness.json
```

`--confcutdir` keeps this endpoint-only test independent of the parent cluster
fixtures. `-o addopts=` clears the default marker filter and 600-second session
cap for this manually selected test. Neither adapter deploys a server or imports
Knotmeter or AgentX.

Verify saved references without an endpoint or GPUs:

```sh
uv run python -m marin.testing.inference.check_endpoint \
  --fixture experiments/evaluation/inference_correctness/fixtures/snowball-v0.1.1.json \
  --lookup
```

Lookup verifies artifact hashes, identities, exact token histories and score
coverage. Its statuses are `found`, `missing` and `mismatched`; only `found`
returns zero. Routine checks read the saved goldens and never regenerate them.

## Checking contract

The five required checks cover deployment identity, reference agreement, batch
dependence, cache reuse and decode versus recomputation. Missing required hits,
non-finite scores, incomplete responses and failed coverage block acceptance.
Later reference cases and cache boundaries still run after individual failures.

Cache probes use the endpoint's retention interval and attention/Mamba alignment.
Retention zero allows initial recomputation of decode-created history. Every mode
must still demonstrate actual reuse after explicitly preparing checkpoints, and
both branches must retain exact scores after a sibling request changes a token
inside the cached region. Direct periodic reuse is `not_required` only when the
retention mode and context cannot reach such a checkpoint. This does not waive
replayed decode reuse or the other checks. Exact branch controls check state
persistence; they cannot detect every initially wrong state.

Reference comparisons bound probability error over saved top-64 and serving
top-50 scores. Cache and batch comparisons use serving top-50 lists. Decode
checks measure the emitted token's probability error and retain its logprob
error without a separate acceptance limit. The policies remain provisional:
passing does not establish full-vocabulary KL, task quality or RL trainer ratios.
The existing step42150 runtime-adoption tests retain their own coverage and bounds.

The [source handoff](https://github.com/marin-community/marin-logbooks/blob/a858ad5f339e9d3a9d0550c12c614e7d3cbb82c0/logbooks/57-inference-perfmax/handoff.md)
and [retention evidence](https://github.com/marin-community/marin-logbooks/blob/a858ad5f339e9d3a9d0550c12c614e7d3cbb82c0/logbooks/57-inference-perfmax/cache-retention-handoff.md)
record BF16 checks for both bindings at zero and positive retention, including
19 exact state comparisons per run. Those runs used vLLM
`0ac9dcb157ee86ccb447bc717259159c41ad456a`. They validate the preserved checking
behavior on that runtime; they do not qualify a newer runtime selected by main.
Performance and runtime adoption require their separate evidence.

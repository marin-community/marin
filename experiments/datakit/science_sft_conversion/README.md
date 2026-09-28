# Science-forward text-to-chat conversion

`sources.json` pins the 17 non-chat source artifacts used by the science-forward
SFT mix. They contain 862 Parquet shards in the CoreWeave copy of the source
pool. The Nemotron Math Proofs, Nemotron Science v2, and TextbookReasoning
components are already structured chat and are excluded here.

The worker reads each source row and splits long text at nearby paragraph or
line boundaries, retaining every source character across chunks. Each chunk
gets one of six answer formats chosen from a stable source ID hash. MiniMax
generates a user request, an assistant `reasoning_content` span, and a final
answer. The worker validates the three fields, the requested final format, and
Datakit's Harmony message structure before writing `CHAT_SCHEMA` Parquet.
For prose, code, and textbook passages, the worker appends the source chunk to
the user turn so the assistant's answer has its evidence. For the Nemotron math
textbooks and Swallow math QA sources, MiniMax extracts a standalone question
without exposing its worked solution. A rejected completion is sent back to
MiniMax with the validation error for repair. If four standalone attempts fail,
the worker requests a source-grounded task and appends the chunk to the user
turn. The v1 probe omitted context in some user turns. A later five-row-per-source
v2 probe found repeatable standalone-question failures and stopped before
writing its result. Neither earlier version has production Parquet output.
Output files live under
`s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted/2026.09.27-v3/outputs/main/`.
`experiments.datasets.science_forward_converted.science_forward_converted_dataset()`
registers this Parquet directory as the Datakit source
`science-forward/minimax-m3-formatted-2026.09.27-v3`. The handle references
the existing output and does not start conversion.
Each file covers one 1,024-row source batch; completed batches are skipped on
restart. A failed batch remains unwritten and fails the Iris task after four
attempts at each of the six answer formats. The worker starts with the format
chosen by the source ID hash, then tries the remaining formats in a fixed order
if validation keeps failing. Its logged format counts reflect the format
actually written. The source artifact stays unchanged.

The production scheduler shuffles source batches reproducibly with seed
20260927, then rotates across the 17 sources one batch at a time. Its first 17
batches cover every source. Use 17 client tasks to give each source a worker
in the first wave. Larger sources continue after smaller sources are exhausted;
the scheduler still covers every original batch. Restarting with a different
task count reassigns pending batches and preserves completed output files.

The pinned source pool contains 105,582,071 rows across 6,350 row groups.
The worker checks these row counts before issuing requests. Every nonempty
row requires at least one MiniMax completion; long rows require more than one.

The conversion uses the shared `/benfeuer/minimax-m3-science-sft` Iris endpoint
on `cw-rno2a`. Serve MiniMax M3 from the science SFT worktree. Three H100x8
workers share one brokered endpoint at Iris's interactive priority. The
65,536-token context exceeds the worker's 8,000-character source chunk size.
CoreWeave's S3 cache needs lower RunAI reader concurrency and a longer read
window while all workers load the 428B-parameter checkpoint:

```bash
uv run marin-serve iris MiniMaxAI/MiniMax-M3-MXFP8 --cluster cw-rno2a \
  --gpu H100x8 --instances 3 --name minimax-m3-science-sft-20260927 \
  --endpoint-name /benfeuer/minimax-m3-science-sft \
  --max-model-len 65536 --max-num-batched-tokens 8192 \
  --cpu 64 --memory 1024g --disk 800g --timeout-hours 168 \
  --proxy-timeout 1800 --vllm-version 0.30.0 \
  --streamer-concurrency 2 --streamer-s3-request-timeout-ms 30000 \
  --vllm-arg=--max-num-seqs=8 --vllm-arg=--gpu-memory-utilization=0.97 \
  --vllm-arg=--block-size=128 --vllm-arg=--reasoning-parser=minimax_m3 \
  --vllm-arg=--kv-cache-dtype=fp8 --vllm-arg=--enable-expert-parallel --no-wait
```

After the endpoint answers a structured completion, start one small batch
before the full run:

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name science-sft-conversion-smoke-20260927 \
  --cpu 8 --memory 32GB --disk 20GB --extra cpu --no-wait \
  -- python -m experiments.datakit.science_sft_conversion.conversion \
    --max-items 1 --max-batches 1 --concurrent-batches 1
```

After checking the first generated records and model response, submit the
full resumable workload:

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name science-sft-conversion-v3-stratified-stage-20260927 --replicas 17 --max-retries 8 \
  --cpu 8 --memory 32GB --disk 20GB --extra cpu --no-wait \
  -- python -m experiments.datakit.science_sft_conversion.conversion --concurrent-batches 2
```

To inspect the conversion across all 17 sources while the full run proceeds,
write sampled source chunks and their validated conversations to a separate
JSON file. The first sample is the first row used by the original probe; the
remaining samples use reproducible random shards, row groups, and chunks:

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name science-sft-conversion-probe-v3-r3-20260927 --cpu 4 --memory 16GB \
  --disk 10GB --extra cpu --no-wait \
  -- python -m experiments.datakit.science_sft_conversion.probe \
    --samples-per-source 5 --seed 20260927 \
    --output-path s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted/2026.09.27-v3/audit/probe-stratified-85-r3.json
```

The conversion is complete only when every expected source batch has a
validated output file. Audit batch coverage, Parquet schema, and the minimum
record count before registering the converted chat source in Datakit or using
it for SFT. Run the audit from the science SFT worktree with CoreWeave S3 access:

```bash
uv run python -m experiments.datakit.science_sft_conversion.audit
```

The audit prints aggregate counts and exits nonzero if a batch is missing,
short, has the wrong schema, or if an unrecognized Parquet file appears in the
output directory. Inspect generated conversations from each source as well;
the audit checks structural completeness, not answer quality.

To compare server batching configurations, use the identical probe JSON file
and an endpoint with no other traffic. The benchmark issues one completion per sampled chunk,
records token usage and request timings, and reports first-attempt validation
failures without repair requests. Rejected completions count toward throughput
and do not stop the benchmark. HTTP failures fail the benchmark. Compare server
throughput logs as well: the reported completion tokens per second includes
initial filling and final draining of the request queue.

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name science-sft-throughput-c16-20260927 --cpu 4 --memory 16GB \
  --disk 10GB --extra cpu --no-wait \
  -- python -m experiments.datakit.science_sft_conversion.benchmark \
    --endpoint /benfeuer/minimax-m3-science-sft-c16 --concurrency 16 \
    --input-path s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted/2026.09.27-v3/audit/probe-stratified-85-r3.json \
    --output-path s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted/2026.09.27-v3/audit/throughput-c16.json
```

For a larger pool, the experiment's serving entrypoint configures both vLLM's
running-sequence limit and the broker's per-worker request limit. Its proxy
accepts up to 2,048 pending requests. Submit it as an interactive CPU job; it
creates eight-GPU H100 worker jobs at the same priority. Use a separate endpoint
while the old pool is serving, then restart the resumable clients against the
new endpoint once its model API answers. A response proves at least one replica
is ready; inspect the worker jobs to count ready replicas before raising client
concurrency to the full-pool target.

The serving entrypoint can retain the existing regional model cache at a
CoreWeave prefix outside `tmp/ttl=*`. The source is the pinned model snapshot
resolved under `tmp/ttl=14d/quick-serve-models`; a cache miss downloads it from
Hugging Face. Use this option while the regional cache is still present.
It copies objects within the same bucket, checks
file sizes, and writes the completion marker after every copy succeeds. This
keeps replacement workers from depending on an expired temporary cache.
The 120-day service deadline closes the serving session and its workers when
it expires. It is separate from the temporary cache retention.

The serving entrypoint pins the Hugging Face MiniMax model checkpoint revision
`c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`. It defaults to eight-way tensor
parallelism and one data-parallel group. To test two data-parallel groups sharing
eight-way expert parallelism, pass `--tensor-parallel-size 4 --data-parallel-size 2`.
The product of these two degrees must equal eight GPUs per worker. Compare the
same sampled workload against the default topology, measuring output tokens per
second and validation rejection counts, before changing the production topology.

The commands below request 23 replicas at a maximum of 128 running sequences
per replica, with the broker admitting up to 128 in-flight requests per worker.
Seventeen clients at concurrency 112 offer 1,904 simultaneous
requests, about 83 per replica when all 23 are serving. This stays below the
proxy's 2,048-request budget. Adjust client concurrency
using measured completion throughput and queue latency. Stop the old conversion
job before submitting its replacement against the same output directory.
Each Iris client task processes two source batches concurrently through one HTTP pool
and one shared request semaphore. A slow tail in one batch therefore permits
the other batch to use available request slots. Output paths are determined by
source, input shard, row group, and batch index. Previously committed files are
skipped; unfinished batches are recomputed on restart and committed atomically
when complete. A failed batch cancels the other batch's outstanding requests
and fails the task. Source reads and output writes run outside the event loop.

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name minimax-m3-science-sft-scaled-r2-20260927 --cpu 4 --memory 16GB \
  --disk 20GB --extra cpu --no-wait \
  -- python -m experiments.datakit.science_sft_conversion.serve \
    --endpoint /benfeuer/minimax-m3-science-sft-scaled --instances 23 \
    --max-sequences 128 --timeout-hours 2880 --startup-timeout-seconds 3600 \
    --model-cache-path s3://marin-us-east-02a/marin/users/benfeuer/science-sft-models/MiniMaxAI_MiniMax-M3-MXFP8_c5454eb03678d8710e54a4e0fc681b9f3b4a3dba

uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name science-sft-conversion-v3-pipelined-20260928 --replicas 17 --max-retries 8 \
  --cpu 8 --memory 32GB --disk 20GB --extra cpu --no-wait \
  -- python -m experiments.datakit.science_sft_conversion.conversion \
    --endpoint /benfeuer/minimax-m3-science-sft-scaled --concurrency 112 --concurrent-batches 2
```

## Full-conversion handoff

The handoff waits for the named conversion job to succeed, then audits every
expected output batch for coverage, schema, and minimum record count. It writes
`audit/final-coverage.json`, stops the dedicated conversion serving pool, builds
the packed token store with loss on assistant reasoning and final answers
only, checks that its conversation count matches the
audited output, and launches one epoch of [Step38 SFT](../../grug/science_sft/README.md)
from `open-athena/Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38`. Overlength conversations
fail preparation. These checks verify structural completeness; they do not
verify the factual correctness of generated answers.

Print the handoff plan with the following command. Add `--run` to execute it
as an Iris CPU job. Pass the key explicitly with Iris
`-e WANDB_API_KEY "$WANDB_API_KEY"`; `--run` does not copy shell variables
into the task environment. Submit it only after
the named producer exists. Use no automatic retries: inspect a failure before
resubmitting, especially if the SFT child job was already launched.

```bash
uv run python -m experiments.datakit.science_sft_conversion.finish \
  --producer-job /benfeuer/science-sft-conversion-v3-pipelined-20260928 \
  --serving-job /benfeuer/minimax-m3-science-sft-scaled-r2-20260927 \
  --store-path s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted-store/2026.09.27-v3 \
  --num-shards 1024 --max-workers 32 --timeout-hours 2880 --version 2026.09.27-v3
```

The `codex/science-sft-conversion` worktree includes the concurrency settings from
[PR #8891](https://github.com/marin-community/marin/pull/8891): the proxy sizes
its thread limiter to its pending-request budget, the dashboard permits up to
4,096 upstream connections, and each worker sizes its HTTP pool to its in-flight
limit. Conversion and benchmark HTTP pools match their client concurrency.
The 64-request admission and overload regression test in
`tests/evals/test_inference_proxy.py` verifies the corrected proxy. The larger
pool command above creates the `scaled-r2` serving job; its endpoint must match
the producer's `--endpoint`. These settings must ship with the jobs; raising
vLLM's sequence limit alone
does not remove the shared proxy's default 40-request limit.

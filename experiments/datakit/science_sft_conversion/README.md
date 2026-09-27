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
source-grounded attempts, or four attempts for sources that start in the
source-grounded mode. The source artifact stays unchanged.

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
    --max-items 1 --max-batches 1
```

After checking the first generated records and model response, submit the
full resumable workload:

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name science-sft-conversion-v3-20260927 --replicas 16 --max-retries 8 \
  --cpu 8 --memory 32GB --disk 20GB --extra cpu --no-wait \
  -- python -m experiments.datakit.science_sft_conversion.conversion
```

To inspect the conversion across all 17 sources while the full run proceeds,
write sampled source chunks and their validated conversations to a separate
JSON file. The first sample is the first row used by the original probe; the
remaining samples use reproducible random shards, row groups, and chunks:

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name science-sft-conversion-probe-v3-r2-20260927 --cpu 4 --memory 16GB \
  --disk 10GB --extra cpu --no-wait \
  -- python -m experiments.datakit.science_sft_conversion.probe \
    --samples-per-source 5 --seed 20260927 \
    --output-path s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted/2026.09.27-v3/audit/probe-stratified-85-r2.json
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

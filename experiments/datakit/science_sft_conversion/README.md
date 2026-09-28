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
For the JSON answer format, the response schema requires a nested `answer`
object with an `answer` string and `evidence` and `caveats` arrays of strings.
The worker supplies this schema to vLLM for constrained generation, then
serializes the answer object into the Harmony final message. The constraints
cover JSON syntax and string escaping, including backslashes and quotes.
Other formats use an answer string in the response schema.
Grounded tasks extract reported facts and reuse supplied worked steps. Supplied
annotations are reported as reference information; the generated reasoning must not
invent how those labels were derived. The request repeats these requirements
after the quoted passage to distinguish the conversion task from instructions
embedded in the source.
Assistant text containing phrases such as `conversion task`, `the reasoning_content`,
or `the answer field` is rejected and sent back for repair. These phrases describe
the text-to-chat preparation instead of the user's task. Scientific unit conversions remain allowed.
For damaged mathematical extraction, the request requires quoting the supplied
text and marking ambiguity instead of restoring missing notation or equations.
For source-grounded tasks that summarize the supplied passage, the validator rejects added square-root
notation when the source contains no radical notation or explicit square-root wording.
Standalone worked exercises can derive new radicals.
The check permits prose explaining that square-root notation is missing; it targets
added half powers such as `^0.5` or `**(1/2)`, `√`, `\sqrt`, and `sqrt` notation.
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

The 2026-09-28 teacher-exercise update changes newly generated BioCollection
instruction and Swallow math textbook batches. These batches use teacher
exercises. BioCollection user turns pose the original challenge with its inputs
and omit the supplied reference answer. A validation check rejects altered tagged
DNA, RNA, protein, peptide, and SMILES strings in the learner question. The teacher uses that reference to check
the reasoning and answer. When a label requires unavailable assay measurements
or structural coordinates, the reasoning states that limitation and identifies
the result as a reference annotation; it must not invent a derivation.
Swallow textbook user turns pose problems exercising the supplied theories and
starting formulas. They reuse source examples when possible, or provide clearly
hypothetical givens. Reasoning shows intermediate steps and checks the solution.
Worked answers stay out of the user turn. These two sources retry teacher
exercises in another response format after four rejected attempts; they do not
fall back to passage summaries or verbatim extraction. Exhausting all formats
fails the batch. Previously committed output batches remain unchanged. The
producer cutover manifest in the experiment's training-data-sources artifacts
lists the batches retained from the earlier extraction policy.

Each file covers one 1,024-row source batch; completed batches are skipped on
restart. The worker gives the assigned format four total attempts, including the
initial request. A standalone task rejected on all four attempts switches to a
source-grounded task with the complete source chunk attached to the user turn.
After four source-grounded responses fail semantic or format validation, MiniMax
selects source paragraph indices and generates a short reasoning trace. The
formatter copies the selected paragraphs into the originally assigned answer
format, preserving the supplied notation in the answer. If at most eight
eligible paragraphs are available, the client includes them all even when the
model repeats an index. The complete original
chunk remains in the user turn. Generated reasoning follows the same validation
as other responses; neither source claims nor reasoning are independently fact
checked. This verbatim extraction fallback has four total attempts. Exhausting
those attempts leaves the batch unwritten and fails the Iris task. Repeated HTTP
failures instead rotate through the remaining answer formats before failing the
batch. Logged format counts reflect the format actually written. The source
artifact stays unchanged.

The production scheduler shuffles source batches reproducibly with seed
20260927, then rotates across the 17 sources one batch at a time. Its first 17
batches cover every source. Use 17 client tasks to give each source a worker
in the first wave. Larger sources continue after smaller sources are exhausted;
the scheduler still covers every original batch. Restarting with a different
task count reassigns pending batches and preserves completed output files.

The pinned source pool contains 105,582,071 rows across 6,350 row groups.
The worker checks these row counts before issuing requests. Every nonempty
row requires at least one MiniMax completion; long rows require more than one.

The conversion uses the shared `/benfeuer/minimax-m3-science-sft-scaled` Iris
endpoint on `cw-rno2a`. Launch the serving pool with the scaled configuration
below. The 65,536-token context exceeds the worker's 8,000-character source
chunk size. The conversion CLI requires an explicit endpoint.

After the endpoint answers a structured completion, start one small batch
before the full run:

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name science-sft-conversion-smoke-20260927 \
  --cpu 8 --memory 32GB --disk 20GB --extra cpu --no-wait \
  -- python -m experiments.datakit.science_sft_conversion.conversion \
    --max-items 1 --max-batches 1 --concurrent-batches 1 \
    --endpoint /benfeuer/minimax-m3-science-sft-scaled --concurrency 112
```

After checking the first generated records and model response, submit the
full resumable workload:

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name science-sft-conversion-v3-stratified-stage-20260927 --replicas 17 --max-retries 8 \
  --cpu 8 --memory 32GB --disk 20GB --extra cpu --no-wait \
  -- python -m experiments.datakit.science_sft_conversion.conversion \
    --concurrent-batches 2 --endpoint /benfeuer/minimax-m3-science-sft-scaled --concurrency 112
```

To inspect the conversion across all 17 sources while the full run proceeds,
write sampled source chunks and their validated conversations to a separate
JSON file. The first sample is the first chunk of the first row in the first
shard's first row group. The remaining samples use reproducible random shards,
row groups, rows, and chunks.
Rows are sampled from the entire selected row group. Shards and row groups are
sampled uniformly within each source; this is not a corpus-wide row-weighted sample:

```bash
uv run iris --cluster=cw-rno2a job run --priority interactive --enable-extra-resources \
  --job-name science-sft-conversion-probe-v3-r3-20260927 --cpu 4 --memory 16GB \
  --disk 10GB --extra cpu --no-wait \
  -- python -m experiments.datakit.science_sft_conversion.probe \
    --samples-per-source 5 --seed 20260927 \
    --endpoint /benfeuer/minimax-m3-science-sft-scaled --concurrency 16 \
    --output-path s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted/2026.09.27-v3/audit/probe-stratified-85-r3.json
```

The conversion is complete only when every expected source batch has a
validated output file. Audit batch coverage, Parquet schema, and exact
source-row/chunk identities before using the converted chat source for SFT.
The audit reads source text and output chunk IDs. Run it on RNO2A to keep
these reads in the storage region:

```bash
uv run iris --cluster=cw-rno2a job run \
    --priority interactive --enable-extra-resources \
    --job-name science-sft-conversion-audit --cpu 8 --memory 32GB --disk 20GB \
    --extra cpu --no-wait -- \
    python -m experiments.datakit.science_sft_conversion.audit --workers 32
```

The audit prints aggregate counts and exits nonzero if a batch is missing,
has fewer records than input rows, has the wrong schema, contains missing, duplicated, or unexpected chunk
IDs, or if an unrecognized Parquet file appears in the output directory.
Generated answers still require quality review; the audit does not assess
their factual correctness.

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
expected output batch for coverage, schema, and exact source-row/chunk identities. It writes
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

# Debugging log for focus crawl WARC downloads

Extract focus-crawl HTML with bounded Common Crawl range requests, then run jusText from worker-local files.

## Initial status

The first production run streamed CloudFront responses through WARC parsing. Workers held responses open during CPU-bound jusText calls and failed with incomplete reads. Full WARC downloads also transferred the crawl's PDF payloads even though the pipeline only extracts HTML.

## Hypothesis 1

Downloading and parsing were serialized on the same response stream. Completing a bounded download before extraction closes the network response before CPU-bound parsing begins.

## Changes to make

Download each WARC into worker-local storage, resume interrupted transfers with HTTP ranges, then parse the completed local file.

## Results

The interrupted-download integration test failed against the streaming implementation and passed with local staging and HTTP range resume.

## Hypothesis 2

The Common Crawl CDX sidecar can identify the byte ranges containing successful HTML responses. Fetching those ranges avoids most PDF transfer while preserving every indexed HTML record.

## Changes to make

Read each WARC's CDX sidecar, coalesce adjacent HTML record ranges with a 1 MiB gap, and parse each downloaded range as a WARC stream. Run 576 workers with one CPU each on the Genoa pool.

## Results

A representative WARC contained 9,352 indexed HTML responses. Range extraction reconstructed all 9,352 and transferred 294.3 MB with a 1 MiB coalescing gap instead of the 1.54 GB WARC. The focused pytest and changed-file pre-commit checks pass.

## Hypothesis 3

Decoded HTML can contain XML 1.0 control characters that lxml rejects inside the pinned jusText fork.

## Changes to make

Remove XML-forbidden control characters before jusText and count each removal. Increment the StepSpec schema version so the repaired run cannot reuse partial output from the failed run.

## Results

Run `focus-crawl-justext-r7` uses output hash `46577c64`. It reached 559 in-flight shards on 560 registered workers in five minutes. A thread profile showed worker 0 inside the jusText classifier. One worker registration failure requeued one shard; no range-download interruption or user-code exception has appeared. Completed-shard throughput is not measured yet.

## Hypothesis 4

One Iris pod per 1-CPU task cannot fill the Genoas because Kubernetes reaches its per-node pod limit at about 100 task pods, before exhausting the 192 vCPUs.

## Changes to make

Run 72 worker actors with 8 CPUs and 64 GB each. Give Zephyr 1 CPU and 8 GB per map task and use subprocess isolation, producing eight concurrent map tasks per actor and 576 total task slots. Submit the root job at interactive priority so its control path is not preempted by ordinary batch work.

## Results

Run `focus-crawl-justext-r14` placed 17–19 worker actors on each of the four Genoas and reached 576 in-flight WARC shards with all 72 workers alive. The previous one-pod-per-task layout plateaued at 394 active workers despite spare CPU and memory.

## Hypothesis 5

The bundled jusText random forest's saved `n_jobs=8` setting oversubscribes the 1-CPU map tasks and emits a scikit-learn warning on every prediction.

## Changes to make

Set the pinned model's estimator to `n_jobs=1` after loading it and extend Zephyr's heartbeat timeout to 15 minutes for CPU-heavy WARC shards.

## Results

The 1-worker benchmark was 1.21 times faster with `n_jobs=1` on a 200-paragraph page. In r14 the warning flood disappeared. Over a longer sample, live counters increased from 560,073 to 782,664 processed documents in 239 seconds, or roughly 930 documents per second. At that rate, processing 40.6 million indexed HTML records takes about 12 hours before shard-tail overhead.

## Hypothesis 6

The run must exclude Genoa nodes under ephemeral-storage pressure and leave enough CPU and memory for resident workloads and control pods.

## Changes to make

Remove the `iris.pool=cpu-genoa` label from the disk-pressured node and run 66 worker actors with eight map slots each on the remaining three Genoas.

## Results

CoreWeave evicted seven r14 worker pods from `g5bea54` after the node crossed its ephemeral-storage threshold. The node also hosted both control tasks, so r14 was stopped. A 72-actor r15 launch saturated the three healthy nodes at 66 running actors, with six permanently pending because node requests were already at 94–98% of memory and 97–98% of CPU. The stable shape is therefore 66 actors and 528 concurrent map tasks.

Run `focus-crawl-justext-r17` placed 23, 22, and 21 worker actors on the three healthy Genoas. All control tasks also landed on healthy nodes. It remained at 66/66 live workers with no failed pods, shard retries, heartbeat failures, download exceptions, or classifier warning flood through the 15-minute failure window. Representative actors ran all eight subprocesses at approximately 100% CPU, confirming that parsing rather than serialized downloading was the bottleneck. Live counters increased from 118,644 to 716,327 documents in 867 seconds, or roughly 689 documents per second. That projects to about 16.4 hours for 40.6 million documents before shard-tail overhead.

## Hypothesis 7

Common Crawl can return transient 403 responses during a crawl-wide access interruption. The retry policy must treat 403 as transient and wait long enough for access to recover.

## Changes to make

Retry 403 responses up to 10 times with exponential backoff and up to 10 seconds of jitter. Apply the policy to both CDX and WARC range requests through the shared HTTP session.

## Results

At 08:28 UTC, r17 began receiving 403 responses across unrelated CDX and WARC objects. The existing policy retried WARC ranges immediately and did not retry CDX requests. By 08:35 UTC, 4,086 shards had failed twice, one shard had failed three times, and only 43 of 4,573 shards had completed. The pipeline failed at 08:36 UTC. The same URLs returned HTTP 200 later that morning, confirming that the interruption was transient. The integration test now returns one 403 from both the CDX and WARC endpoints before succeeding; it failed before the retry-policy change and passes afterward.

## Hypothesis 8

XML-invalid numeric character references can survive source-text sanitization because lxml decodes them only after parsing. XenonMolecule's table rewrite then fails when it assigns the decoded control character to a replacement element.

## Changes to make

Remove numeric character references that decode to XML-invalid code points before calling jusText. Preserve valid references and count removals with the existing XML-invalid-character counter.

## Results

Run `focus-crawl-justext-r18` processed 1.9 million documents before shard 195 failed in `rewrite_data_tables` with an XML compatibility `ValueError`. A three-row data table containing `&#1;` reproduces the live failure. The focused test fails before sanitizing numeric references and passes afterward.

## Hypothesis 9

The crawl's Parquet manifest needs the same transient-403 retry policy as its CDX sidecars and WARC ranges.

## Changes to make

Stage the manifest through the shared retrying HTTP session before reading it with PyArrow.

## Results

Run `focus-crawl-justext-r19` failed immediately when the unretried fsspec manifest request received HTTP 403. The integration test now returns one 403 from the manifest endpoint before succeeding. Run `focus-crawl-justext-r20` reused all 45 committed output shards and reached 520 in-flight tasks on 65 live worker actors.

## Hypothesis 10

The fork's calendar detector treats non-decimal compatibility digits as integers because Python's `str.isdigit()` includes characters such as `①`, while `int()` rejects them.

## Changes to make

Normalize only compatibility digits with a Unicode digit value to their decimal representation before they reach the fork's calendar detector. Handle both numeric HTML references and decoded characters.

## Results

In r20, shards 59, 122, and 536 failed with `ValueError: invalid literal for int() with base 10: '①'`. A ragged data table containing `&#9312;` reproduces the failure. The focused test fails before compatibility-digit normalization and passes afterward. Run `focus-crawl-justext-r21` resumes from the 70 committed output shards.

## Hypothesis 11

Malformed HTML can trigger `AssertionError` in `lxml_html_clean.Cleaner` while jusText drops an element whose parent was already removed. Retrying the WARC does not change the input and will reproduce the failure.

## Changes to make

Skip only records that raise this cleaner assertion, count them, and cover the malformed DOM shape with the focused WARC integration test. Keep the StepSpec output identity unchanged so the replacement run can reuse committed shards, whose successful records are unaffected by this change.

## Results

At 16:09 PDT, r21 had committed 449 of 4,573 shards with all 66 workers alive. Shards 894 and 979 failed in `lxml.html.HtmlElement.drop_tag` with `AssertionError` and were requeued behind the remaining work. A root `<button>` containing a `<form>` reproduces the cleaner failure. The remaining observed first-attempt retries were transient Common Crawl access failures; no shard had reached a second retry. The user approved stopping r21 at 16:21 PDT after it had committed 451 shards.

The focused WARC integration test passes with the malformed record skipped and counted while the following valid record is written.

Run `focus-crawl-justext-r22` resumed from all 474 shards committed before r21 stopped. It reached 66/66 workers and 528 in-flight shards. Live records under `zitogiuseppe.com` reproduced the cleaner assertion and were skipped without failing their shard, confirming the recovery path in production.

## Final result

Run `focus-crawl-justext-r22` completed all 4,573 shards. The normalized artifact contains 36,327,068 documents in 4,573 Parquet files totaling 89,446,357,030 bytes at `gs://marin-us-central1/data/datakit/normalized/common_crawl_focus_2026_22_ed4b8bc9`. Full tokenization with `marin-community/marin-tokenizer` measured 49,702,569,456 tokens; the temporary tokenized cache was deleted after reading `.stats.json`. [PR #7382](https://github.com/marin-community/marin/pull/7382) registers the normalized artifact in the Datakit source catalog.

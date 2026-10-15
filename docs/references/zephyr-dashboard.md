# Zephyr dashboard

Open the coordinator's dashboard from its Iris task to inspect pipelines,
stage progress, counters, workers, and throughput history. Select a pipeline
to inspect its current-stage shard grid.

## Mapper and reducer progress

Each running tile fills with the fraction of its current input consumed.
Select a tile to see row counts, elapsed time, worker
heartbeat age, and a link to the worker task. Use the shard-state selector to
show only running shards when looking for the remaining work in a stage.
Progress remains separate for concurrent shards on the same worker.
When reader record counts are available, the inspector also shows their
average rate over the shard's elapsed time, including earlier input files.

Input totals come from metadata already used by readers:

| Input | Progress measurement |
| --- | --- |
| Parquet | Selected rows from footer metadata, limited to the shard's row range and row groups retained by predicate pruning. Advances after each row group is consumed. |
| Unfiltered Vortex | Rows in the selected range. Advances after each read batch is consumed. |
| Shuffle reducer | Rows routed to the target partition, summed from mapper metadata files containing per-partition row counts. Advances after each merged batch is consumed. |
| JSONL, filtered Vortex, custom iterators | Unknown total; striped tiles. Reader record counts, when available, count records yielded after filtering. |

Parquet counts input rows before row-level filtering, so a selective filter
can finish without producing as many records as the progress total. A single
large row group can leave the bar unchanged while its rows are being mapped.
Shuffle progress can remain at zero during metadata loading or merge setup.

The fraction measures input consumption. A task at 100% may still be mapping
buffered rows or flushing output. A shard that reads several files resets its
input counts for each file. The dashboard does not infer a task ETA from this
fraction.

Workers send progress on their five-second heartbeat. For shards running in
subprocesses, workers must first read periodic counter snapshots from the
subprocess. This and the dashboard refresh interval can add delay.
The inspector marks worker heartbeats older
than 15 seconds as potentially stale; that alone does not establish that the
mapper is stuck. Progress from earlier retries and stages is discarded.
Once the caller releases a finished execution, its live shard details disappear.
Its overview and throughput history remain available.

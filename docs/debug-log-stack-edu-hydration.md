# Debugging log for Stack-Edu hydration

Enable stable Stack-Edu tokenization for Dolma 3 Pool by converting the metadata-only HuggingFace export into text-bearing JSONL shards before tokenization.

## Initial status

`stack_edu/*` tokenization failed after the raw path bug was fixed because the downloaded parquet files only contained Software Heritage blob ids and metadata, not a `text` column. The tokenizer then crashed on missing `text`.

## Hydration stage is missing

The failure is architectural rather than a tokenizer bug: Stack-Edu requires a content-hydration stage from Software Heritage blob storage before the existing tokenizer can run.

## Changes to make

- Add a resumable Stack-Edu hydration transform under `lib/marin/src/marin/transform/stack_edu/hydrate.py`
- Wire `experiments/pretraining_datasets/dolma3_pool.py` so `stack_edu/*` partitions depend on hydrated JSONL instead of raw parquet
- Update `experiments/domain_phase_mix/swarm_run.py` to include hydration steps explicitly, support exact partition names like `stack_edu/SQL`, and bound executor concurrency for Stack-Edu runs

## Future Work

- [ ] Measure skipped-blob rates across all 15 languages once a full run completes
- [ ] Decide whether Stack-Edu hydration metrics should be promoted into a permanent dataset audit script
- [ ] Consider whether the hydrated Stack-Edu shards should be reused outside Dolma 3 Pool

## Results

Implemented a real hydration stage that fetches `softwareheritage/content/{blob_id}` objects, decompresses them, decodes source text, and writes JSONL shards under `documents/stack_edu/<lang>/train/`. Dolma 3 Stack-Edu tokenization is now wired to those hydrated shards instead of hard-failing on the metadata parquet export.

## Cluster packaging drops generated Iris protobufs

The local hydration code worked, but Ray jobs still failed before any Stack-Edu work started because `iris.rpc.*_pb2.py` files are gitignored and the default `working_dir` upload path respects `.gitignore`.

## Changes to make

- Update `lib/marin/src/marin/run/ray_run.py` to upload `working_dir` with `.gitignore` disabled
- Expand the explicit upload excludes so the bundle does not accidentally pick up large local junk
- Verify with a minimal cluster job that imports `iris.rpc.time_pb2`

## Results

`ray_run.py` now opts into explicit excludes and successfully uploads the generated Iris protobuf files. A cluster smoke job importing `iris.rpc.time_pb2` succeeded under package `gcs://_ray_pkg_77feb87795ebe326.zip`, confirming the packaging issue is fixed.

## Hydration step callable is wrapped incorrectly

Once the protobuf packaging issue was fixed, the Stack-Edu SQL job advanced to the hydration step and then failed inside Marin executor because `hydrate_stack_edu` was registered as an `ExecutorStep` target while still decorated with `@draccus.wrap()`. Executor passes a resolved config object directly; the Draccus wrapper expects CLI-style argument parsing.

## Changes to make

- Keep `hydrate_stack_edu` as a plain callable for `ExecutorStep`
- Apply `draccus.wrap(...)` only under `if __name__ == "__main__":`
- Re-run a local end-to-end hydration smoke test through `hydrate_stack_edu(config)`
- Re-submit the Stack-Edu SQL cluster job

## Results

The hydration function now works both as a direct Python callable and as a CLI entrypoint. A local 10-row smoke test wrote hydrated JSONL plus metrics successfully. On cluster, `ray-run-calvinxu-stack-edu-sql-20260314-214950` progressed past startup and entered live Zephyr hydration, preparing 126 hydration tasks for `stack_edu/SQL`.

## Full Stack-Edu parallelism is too conservative

The SQL canary proved the end-to-end codepath works, but full Stack-Edu was still artificially limited by `max_concurrent=2` even though the cluster had ample free CPU capacity. The desired rollout is to keep per-language hydration at 32 workers and scale across the remaining 14 languages without overlapping the live SQL canary.

## Changes to make

- Lower per-language Stack-Edu hydration to 32 workers in `experiments/pretraining_datasets/dolma3_pool.py`
- Raise Stack-Edu executor concurrency to 14 in `experiments/domain_phase_mix/swarm_run.py`
- Add a `stack_edu_rest` category that excludes `stack_edu/SQL`
- Submit a cluster run for the remaining 14 languages and verify it reaches live hydration

## Results

The config now targets the requested `max_workers=32` and `max_concurrent=14`, with `stack_edu_rest` available as a clean non-overlapping submission target for the remaining languages.

## Higher Stack-Edu hydration parallelism is worth trying

The SQL canary succeeded, cluster CPU headroom remained high, and the dominant rest-language runtimes are driven by the number of hydration task waves per language. Increasing per-language hydration workers should reduce those waves materially without exhausting current cluster capacity.

## Changes to make

- Raise the Dolma 3 Stack-Edu hydration override from 32 to 96 workers per language
- Leave the live 32-worker rest job alone; apply the new setting on the next submission

## Results

The Dolma 3 Stack-Edu path now targets `max_workers=96` for future jobs while preserving `max_concurrent=14`.

## Use tonight's idle cluster headroom

The 96-worker run is healthy, but the observed bottleneck remains Java hydration. With the cluster mostly idle overnight, the best way to pull the wall clock into the next several hours is to resubmit with a much higher per-language worker cap while reusing the already-written hydration and tokenization outputs.

## Changes to make

- Raise the Dolma 3 Stack-Edu hydration override from 96 to 200 workers per language
- Stop the in-flight 96-worker rest run
- Re-submit the same `stack_edu_rest` job so completed shards are reused and the large remaining languages launch larger Zephyr worker groups

## Results

The Dolma 3 Stack-Edu path now targets `max_workers=200` for future jobs while preserving `max_concurrent=14`.

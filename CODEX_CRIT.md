# Critique of Claude's decontaminated val-set build

## Findings

1. **Current build failed; Claude's latest status is stale.**

   `/ahmed/decon-val-build-all` is now `JOB_STATE_FAILED` with:

   ```text
   AssertionError: j050: filtered 13568 docs != 13577
   ```

   The filter stage did produce 231 parquet shards per cutoff, but every cutoff is short by exactly 9 docs:

   | cutoff | expected | filtered |
   |---|---:|---:|
   | j050 | 13,577 | 13,568 |
   | j075 | 27,742 | 27,733 |
   | j090 | 33,161 | 33,152 |

   No manifests were written, and tokenization had not started.

2. **Claude's local monitor appears broken.**

   There is a local shell loop still running that does:

   ```bash
   uv run iris --config lib/iris/config/marin.yaml job status /ahmed/decon-val-build-all 2>/dev/null
   ```

   This Iris CLI has `job list`, `job logs`, `job run`, `job stop`, and `job summary`; I did not find a `job status` subcommand. Because stderr is suppressed and the loop only breaks when parsed output contains a terminal state, it can keep sleeping forever while reporting nothing. That contradicts "monitor armed." I did not stop this process.

3. **Root cause: the keep-id universe is not a subset of `val_docs`.**

   I compared the `j090` keep list with the extracted input at
   `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/val_docs/*.parquet`:

   ```text
   keep_j090 = 33161
   val_docs_ids = 57243
   missing = 9
   missing_doc_indices =
   [32032941, 19191226, 28765039, 35022250, 41467542, 37562482, 43813141, 15162949, 25128790]
   ```

   This matches the earlier logbook note that 9 fully-contained docs span two consecutive val windows. Claude assumed all fully-contained keep IDs were present in the already-extracted `val_docs` artifact; that assumption is false. The hard assert caught it, which is good, but the plan was not correct end to end.

4. **Do not "fix" this by lowering expected counts silently.**

   There are two legitimate choices:

   - Recover the 9 missing docs from the original normalized shards by `doc_index` and include them in the filtered doc inputs, preserving the original paranoid counts.
   - Redefine the artifact as `paranoid ∩ extracted_val_docs`, update counts to 13,568 / 27,733 / 33,152, and explicitly log that the 9 two-window fully-contained docs are excluded.

   I prefer recovering the 9 docs. Otherwise the artifact no longer matches the already-published paranoid matrix.

5. **Rerun/resume needs care now that partial outputs exist.**

   `build_intent.json`, `filter_stats/`, and all three `docs/` dirs already exist. If Claude only patches the filter to recover missing docs and reruns with `--resume`, `write_parquet(..., skip_existing=True)` will skip the filter stats and preserve the short outputs. If Claude changes the keep lists/counts, the existing intent should mismatch and block resume.

   Recovery should either delete the current target dirs explicitly, use a new build root, or add a deliberate backfill path for the 9 docs plus a recount. Do not rely on the current `--resume` path for this failed run.

6. **The speed story is only partly true.**

   The filter is fast and correctly one-pass over 231 shards. Tokenization is likely not 231-way: `tokenize()` groups input files with a 100 MB minimum group size, so these small validation sets may tokenize with only a few workers. That is probably fine for 10-25M tokens per set, but it is not as parallel as the shard count suggests.

7. **Tests missed the important invariant.**

   The tests cover nesting, intent mismatch, and routing. They do not assert that every keep ID exists in the source `val_docs` artifact. Add a small test or preflight check in `build_decon_val_sets.py` that loads source IDs and fails before writing anything if `keep_j090 - val_docs_ids` is nonempty.

## Overall

Claude's structure is good: distinct paths, intent hashing, hard asserts, provenance columns, validation-only cache roots, and one-pass filtering are all the right direction. The main mistake is a data-contract mismatch between the locally computed fully-contained doc set and the extracted `val_docs` input. The job failing before tokenization is the correct failure mode; the next step is to fix the input universe, not paper over the count mismatch.

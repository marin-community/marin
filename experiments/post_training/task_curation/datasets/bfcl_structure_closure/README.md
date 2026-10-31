# BFCL structural completion tasks

This curation source creates online training tasks that ask a model to continue an
unfinished assistant message. The disjoint validation split measures the same
completion behavior. Tasks cover JSON objects, arrays, nesting, escapes, tool
and reasoning delimiters, code fences, and repeated tool openers. Some
instructions include the reference answer; these measure conditional
completion, not independent task solving.

The source directory contains `bfcl_complement-manifest.json` and each listed
task's `bfcl_complement/<name>/solution/solve.sh`. The split JSON has
`train_task_ids`, `diagnostic_reserved_task_ids` (validation), and
`parity_task_ids`. Creation checks that training and validation parents are
disjoint and that parity parents enter neither split.

`source.sources()` declares train and validation `RlDataSource` entries with
`pipeline=process_rows`. Its custom reader expands each manifest parent into
closure cases as the curation pipeline normalizes rows. The converter makes
each row an ordinary TaskSpec with a user instruction, an
assistant text prefix, and a `StructureClosureSpec` grader. Select
RolloutEngine's `PrefillHarness` explicitly when training or evaluating; it
grades the raw assistant continuation. Parsed reasoning and tool actions are
unscored. The synthetic `write_result` tool is never executed.

`TrainingPrefix.NEAR_BOUNDARY` advances training continuation prefixes to a point where
closing immediately would lose required content. For example, `["first", "second"]`
may have prefix `["first", "secon`: appending `"]` gives valid JSON but loses
the final `d`. Validation cases remain unchanged. Use `TrainingPrefix.ORIGINAL`
for the initial boundaries.

`ToolPrompt.NATIVE_DECLARED` uses a pinned tokenizer input to serialize tool
calls with its chat template and advertise their schema.
`tasks.paired_task_probes` creates four diagnostic TaskSpecs that cross
declared and undeclared tools with compact and native serialization.

The source factory requires pinned BFCL, split-manifest, and, for native tool
serialization, tokenizer locations. The BFCL artifact is currently available
only as an S3 path and the split-manifest location is not recorded here, so the
factory is not yet in `all_sources()`. Supply those pins before publishing the
source through the catalog.
The [reproduction guide](../../../structure_closure/README.md) covers exact
token exports and historical evaluation audits.

# CLI-Gym native conversion

The converter preserves the pinned dataset's instruction, Dockerfile, Compose
settings and original test script. Source deadlines live in the archived
`task.toml`; unspecified machine capacities remain unspecified. The two declared
2-CPU Compose profiles retain that partial declaration.

The original test script is delivered privately after the actor stops. Its output
is copied to a fresh parser machine. `upstream.json` contains byte-verified parser
files from LiberCoders/CLI-Gym revision
`48bb920b728a25a55a5b442303e901919654599e`, with the upstream MIT license.
The parser's skipped/xfail handling and empty-map all-passed predicate are retained.
Missing source markers raise a parser error; test-command exit status is not a
replacement reward. The root-delivery contract records the remaining dependency
and candidate-output injection gaps.

Malformed upstream YAML produces a typed `source_defect:malformed_task_yaml`
rejection carrying the original task ID, YAML hash and parse diagnostic. No source
text is repaired. Other unmeasured changes fail conversion instead of being dropped.

The dataset supplies no original solution. Its later static-retention receipt must
bind the approved no-oracle ruling, current private pin, cohort and TaskSpec hashes.
QUICK conversion and private publication alone establish no execution or admission.

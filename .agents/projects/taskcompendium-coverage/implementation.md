# First-wave implementation

Status: implemented and locally validated, 2026-09-13. This implements the four first-wave cases in
[suite.json](suite.json). Source and individual fixture inspection stays with
fresh-context Terra or Luna agents. The coordinating agent owns integration
review, example assembly, documentation, and validation scheduling.

## Work boundaries

1. Acquire the pinned instruction-following and competitive-code examples,
   preserving raw data, source identities, and hashes. Add bounded importers and
   reuse the pinned verifier library where its semantics match. Support binary
   and fractional constraint aggregation. Materialize submitted code only in
   the private verifier sandbox; the public task does not require agent tools.
2. Add a final-action submission contract and source-specific private action
   comparison. Preserve original advertised schemas and permitted response
   types. Do not derive public allowed actions, batch size, or ordering from
   the expected answer. A predicted function call must never dispatch a tool.
3. Add a provider interface for source-native domain tools and the pinned
   Workplace example. Reuse the original seeded state and tool behavior.
   Keep tool dispatch in the provider adapter, with the generic harness loop
   handling request/result correlation. Preserve isolated state per execution.
4. Assemble a local first-wave artifact with canonical specifications, public
   tasks, rendering/execution choices, and native Harbor exports. Reuse existing
   answer, code, and repository controls. Keep source grading differences and
   known unsupported contracts explicit.

Shared schema, rendering, execution, and grading files have one implementation
owner. Source adapters initially add separate modules/fixtures/tests and agree
on the shared interfaces before integration. Schema changes require regenerated
JSON/Arrow contracts and local examples; no compatibility shim is planned.

## Verification

Run behavioral tests for good, plausible-but-wrong, malformed, and failed
infrastructure outcomes. For action prediction, exercise extra-call and batch
rules only under the corresponding source configuration. For Workplace, test
wrong/no-op mutations, error recovery, trace correlation, and fresh/concurrent
session isolation.

Run existing safe package tests and appropriate Docker/Harbor tests after
integration. Docker image builds and tests are serialized across workers to
avoid shared-tag races. Run the repository lint entry point, explicit checks
for untracked files, and package type checks. Record which concrete cases ran;
source inspection and export loading do not count as successful execution.

Training, live external models, user simulators, the second-wave source queue,
and repairs to the existing problematic-task ledger remain outside this work.

## Result

The [local first-wave manifest](../../../lib/taskcompendium/examples/first_wave/manifest.json)
contains five specifications and eleven Harbor exports. Instruction and code
answers use plain/JSON/XML variants; prediction records native calls without
dispatch; Workplace executes its 27 seeded domain tools. Existing controls
regenerate as 44 specifications and 88 exports, with their 92 prompts unchanged.

Validation passed 253 safe tests and 55 distinct Docker cases. After the code
answer materialization fix, a 28-case Docker regression run passed. All 99 local
exports load through Harbor and retain their specification hashes. The built
wheel preserves all 14 vendored provider files and initializes its pinned seed
and 27 tools. Package types and repository lint pass. Exact source CSV bytes are
excluded from whitespace rewriting and checked by provenance tests instead.

The [validation record](../../../lib/taskcompendium/examples/first_wave/validation.json)
separates source-fidelity limitations, actual execution evidence, and export-only
checks. The predicted-action comparator retains its source's weak message/type
checks; these are recorded in the problematic-task ledger. Workplace retains the
source's selected-table comparison and string normalization. Neither adapter is
claimed to be a stronger verifier than its source.

The [schema 0.5 publication](https://huggingface.co/datasets/open-athena/taskcompendium-spike/tree/18ad898b8fe5b50043af9d843209d755a4fbc00c)
contains the existing sample and first wave: 49 task rows and 99 lowering rows.
All 14 payload checksums and both dataset configurations were verified remotely.


## Multi-step Workplace follow-up

The derived `nemo/workplace/0-derived-multistep` adds three separately delivered
requests: reply to Carlos and create a follow-up task, move it to In Progress,
then move it to In Review and send the specified notification. One provider state
and conversation persist across native Harbor steps. Each private checker includes
all prior mutations; explicit mean aggregation retains intermediate failures.

The original single-step example is unchanged. The first-wave artifact now has
six specifications and twelve exports; the combined dataset has 50 tasks and
100 lowerings. Native scripted trials produce `[1,1,1]` for correct execution and
`[1,0,1]` for a wrong or omitted middle action. All 257 safe tests pass, including
four new multi-step tests. Types and scoped repository lint pass. No Docker or
live model was needed for this addition.

See the [requests](../../../lib/taskcompendium/examples/first_wave/workplace-multistep.md)
and [successful rollout](../../../lib/taskcompendium/examples/first_wave/workplace-multistep-rollout.json).
Reproduce it with `examples/validate_workplace_multistep.py` in the package.

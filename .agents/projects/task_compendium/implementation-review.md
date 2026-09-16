# Implementation review and Harbor-backed generation

## Scope

Review the current TaskCompendium implementation boundaries, fix concrete defects,
and run the package and Harbor conformance checks. Preserve the semantic schema,
source archives, and design overview. Source quality repairs are a separate sweep.

## Work

1. Review runtime lifecycle and command deadlines against the pinned Harbor API.
   Fix ignored timeout and preservation settings with behavior regressions.
2. Review requirements, binding compatibility, intrinsic answer constraints,
   resource visibility, ordered context, and success-policy validation.
3. Review submission extraction, judge evidence, private execution, and outcome
   persistence. An infrastructure failure must never become a numeric score.
4. Run safe TaskCompendium tests with the Harbor extra, then applicable conformance
   tests. Report actual Docker coverage separately from unavailable checks.
5. Report the implementation review results before starting the consumer work.
6. Prepare TaskCompendium packages for MarinSkyRL's pinned trajectory-generation
   entrypoint. Reuse the current runner and dataset contracts; do not add training
   or direct-chat lowering yet.
7. Generate a mixed batch using the actual consumer interface, with a scripted
   local service where possible. Preserve trajectory identity, tokens, semantic
   reward, and distinct extraction/infrastructure failures. Document the command
   and validation limits.

## Validation

Use public package APIs and real Harbor trial lifecycles for compatibility,
visibility, and verdict regressions. Use fakes only at external boundaries.
Validate consumer generation without starting a trainer or a GPU service.
Review automatic lint findings for current behavior; do not perform cosmetic
changes solely to satisfy a lint suggestion.

## Review results

The focused review fixed submission/verifier compatibility, required provider
tool bindings, unsupported ordered native-agent launches, command deadlines,
Docker preservation, provider failure persistence, and judge evidence paths.
Agent results now include the retained messages expected by SkyRL.

Validation: 317 safe TaskCompendium tests passed (55 excluded); affected Marin
tests passed (220 passed, two skipped); changed-file pre-commit passed. Docker
cleanup was exercised through a fake CLI with persistent resource state, rather
than a live daemon. This is a boundary review, not a complete audit of every
importer or source task.

The automatic catalog review reported 30 suggestions and timed out in its cruft
and prose lanes. The two dropped runtime settings are fixed. Its broader
function-splitting, importer consolidation, typing, and constant suggestions
remain follow-up work; the PR's `agentic-lint` label is still withheld. Raw
fsspec is intentional in the independent package: importing Marin's storage
stack would reverse the dependency boundary. The vendor comparator flags and
provider dictionaries preserve the pinned source interface. Reader and writer
Parquet batch sizes need not agree. Launch options remain separate from the
task-owned binding so export does not select a harness.

## Generation implementation

The pinned MarinSkyRL generation command discards its returned batch and its
generic Harbor builder cannot resolve these mixed task bindings. Add a small
Harbor generation API that retains every attempt, then an explicit SkyRL
consumer adapter to demonstrate the actual `TrajectoryRunner.run` interface.
Keep the adapter in the spike examples until the external runtime dependency
is packaged. Use the pinned token-processing helpers and mark HTTP transcript
tokens as reconstructed; do not fabricate logprobs.

SkyRL's numeric batch cannot represent absent semantic rewards. Persist those
attempts with null reward, and reject a consumer batch containing any such
attempt. Generation can still archive them; a later admission stage can select
graded attempts before forming a training request. This first path does not
start a trainer, GPU engine, or Ray worker.

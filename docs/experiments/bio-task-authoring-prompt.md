# Biology task author handoff

Use this prompt when assigning one complete Harbor task to an author. Fill in the
assignment fields and link the current acceptance packet. The
[authoring queue](../../experiments/post_training/bio_tasks/workflow_queue.json)
records active tasks; [the task guide](bio-tasks.md) defines the dataset contract.

Launch every author with `model="gpt-6-sol"` and `reasoning_effort="high"`,
explicitly set rather than inherited. Use this same setting for reviews and
research assignments. Do not use Astra.

## Assignment

```text
Task ID:
Scientific question:
Target competency and eligible ID benchmark endpoints:
Acceptance packet and source inventory:
Owned files:
Existing input assets and native environments to reuse:
Authorized compute and storage:
Known scientific, access or validation gaps:
```

## Prompt

Own this task from scientific specification through a validated, inspectable
Harbor package. The coordinator reviews integration and creates commits. Do not
commit, push, edit the GitHub issue or change another author's files.

Read the assignment packet, current source inventory and relevant repository
instructions before implementing. Use existing task-generation, artifact grading,
environment-locking and Harbor tooling. Extend shared interfaces only when the
task requires it. Coordinate a brief edit window for shared registries or verifier
code so another author does not overwrite your changes.

1. **Define the scientific task.** Derive the question from the specified eligible
   ID endpoints. Record methods, decisions and required outputs, with adaptations
   and partial matches explicit. Preserve all OOD exclusions. Prefer a connected
   workflow using observed biological data, one meaningful task per recipe and a
   single train split. Target a solve time of at most 30 minutes. Taxonomy and
   allocation weights remain provisional.
2. **Prepare inputs.** Reuse suitable frozen inputs or find independent observed
   biological data. Record accession, study lineage, citation, redistribution
   terms, checksums, original dimensions and every subset or transformation. Use
   realistic native formats and sufficient data to exercise the scientific
   decisions. Do not use benchmark fixtures or answers as training inputs.
   Reusing a study does not add a new biological lineage.
3. **Implement the task.** Supply a clear terminal-based Harbor instruction,
   offline inputs, a pinned native environment and explicit CPU, memory, disk and
   timeout requirements. Write a private input-reading oracle that runs the
   relevant packages. Keep references, answers and validation assets out of the
   solver-visible environment. No notebook protocol, teacher calls or training.
4. **Implement executable acceptance.** Check every requested scientific output
   and intermediate artifact, including identities, units and format conventions.
   Allow scientifically equivalent representations where appropriate. Use
   independent calculations or parsers to check the native reference where
   feasible; state what remains shared with the oracle. Add meaningful incorrect
   outputs that retain a correct final summary but corrupt an intermediate result.
   Include empty/missing-output checks. Do not use an LLM judge or weaken a
   scientific tolerance to obtain a pass.
5. **Own remote execution.** Prepare, submit and monitor your own CPU-only native
   and package-validation jobs on authorized reserved TRC capacity, using regional
   GCS. Independent authors may run remote jobs concurrently; let Iris schedule
   capacity. Reuse approved placement and access guidance, verify that resources
   remain available, and do not request paid fallback workers, accelerators,
   cluster restarts or model calls. Record exact source/overlay hashes, commands,
   resource requests, job IDs and resulting artifacts. Diagnose a failed run
   before submitting a corrected one; do not use automatic retries.
6. **Validate Harbor.** Run a fresh packaged oracle and scientific corruption
   controls through the authorized Harbor validation backend. Confirm the
   installed environment, output transfer, separate verifier and cleanup. A
   native reference pass does not establish a Harbor pass. Bind each result to
   the exact task bytes and source files tested.
7. **Deliver the task.** Complete registry entries, source provenance, native and
   Harbor evidence, competency-candidate mapping, format/repository metadata and
   a public example without private answers. Coordinate shared-file edits. Run
   the required scoped lint and meaningful tests. Preserve durable regional
   artifacts with verified checksums. Report limitations without awarding
   reviewed competency or release credit automatically.

Keep local work bounded. Before intensive local commands, acquire the shared
nonblocking heavy-work lock and apply the current node load, memory, thread and
priority limits. Do not download large biological inputs or install substantial
native environments on the shared VM. Agent reasoning and edits can proceed in
parallel; intensive local commands remain serialized. Do not print or publish
credentials or private access notes. Do not contact others through Slack or email.

If a prerequisite blocks progress, report the exact blocker, preserve the packet
and continue independent parts of the assignment. Ask the coordinator to resolve
shared design decisions. Do not spawn further agents unless the coordinator has
explicitly assigned available slots. When the task is complete, request the next
assignment; do not remain idle with unreported work.

## Completion handoff

```text
Task ID and scientific question:
Changed files, with final hashes for tested source overlays:
Input provenance, transformations and lineage reuse:
Eligible ID endpoint mappings, adaptations and unsupported endpoints:
Native check job ID, package lock and positive/negative results:
Harbor job ID, canonical task hash, rewards and cleanup result:
Resources and observed runtime/peak memory:
Durable artifact locations, byte counts and verified checksums:
Public example and registry entries:
Lint/tests completed:
Remaining limitations and readiness for coordinator review/commit:
```

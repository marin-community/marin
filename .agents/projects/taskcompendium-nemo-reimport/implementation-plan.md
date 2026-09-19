# NeMo environment re-import plan

## Scope

Re-import the two sampled rows from each NeMo corpus whose source state and
verification contract are available now:

- Workplace Assistant: offsets 536 and 1163, using the existing seeded provider.
- Agent Calendar Scheduling and Instruction-Following Calendar v2 remain
  rejected: their source verifier accepts missing or incorrect event names.
- Agentic SWE Pivot remains deferred: its release supplies neither a pinned
  checkout nor a final-state oracle, so a Docker workspace would not preserve
  the source task.

## Implementation

1. Generalize the Workplace importer from its id-0 fixture to Hub rows. Pin
   source revision, split, and sampled offset privately; retain the shared
   provider seed and existing authoritative-state verifier.
2. Store raw source rows and Hub provenance as verifier-only fixtures. Generate
   Harbor provider-tool lowerings for Workplace.
3. Test good, malformed, wrong, and prompt-visibility cases. Verify source
   provenance, task serialization, and Harbor compatibility.

Calendar can be reconsidered only after a corpus-wide, source-backed event-name
mapping is available and a repaired checker enforces it privately.

## Shared runtime policy

The Workplace provider remains the one shared seeded runtime. Calendar grading
is a pure private verifier and does not require a Docker image. Future
repository tasks use a small shared toolchain image plus pinned task resources;
they are blocked on source checkout and test manifests, not container support.

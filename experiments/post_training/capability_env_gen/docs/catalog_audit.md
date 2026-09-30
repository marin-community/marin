# Catalog ingestion and pilot provenance

The expanded `new_catalog.json` is the source for future coverage as of 2026-09-21.
Strict ingestion reports 45 complete curricula, 2,405 sections, and 1,999
capabilities, with zero dropped records or orphaned capabilities. It is 7,638,404
bytes; SHA-256 is
`0ce32038771d4fb0c66498f29604474d6766040e8917f894b31940a6a828149a`.
The validated manifest and audit can be reproduced with:

```bash
uv run --frozen -m capability_pipeline.catalog new_catalog.json \
  --audit-json data/new-catalog-audit.json \
  --all-json data/new-catalog-all-capabilities.json
```

Historical pilot evidence stays bound to the earlier source below. The catalogs
have no capability IDs in common. The full new manifest retains all 1,812
learning-progression edges and their source hash. Each planning, proposal,
review, and repair prompt receives only incoming edges for its capability, with
their enabled scope, transfer rationale, witnesses, and artifact-substitution
test. Raw capability records retain sampling facets and sample tasks unchanged.

Generate a deterministic development cohort with:

```bash
uv run --frozen -m capability_pipeline.catalog new_catalog.json \
  --cohort-json data/new-catalog-cohort-001.json
```

The default cohort selects one capability per subject by canonical record hash:
45 capabilities with 38 incoming progression edges. This sampling rule avoids
depending on the old IDs; it is not evidence of task quality or full coverage.
Use `--cohort-per-subject N` for more examples per subject, or `--all-json` for
every capability. The historical `--pilot-json` selection still requires the
original catalog's IDs.

The earlier `catalog.json` is a complete, strict JSON document. Its authoritative
ingestion audit is:

- status: `complete`
- size: 1,297,781 bytes
- SHA-256: `b3318b9fa7965b70b1cd1aa15513faca63b629db67bdc991a277173b51b5c0c6`
- 34 complete curricula
- 959 complete sections, including 854 complete capabilities
- zero dropped bytes or records and zero orphaned capabilities

During initial development, the file at this path had been overwritten with NUL
bytes from byte 1,047,725 onward. Its SHA-256 was
`4ccb7dc5e2a259afbeac499bf10087c5003a62d7f1df0e874ac7902c62ce24ed`.
That version yielded 25 complete curricula plus 21 individually complete C13
capabilities whose subject metadata appeared after the damage. The recovery audit
marked the source `incomplete`, reported a recoverable-record boundary at byte
1,047,014, and explicitly left the missing record count unknown. No complete copy
was found under `/Users/k3sc0re/openathena` at that time. The user subsequently
replaced it with the complete source above; the damaged file was never rewritten
or presented as complete by this pipeline.

`capability_pipeline.catalog.ingest_catalog` retains the fail-closed recovery path
as a regression-tested guard for future interrupted transfers. It uses strict
local JSON decoding, accepts only fully decoded curriculum or section objects,
and never closes partial objects, reconstructs strings, or infers missing subject
metadata. Synthetic corruption tests cover that path without relying on the
checked-in catalog being damaged.

Regenerate and inspect the audit with:

```bash
uv run python -m capability_pipeline.catalog catalog.json \
  --audit-json data/catalog_audit.json
```

A complete source exits with status 0. A damaged source exits with status 2 after
writing the audit so automation cannot silently treat salvaged coverage as full.

## Pilot selection

`data/pilot.json` contains one capability from each of the 34 subject curricula.
The selection favors workflows that can support realistic files, repositories,
services, simulations, or documents and strong verification signals. It spans
pure reasoning, lightly agentic artifact work, and fully agentic environments,
as well as exact-answer checks, executable tests, and rubric-based review.

Every entry contains the original capability object in full, its original subject
identifier and name, a short selection rationale, and a SHA-256 over canonical
JSON for the capability object. `load_pilot` validates IDs and hashes before the
pipeline consumes the file. Regenerate the deterministic artifact with:

```bash
uv run python -m capability_pipeline.catalog catalog.json \
  --pilot-json data/pilot.json
```

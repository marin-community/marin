# Debugging log for Shuttle evaluation manifest provenance

Keep the checked-in Shuttle evaluation scorecard loadable after the canonical
H100 evidence plan changes, without promoting architecture-nonconforming
results.

## Initial status

On canonical commit `92b5dd7da84492d3766a51b8cb915a08b2a9e56a`, the checked-in
loader test rejected the manifest because its excluded H100 evidence digest
still identified the file from the older baseline.

## Hypothesis 1

The manifest baseline and excluded-evidence record should identify the commit
that records the current evidence plan. The evidence source should identify the
same commit because that revision changes the plan's status claims as well as
its recorded bytes.

## Changes to make

Update the manifest baseline, the excluded evidence source and record commits,
and its SHA-256 digest to the identities at `92b5dd7da8`. Keep the evidence
excluded and architecture-nonconforming. Pin these identities in the checked-in
loader test so a later evidence-plan edit requires an intentional manifest
update.

## Results

The exact checked-in loader test passes after the provenance update. The full
Shuttle test suite passes 38 tests, including all manifest mutation cases. The
repository changed-file pre-commit gate passes, including Pyrefly.

## Future work

- [ ] Update the manifest provenance whenever a linked evidence artifact changes.

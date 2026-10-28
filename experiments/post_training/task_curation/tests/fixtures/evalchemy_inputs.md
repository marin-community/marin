# Evalchemy test inputs

`evalchemy_inputs.json` contains source-shaped rows, reference answers, few-shot
pools and NUPA panel identities used by reader and publication tests. It contains
no captured generated requests, expected grades or expanded trial outputs.

The inputs were selected from migration captures of Marin
`623134de90723e7f189036da005067f36817b6e7`; they include source examples and
constructed boundary cases.
The full captures are retained as local migration evidence under
`scratch/migration-evidence/evalchemy-oracles`; they are not current parity tests.

These fixtures exercise staged auxiliary files, duplicate source identity,
panel joins, verifier-only code resources, canonical instruction resources and
judge transport failures. Source equivalence is checked by the separate pinned
old/new comparison tool. Its coverage ledger states which families were
compared. Passing these fixture tests does not establish equivalence for an
uncompared family.

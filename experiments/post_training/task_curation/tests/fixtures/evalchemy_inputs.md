# Evalchemy test inputs

`evalchemy_inputs.json` contains source-shaped rows, reference answers, few-shot
pools and NUPA panel identities used by reader and publication tests. It contains
no captured generated requests, expected grades or expanded trial outputs.

The inputs were selected from migration captures of Marin
`623134de90723e7f189036da005067f36817b6e7`; they include source examples and
constructed boundary cases.
These fixtures exercise staged auxiliary files, duplicate source identity,
panel joins, verifier-only code resources, canonical instruction resources and
judge transport failures.

The [manual ingestion acceptance record](https://github.com/marin-community/marin/pull/10757#issuecomment-6483684301)
documents actual old/new comparisons for all 96 declarations, with commands,
source pins, row identities, per-family coverage and observed differences.
Comparison scripts and raw captures remain session artifacts. These fixture
tests protect pipeline behavior; they do not supply an equivalence oracle.

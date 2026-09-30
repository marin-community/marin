# Review criteria fixed before launch

Compare source coverage, meaningful unit boundaries, evidence, data identity and
unit-to-data links, tool roles, prerequisites, source-map consistency, and honest
budget/coverage limitations. Independently validate JSONL fields, uniqueness,
references, source locators, and whether another worker could use the inventory.
Counts alone are not quality scores; preserve raw outputs and distinguish source
access failures from scientific errors and unclear prompt instructions.

Review API versus tutorial sources, ingest versus BBKNN prerequisites, observed PBMC3k raw/processed identity, independent PBMC10k/PBMC68k/pancreas/Paul15 records where reached, hidden AnnData state, and queue coverage. These are non-exhaustive reviewer probes.

These criteria are not worker inputs. There is no exhaustive reference inventory.
Any additional probes identified after output inspection will be labeled post hoc.

Round 2 also checks the round 1 failure classes: unsupported identifiers, shared-data identity, exact input stages, final source-map consistency, and continuation while useful inspection time remains. These criteria are fixed before launch and withheld from the worker.

Round 3 repeats the identical round 2 prompt to inspect run-to-run variability before another revision. Review the same failure classes and preserve both positive and negative results.

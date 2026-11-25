# PyMethods2Test native intake

`pymethods2test.tasks.sources()` declares the complete raw validated-task
population of `FWeindel/validated-tasks@317659ac88c8b706d318020c5fc3813a931fb6ec`,
folder `pymethods2test_v1`: 4,990 rows in `pymethods2test_v1/tasks.parquet`
(SHA-256 `cbf2edf6a4870f8503914d974832693f01fd1977ae5f98744d5dc56c1c60f0dd`),
every row with validation stages 1–4 passed. The sealed intake census is
`s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.24.1/pymethods2test-intake-census-v1/census.json`
(SHA-256 `574937d678632ded04ebe284995771436f1682a0fc938b93bad43e5fc09ff781`).

Every row is a complete native Harbor package (`instruction.md`, `metadata.json`,
`task.toml`, `environment/Dockerfile`, `solution/`, `tests/`) whose Parquet
metadata binds it to one of three pristine Apptainer SIF environment bundles in
the same repository. Conversion keeps the original package bytes and the
original Dockerfile build — no image is substituted and no image publication is
required for this intake — and binds the SHARED grader and `RootDeliveryContract`
at conversion (rule C, 2026-11-24): the grader restores the pinned private
`tests/` files after the actor stops, runs the package's original `test.sh`, and
reads its reward file. The declared verifier deadline (720 s) and the actor's
declared deadline (900 s) pass through; no allocation beyond the source
declaration is invented.

The upstream license and lineage scan is HELD: the recorded CANDIDATES P1 basis
(license PASS on the original upstream) does not resolve to a checkable receipt,
and the revision-history scan awaits the user ruling on license-less early
revisions. The declaration makes no publication or admission claim; the intake
population stays in `PENDING_MEASUREMENT_SOURCES` until its census, dedup,
lexical, oracle/NoP and SAMPLE receipts exist.

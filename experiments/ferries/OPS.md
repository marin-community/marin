# Datakit Ferry Operations

Ad-hoc run/stop/validate for `experiments/ferries/datakit_ferry.py`.
The ferry runs download → normalize → dedup (fuzzy document) → consolidate →
tokenize on FineWeb-Edu `sample/10BT`. It runs nightly via the
`Marin - Canary - Datakit - Tier {1,2,3}` GitHub Actions workflows
(`.github/workflows/marin-canary-datakit-tier{1,2,3}.yaml`).

To run, stop, and validate the ferry **by hand** from a dev box — the submit
(a unique `SMOKE_RUN_ID`, the `--cluster=marin`-not-`marin-dev` OS-Login caveat,
`MARIN_PREFIX`), the stop, and the `scripts/datakit/validate_ferry_outputs.py`
checks — see `.agents/runbooks/run-datakit-ferry-manually.md`. Generic
`iris job run` flag semantics live in `lib/iris/OPS.md` "job run gotchas".

# Knotmeter

A Marina application for saved AgentX results. See the [guide](../../../../docs/references/knotmeter.md)
for local previews, imports, data contracts and the comparison with the standalone dashboard.

- `src/`: Vue views and SVG charts; uses Marina's shared browser kit.
- `server/`: FastAPI reads and the app-owned Postgres schema.
- `ingest/`: separate collector-bundle adapter and Marina SQL loader.
- `tests/`: metric, timestamp, replay and concurrent-import contracts.

No AgentX or InferenceX Python package is installed in Marin. No upstream dashboard
code is copied or patched. The collector's pinned upstream processor supplies the
aggregate metrics; the adapter retains its identities and calculations.

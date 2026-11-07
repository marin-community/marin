# Knotmeter

A Marina application for saved AgentX inference measurements. The [reference](../../../../docs/references/knotmeter.md)
covers previews, imports, metric definitions and deployment prerequisites.

- `src/`: React views, hash navigation and Marina CSS tokens.
- `src/upstream/`: adapted InferenceX charts, diagnostics, timelines and corpus flamegraphs.
- `server/`: FastAPI reads and the app-owned Postgres schema.
- `ingest/`: explicit collector/corpus loading and an offline server-series reducer.
- `tests/`: import contracts and separate synthetic UI fixtures.

The frontend and reducer adapt InferenceX revision `315128ce`; see
[THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSE.inferencex](LICENSE.inferencex).
Node runs the build and offline reducer. Marina serves the packaged browser app;
there is no Node serving process. AgentX and InferenceX Python packages stay
outside Marin; this app consumes the collector's saved processor outputs.

```bash
npm --prefix infra/marina/apps/knotmeter ci
npm --prefix infra/marina/apps/knotmeter test
npm --prefix infra/marina/apps/knotmeter run build
uv run --package marin-marina pytest infra/marina/apps/knotmeter/tests
```

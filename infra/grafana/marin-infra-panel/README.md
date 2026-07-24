# Marin infra panel

This internal Grafana panel provides the three dense views that built-in panels
cannot preserve from Marin's retired infra status page:

- seven UTC days of linked nightly status and durations;
- an equal-width main-branch CI history strip;
- W&B hero-training series against cumulative tokens.

The panel receives ordinary Grafana data frames. Upstream credentials and query
logic remain in the Python bridge one directory above.

```bash
npm ci
npm run typecheck
npm run lint
npm run test:ci
npm run build
```

The parent Dockerfile builds `dist/` and copies only that output into Grafana.
The private plugin is image-reviewed and allowlisted by its exact ID;
`marin-infra-panel` is the only unsigned plugin Grafana accepts.

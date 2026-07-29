# Marin infra panel

This internal Grafana panel restores the retired infrastructure status page inside
Grafana. The `status` view combines these sections:

- Seven UTC days of linked nightly status and durations.
- An equal-width main-branch CI history strip.
- Current worker capacity and a 24-hour region history.
- Fleet and resource-pool provisioning status and history.
- W&B hero-training series against cumulative tokens.

The separate `nightlies`, `commits`, and `wandb` views remain available. The panel
receives Grafana data frames. The Python bridge owns credentials, queries, and caches.

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

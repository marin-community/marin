# Zephyr coordinator dashboard

This directory contains the Vue and TypeScript source for the Zephyr coordinator dashboard. It uses the same Rsbuild, Tailwind, and Vue Router structure as the Iris and Finelog dashboards.

The build produces one HTML file at `../src/zephyr/dashboard.html`. The Zephyr wheel includes this file. The coordinator serves it from the same private endpoint as its actor RPC service.

## Build

```bash
npm ci
npm run build:check
```

`build:check` checks the TypeScript source, builds the dashboard, and fails if the committed HTML file is not current. Use `npm run build` after a frontend change to update the HTML file.

## Access

Open the coordinator task in the Iris dashboard. Select its endpoint link. Iris authenticates the browser request and forwards it through `/proxy/<endpoint>/`. The laptop does not need direct access to the Zephyr network. Iris removes browser credentials before it sends the request to the coordinator.

The dashboard uses relative Connect RPC paths. The coordinator rewrites the HTML base element from `X-Forwarded-Prefix`, so page routes and RPC requests stay under the Iris proxy path.

The dashboard is read-only. It shows the physical pipeline plan, stage and shard state, counters, worker state, CPU and memory use, Finelog time series, and links to Iris task pages.

# Hublot

Open `/hublot/` in Marina and enter an exact Finelog run ID, Iris job path, or
execution UID. A link with `?run=<execution-id>` opens the recorded run.
Use the time bar to select a portion of it. Hublot searches the last seven
days on the Marin Finelog hub and offers a UTC start date when lookup fails.
If an ID matches multiple executions, choose one by cluster and job.

Select a stage or stream to open its inspector. Select a batch wait in the
Trainer card to focus the shared time range. Inference shows request counts,
KV cache usage, and mean prefill and decode time per finished request when
those samples are available. Model input and Model output show fleet token
rates. Select **Per GPU** when the configured inference GPU count is available.
The graph samples fleet request counts and token rates automatically for the
selected time range. The Inference chart reuses those counts and samples timing
or KV cache usage when you select that tab. Each sampled point covers up to
30 seconds. Hublot also loads detail for every chart in the selected range,
including charts you have yet to open. The bar at the bottom shows overall
progress. Select **Load detail** to move a chart ahead of background reads.
Select ten minutes or less for 10-second points and per-engine charts.
White chart backgrounds mark gaps between recorded phases.

## Finelog reads

Marina authorizes access to the named app and proxies its read API. Hublot
looks up the controller start and end first. It reads settings with the
timeline for the resolved window of 10 seconds to 7 days. The graph reads up to
twelve short slices, starting near recent activity. Open charts can read
more samples while waiting for detail, up to 72 slices per overview.
The first request reads up to four slices. Later requests read up to twelve
and skip the intervening data. Background detail reads requests and token
rates before inference timing and KV cache. Opening a tab brings its metrics
forward in the queue. Selections over two hours start with a 20-minute page.
Later pages cover up to two hours. Charts fill in as each page arrives.
Each Marina worker runs one Hublot read at a time and waits at least two seconds
after a background read before accepting another. Detail uses time buckets
for long windows and engine samples for short ones. Changing the time range
updates the queue to the new range.
Finelog adds observed counter increments across engines,
skips resets and gaps, and computes latency from matched request counts and
duration totals. It averages gauges within each engine before combining engines.
Counter reads include 15 seconds of samples on either side. The overview looks
30 minutes ahead for spans that finish outside the window. Hublot asks for a
shorter window when a read reaches the record, size, or time limit.

The signal definitions in `web/src/contracts.js` drive both the browser charts
and the server predicates. Run `npm run sync-query-contract` in `web/` after
changing a signal. The app build checks the generated Python contract.

## Validate and release

```sh
npm --prefix infra/marina/apps/hublot/web test
uv run --package marin-marina --group test pytest infra/marina/apps/hublot/tests
HUBLOT_FINELOG_URL=iap+https://iris.oa.dev/proxy/system.log-server \
  uv run marina release infra/marina/apps/hublot --local
```

The loopback preview uses your cached Marin IAP login for Finelog. Open
`/hublot/` at the printed origin. In production, the app uses the Marina
service identity and VPC path to the internal Finelog hub, as Zephyr does.

A Marina operator can register the named app after merge using the normal
application release from a clean current-main checkout. This uses the existing
runtime and its Finelog access.

```sh
uv run marina release infra/marina/apps/hublot --base-version 0
```

After release, sign in at the production `/hublot/` route and open the
largest reference window in the private handoff. Check that the full window loads.

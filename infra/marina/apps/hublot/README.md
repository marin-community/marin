# Hublot

Open `/hublot/` in Marina and enter an exact Finelog run ID, Iris job path, or
execution UID. Hublot searches the last 72 hours on the Marin Finelog hub. Use
**Load window** for an older execution or a different time range. Links can
include `run`, `from`, and `to`, with times in UTC epoch milliseconds. If an ID
matches multiple executions, choose one by cluster and job.

Select a stage or stream to open its inspector. Select a batch wait in the
Trainer card to focus the shared time range. Inference shows request counts,
KV cache usage, and mean prefill and decode time per finished request when
those samples are available. Model input and Model output show fleet token
rates. Select **Per GPU** when the configured inference GPU count is available.

## Finelog reads

Marina authorizes access to the named app and proxies its read API. The backend
uses fixed Finelog queries scoped to one resolved execution and a UTC window of
10 seconds to 60 minutes. It reads counter samples up to 15 seconds outside
the window and looks 30 minutes ahead for completed spans. Finelog pages resume
after the last timestamp and sequence number. The browser computes measurements
after all pages arrive. The backend asks for a shorter window when the read
exceeds the record, response size, or time limit.

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

# Grafana alerting

## Alerting

Grafana unified alerting, provisioned entirely from the files under
`provisioning/alerting/` — contact points, the notification policy tree, and the rules.
File provisioning owns that tree: UI edits to provisioned alerting resources are
rejected by Grafana and would be overwritten by the files anyway. Change the YAML and
redeploy.

Critical rules notify operators immediately: an unreachable cluster or
federation peer, a crash-looping watched component, an admission webhook with no ready endpoints, a
dead production Iris controller, an unhealthy finelog hub or mirror, CoreWeave
storage above 80 percent of quota, or stalled training, a loss spike, silent
telemetry, or an unstable optimizer on an enrolled hero run.
Warning rules remain in Grafana's home alert list without sending email, Slack,
or Loom notifications: a degraded component, a GPU pod that stays node-bound and
nonterminal without finalizers for five minutes after the bridge's two-minute
overdue threshold, and a GB200 rack with fewer than 16 trays Ready for five
minutes (the NVL72 rack spec is 18; a floor rather than an outright outage —
see `gpu_racks` above). The hero training rule selects fresh running
`iris.task_state` roots named `/marin/hero-*-coord` or `/marin/hero-*-coord-<retry>`. It derives
their run IDs and reads exact matches from structured `service=levanter` telemetry. It waits 15 minutes for
training progress or 45 minutes for initialization, then remains pending for five
minutes. Its `notification=hero-run`
route uses the critical receiver and groups notifications by logical run. It does not
require task-to-node GPU attribution. The run ID before `-coord` and the Levanter
trainer ID must match; zero eligible roots produce an explicit healthy
row. A second rule over the same enrolled roots watches their `train_loss`, firing
when the lowest loss of the last five minutes clears the mean plus six standard
deviations of the preceding 55, or when the loss stops being finite. Six sigma is
the band Levanter's skip-step optimizer rejects a step on, and reducing the recent
window to its floor is what separates a divergence from the single excursion
skip-step already absorbs. It takes the `notification=hero-run` route as well: a
hero run diverging unwatched costs more than a false page, which a silence
answers. Both hero rules share one enrolment query per cache interval.

Three more rules carry the rest of the hero on-call policy, the checks the
standalone Pushover monitor applies — see
[the run-health contract](../../../docs/ops/hero-run-health-alerts.md).
`TrainingTelemetryGone` and `TrainingOptimizerUnstable` page on the same
`notification=hero-run` route: telemetry silent for ten minutes, labelled
`telemetry_gone` while Iris still counts the tasks and `run_down` when it no
longer does, a loss floor a whole unit above its trailing floor that the
six-sigma band did not catch, a gradient norm above 2, or three skipped steps in
fifteen minutes. `TrainingRunHealthDegraded` takes the `notification=slack`
route for token drops, router collapse, a throughput or
MFU floor, a worse evaluation, a stale `iris.task_state` row, and retries.

These three watch a wider enrolment: a run that either the Iris rollup or fresh
Levanter `phase` telemetry reports. The stall and loss rules enroll from
`iris.task_state` alone, so a break in that path stops them watching a training
run with no signal that it happened, which is what `iris_state_stale` reports.
One `levanter.metrics` scan per cache interval feeds all three, reduced over the
newest execution process zero reports so a retry cannot mix two attempts; the
loss-jump check filters its two windows to that execution for the same reason.
`TrainingProgressStalled` labels a silent run `telemetry_gone` and emits a zero
rather than firing beside `TrainingTelemetryGone`, so one outage stays one page.
A warning-only Zephyr rule reads fresh
`progress_time_seconds` rows from `service=zephyr` telemetry. It waits 45 minutes after a
stage start or shard completion, then remains pending for five minutes. The
execution ID separates concurrent pipelines under one root job. The stuck-pod
rule groups by node and links the cordon-first
recovery skill; terminal, unbound, and finalizer-held pods stay dashboard-only.
The CoreWeave storage-telemetry freshness and growth warnings and `TrainingRunHealthDegraded`
carry the explicit `notification=slack` exception, so they announce in Slack
and open a session on the default `operator` channel.
`CoreWeaveStorageQuotaPredicted` projects 24 hours ahead using the faster
positive 24-hour or 72-hour usage trend against the live zone quota. These
windows tolerate the observed 4–10-hour collection gaps; the projection
includes the latest sample's age. The critical quota page reads measured
usage from the same deduplicated query, and the telemetry warning fires
after twelve hours without a collection.
`CoreWeaveStorageHeadroomLow` uses `notification=storage-daily` to warn above
80% current or projected quota use in `#marin-alerts` during the daily 15:00–16:00
`America/Los_Angeles` window. Its 23-hour repeat interval permits a reminder
the next afternoon across spring daylight saving time. The
[CoreWeave viewer](https://cw-s3.oa.dev/?v=time) link in the alert shows bucket
sizes and growth from its latest scan. The storage operator investigates growing
prefixes and asks verified owners to review them in the alert thread. The
[storage runbook](../../../scripts/ops/storage/README.md#coreweave-storage-telemetry)
describes sampling requirements and forecast limits.
If Loom delivery fails, the bridge keeps returning an error so Grafana retries,
but reports the failure only once in the alert thread until delivery recovers.
The policy's five-minute group interval bounds each notification attempt; it
does not impose a lifetime retry limit on a firing alert. The daily active
window suppresses storage notifications outside 15:00–16:00 Pacific.

`B200ClusterUnderutilized` uses the same route when a cluster's Ready GB200
GPUs stay idle for 30 minutes; the thresholds are the constants in
`src/b200_utilization_alert.py`. A cluster with incomplete telemetry or an
unreadable node inventory does not fire, and its `state` label says why.
Other workload-tier signals (gated pods, Kueue backlog, workload crashloops) are
dashboard panels rather than alert rules because they have expected benign
causes. `severity=critical` routes to `ops-critical` (email
ops@openathena.ai, and the bridge, which announces the alert in Slack and opens a
Loom triage session on that thread). `severity=warning`
matches the always-active `dashboard-only` mute timing: Grafana continues
evaluating and displaying the alert, but creates no notification. Every rule
sets explicit error and no-data behavior, and the alert endpoints return
explicit zeros when healthy. Every rule alerts on invalid query results and
other execution errors. Finelog alert endpoints convert only retryable service
failures into valid non-firing results because `FinelogFleetUnhealthy` reports
that shared failure. The CoreWeave storage rules stay normal when a successful
query returns no data.

Federation peer reachability comes from `ListPeers` on the Marin controller.
The controller heartbeat traverses production DNS, TLS, Traefik, the source-IP
allowlist, and the Iris RPC path from Marin's static egress address. Grafana
reads the resulting state over its existing VPC path to Marin, so Grafana's
Cloud Run egress does not need admission to the CoreWeave federation ingress.

Alert state — pending (`for`) timers, notification dedup, silences — lives in the
shared `marin-metadata` Postgres with the rest of Grafana's state (see Deploy), so it
survives redeploys. `min=max=1` keeps a single alert evaluator.

Email is optional. SMTP is plain Gmail submission (`smtp.gmail.com:587`, STARTTLS),
sending as grafana@openathena.ai with an app password from Secret Manager; the app
sends mail itself, so deliverability (SPF, spam filtering) rests on the sending
account. The deploy enables SMTP only when the `marin-grafana-smtp-credentials`
secret exists — without it the service still deploys, the email receiver fails
silently, and critical alerts still reach Slack and Loom. After changing contact
points or their credentials, send a test notification to all receivers (Alerting
→ Contact points → Test) rather than trusting config presence.

Slack is deliberately *not* a Grafana receiver at all. A Slack incoming webhook
answers with a bare `ok` and never reveals the message timestamp, so an alert
Grafana posts cannot be joined by anything else — and the timestamp is exactly
what Loom needs to route the thread to the triage session. The bridge posts
instead, which also means one message per alert rather than two side by side.

Both receivers post through the bridge and open a Loom triage run on the alert's
Slack thread. They share one channel, credential, and rendering. `ops-critical`
also sends email; `ops-slack` handles explicitly notified warnings and the root
fallback for unlabeled alerts. A fallback notification does not reach Slack while
the bridge is down; email remains an independent path for critical alerts.

The Loom receiver posts to the bridge on `127.0.0.1`; it is not exposed through
Grafana or IAP. For each firing group the bridge first posts the alert to
`slack_alerts_channel` with `chat.postMessage`, keeping the returned message
timestamp. It then asks the Cloud Run metadata server for a Google-signed
identity token for `https://loom.oa.dev`, exchanges it for a short-lived `ops`
Loom token, and creates an idempotent run for `marin-community/marin` on the
`operator` channel, naming that thread. Loom routes the thread to the triage
session, so the session answers in the thread and an `@russbot` reply there
reaches it instead of launching a second session — see [Loom's
slack-trigger docs](https://github.com/marin-community/loom/blob/main/docs/slack-trigger.md).
The session link is threaded under the announcement.

Generic firing groups feed the live `Grafana operator` session on the `operator`
channel. `TrainingProgressStalled`, `TrainingLossSpike`,
`TrainingTelemetryGone`, and `TrainingOptimizerUnstable` also carry the trusted
`operator_behavior=hero` label, which selects the `operator:hero` channel. Loom
therefore keeps a separate durable coordinator for Hero while both behaviors use
the same four-session `ops` policy pool. Every nonterminal session explicitly
using `ops` counts toward that pool; an operator-launched child with no explicit
profile uses Loom's `default` profile. The extra `ops` capacity covers handoffs or
explicit same-profile delegation but is shared, not reserved per channel.
Repeated notifications for the same alert fingerprint and start time reuse the
same Loom run, and thread a
short "still firing" note under the original announcement. The Slack thread key is
the Grafana notification group key. A replacement alert instance in the same group
therefore keeps the thread even when the prior instance first resolves. Resolved
notifications create no run and are noted on that same thread.
Ordering is deliberate: Slack first, so an alert reaches people even when Loom is
unreachable, and that failure is reported into the thread instead of only into
Grafana's notification history. A Slack failure is logged and still opens the
triage session.

The Hero behavior receives static discovery and query guidance instead of a
precomputed evidence snapshot. It tells the coordinator how to follow the alert's
logical run across execution and coordinator retries, then query current
`telemetry_v1`, `iris.task_state`, `iris.task_event`, and `log` rows. The guidance
uses exact IDs and Finelog's literal `prefix` predicate, includes the deduplicated
event `count`, and asks the coordinator to compare tasks and ranks rather than
grep a fixed error-signature list. Live evidence stays out of the webhook path, so
query latency, truncation, redaction, or an unknown cluster label cannot silently
turn a partial snapshot into the operator's starting facts.

Open threads are tracked for six hours in the bridge's memory, which is sound
because Cloud Run runs this service at `min=max=1`. A revision rollover forgets
them: the next notification for a still-firing alert announces afresh, and a
resolution for an alert this revision never announced is dropped rather than
posted bare.

The Loom Pulumi stack binds the exact `marin-grafana` service-account email and
numeric subject to the `ops` profile. The Grafana stack reads the Loom URL and
profile from that stack's `workloadClients` output, so
the caller and verifier cannot drift through duplicated configuration.


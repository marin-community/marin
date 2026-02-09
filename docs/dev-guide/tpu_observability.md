TPU Observability: Field Manual
---

This document serves as the central reference for operational observability regarding TPU health and preemptions.

**Intended workflow:**

- **[Dashboard](https://console.cloud.google.com/monitoring/dashboards/builder/b6f6266c-883f-457b-81cd-13634ea53b96;duration=P7D?project=hai-gcp-models) (macro):** At-a-glance signals to answer “is something wrong?” (stockouts, widespread terminations, utilization anomalies).
- **CLI (micro):** Ground-truth inventory and drill-down when the dashboard looks suspicious.

# CLI Commands

## Fleet Status (Ground Truth)

The CLI commands below provide the most accurate view of TPU state. Dashboard metrics may lag or underreport.

### Quick Fleet Summary
Get a quick summary of READY TPUs across all zones:

*Note: This uses Cloud Asset Inventory for cross-zone aggregation. It may lag slightly; cores are inferred from the `ray-user-node-type` label (estimate). Use `tpu-vm list` for ground truth within a specific zone.*
```shell
gcloud asset search-all-resources \
  --scope=projects/hai-gcp-models \
  --asset-types=tpu.googleapis.com/Node \
  --limit=5000 \
  --format=json \
| jq -r '
    [
      .[]
      | select(.state=="READY")
      | {
          zone: .location,
          cores: (
            try (
              .labels["ray-user-node-type"]
              | capture("_(?<n>[0-9]+)$").n
              | tonumber
            ) catch 0
          )
        }
    ]
    | sort_by(.zone)
    | group_by(.zone)
    | map({zone: .[0].zone, nodes: length, cores: (map(.cores) | add)})
    | sort_by(.cores)
    | reverse
    | .[]
    | "\(.zone): \(.nodes) READY nodes, \(.cores) READY cores"
  '
```

### Fleet State Breakdown
How many TPUs are in each state (READY, PREEMPTED, CREATING, etc.) by zone:
```shell
gcloud asset search-all-resources \
  --scope=projects/hai-gcp-models \
  --asset-types=tpu.googleapis.com/Node \
  --limit=5000 \
  --format=json \
| jq -r '.[] | "\(.location) \(.state)"' \
| sort \
| uniq -c \
| sort -rn \
| awk '{print $2": "$1" "$3}'
```

### Detailed Fleet Status by Zone
Show TPU count, cores, and accelerator types for a specific zone:
```shell
ZONE="us-central2-b"

gcloud compute tpus tpu-vm list \
  --zone="$ZONE" \
  --project=hai-gcp-models \
  --format="table(name,acceleratorType,state)" \
| head -30
```

### Fleet Summary with Core Counts
Count READY cores by zone (useful for capacity planning):
```shell
ZONE="us-central2-b"

gcloud compute tpus tpu-vm list \
  --zone="$ZONE" \
  --project=hai-gcp-models \
  --filter="state=READY" \
  --format="value(acceleratorType)" \
| sed 's/.*-//' \
| awk '{sum+=$1} END {print "READY cores:", sum+0}'
```

### TPU State Breakdown
Check how many TPUs are in each state (READY, PREEMPTED, CREATING, etc.):
```shell
ZONE="us-central2-b"

gcloud compute tpus tpu-vm list \
  --zone="$ZONE" \
  --project=hai-gcp-models \
  --format="value(state)" \
| sort \
| uniq -c \
| sort -rn
```

### Accelerator Type Breakdown
See what TPU types are running:
```shell
ZONE="us-central2-b"

gcloud compute tpus tpu-vm list \
  --zone="$ZONE" \
  --project=hai-gcp-models \
  --filter="state=READY" \
  --format="value(acceleratorType)" \
| sort \
| uniq -c \
| sort -rn
```

## Troubleshooting Commands

### Check for Rejected TPU Demand (Stockouts or Quota Errors)
```shell
gcloud logging read \
  'protoPayload.methodName="google.cloud.tpu.v2alpha1.Tpu.CreateNode" AND severity>=ERROR' \
  --project=hai-gcp-models \
  --freshness=1h \
  --limit=20 \
  --format="table(timestamp, protoPayload.request.node.accelerator_type, protoPayload.resourceName, protoPayload.status.message)"
```

### Check for Recently Preempted Worker Instances (Low-Level)
This is the low-level “GCE instance got preempted” signal. It often produces multiple events for a single TPU VM node.
```shell
gcloud logging read \
  'protoPayload.methodName="compute.instances.preempted"' \
  --project=hai-gcp-models \
  --limit=20 \
  --format="table(timestamp, protoPayload.resourceName)"
```

### Count Worker Preemptions in Last Hour (Low-Level)
```shell
gcloud logging read \
  'protoPayload.methodName="compute.instances.preempted"' \
  --project=hai-gcp-models \
  --freshness=1h \
  --format="value(resource.labels.zone)" \
| sort \
| uniq -c \
| sort -rn
```

### Check for Recently Terminated TPU Nodes (Macro)
This is the closest “one event ~= one TPU VM node lost” signal available via logs. It’s better than counting `compute.instances.preempted` when you want macro fleet health.
```shell
gcloud logging read \
  'protoPayload.methodName="tpu.nodes.terminate" AND protoPayload.resourceName:"projects/hai-gcp-models/"' \
  --project=hai-gcp-models \
  --freshness=1h \
  --limit=50 \
  --format="table(timestamp, protoPayload.resourceName, protoPayload.metadata.terminateReason)"
```

### Count TPU Node Terminations in Last Hour (Macro)
```shell
gcloud logging read \
  'protoPayload.methodName="tpu.nodes.terminate" AND protoPayload.resourceName:"projects/hai-gcp-models/"' \
  --project=hai-gcp-models \
  --freshness=1h \
  --format=json \
| jq -r '.[].protoPayload.resourceName | capture("locations/(?<zone>[^/]+)/").zone' \
| sort \
| uniq -c \
| sort -rn
```

### Check Specific TPU Node Status
```shell
NODE_NAME="ray-marin-us-central2-worker-xxxxx-tpu"
ZONE="us-central2-b"

gcloud compute tpus tpu-vm describe "$NODE_NAME" \
  --zone="$ZONE" \
  --project=hai-gcp-models
```

### List TPU Workers Across All Zones
*Note: The standard `tpu-vm list` command is zonal. To search across the entire project, use Cloud Asset Inventory.*
```shell
gcloud asset search-all-resources \
  --scope=projects/hai-gcp-models \
  --asset-types=tpu.googleapis.com/Node \
  --limit=5000 \
  --format="table(displayName, location, labels.ray-user-node-type, state)"
```

# Dashboard

## ⇒ [GCP "TPU Capacity & Health" Dashboard](https://console.cloud.google.com/monitoring/dashboards/builder/b6f6266c-883f-457b-81cd-13634ea53b96;duration=P7D?project=hai-gcp-models)

## Metric Accuracy Warning

**Important**: Dashboard metrics have known limitations:

| Metric | Accuracy | Notes |
|--------|----------|-------|
| `quota/allocation/usage` | ⚠️ Unreliable | Shows quota reservations, NOT actual running TPUs. Can show cores when nothing is running. |
| `duty_cycle` | ⚠️ Partial coverage | Great for utilization *trends* where it exists, but it is not a reliable fleet-size metric. Some TPU types/zones don’t emit it, and its series counts don’t map 1:1 to “cores”. |
| `tpu_creation_requests` | ✅ Accurate | Log-based, reliably tracks creation attempts. |
| `tpu_node_terminations` | ✅ Accurate | Log-based, tracks TPU node termination events (macro health signal). |
| `tpu_preemptions` | ✅ Accurate (low-level) | Log-based, counts `compute.instances.preempted` events. Useful for spikes, but often multiple events per TPU VM node. |

**For accurate fleet counts, use CLI commands above.**

## TPU Nodes vs Workers vs Cores

Understanding the difference between nodes and cores is critical for interpreting metrics correctly:

| Term | Description | Example |
|------|-------------|---------|
| **TPU VM node** | A TPU VM resource returned by `gcloud compute tpus tpu-vm list` | One `v4-64` TPU VM node |
| **“Core” (from `acceleratorType`)** | The numeric suffix in `acceleratorType` | `v4-64` ⇒ 64 “cores” |
| **GceTpuWorker** | The monitored resource that emits `duty_cycle` | Many per TPU VM node |
| **`worker_id` label** | Identifier for a GceTpuWorker | Not equal to TPU VM node count |
| **`accelerator_id` label** | Sub-unit index within a worker | Not equal to TPU VM “core” |

**Metric granularity:**

| Metric | Granularity | Notes |
|--------|-------------|-------|
| `duty_cycle` | Per GceTpuWorker / accelerator_id | Great for utilization where present. Do not interpret `count(duty_cycle)` or `count(worker_id)` as “cores” or “nodes”. |
| `quota/allocation/usage` | Quota allocations | Tracks quota reservations, not running TPUs. |
| `tpu_creation_requests` | Per API call | One event = one TPU VM creation attempt. |
| `tpu_node_terminations` | Per TPU node | One event = one TPU node termination (`tpu.nodes.terminate`). |
| `tpu_preemptions` | Per GCE instance | One event = one underlying worker instance preempted; can be many-per-TPU-node. |

This matters for macro dashboards: avoid normalizing by `duty_cycle` “node counts” unless you explicitly want “per duty_cycle worker”.

## How to Read the Dashboard

### Detecting Stockout Periods
Look at **TPU Acquisition Success Rate**:
- **Normal operation**: 20-30% success rate (Ray retries are expected)
- **Stockout**: <10% success rate sustained for >1 hour
- **Easy period**: >40% success rate

### Detecting Termination / Preemption Storms
Prefer **TPU Node Terminations** (macro) over raw **TPU Preemptions** (low-level worker instance preemptions).

- **Normal**: low, sporadic node terminations
- **Storm**: sustained elevated node terminations, especially across multiple zones
- **Severe**: simultaneous spikes in terminations + acquisition success rate dropping

### Widget Reference

| Widget | What it shows | Granularity | Reliability | Concern threshold |
|--------|--------------|-------------|-------------|-------------------|
| TPU Acquisition Success Rate | % of creation requests that succeed | API calls | ✅ High | <10% |
| TPU Node Terminations | TPU node termination events | Nodes | ✅ High | Sustained spikes |
| TPU Creation Requests | Success vs failure creation attempts | API calls | ✅ High | Error >> Success |
| TPU Preemptions (worker) | Underlying `compute.instances.preempted` events | Instances | ✅ High | Spikes (interpret carefully) |
| Fleet Utilization | Avg duty cycle across reporting subset | Series | ⚠️ Medium | Sustained drops |
| Idle duty_cycle (%) | % of duty_cycle series with <1% | Series | ⚠️ Medium | Sustained increases |
| Duty Cycle Coverage | Where `duty_cycle` exists (by location) | Series | ⚠️ Medium | Sudden drops to ~0 |
| Quota Usage by Region | Quota allocation per zone | Allocations | ❌ Low | N/A |
| Quota Usage by Type | Quota allocation by quota metric | Allocations | ❌ Low | N/A |

## Widgets

### Primary Troubleshooting Widgets (High Reliability)

These widgets are based on log events and are the most reliable for troubleshooting.

**1. TPU Acquisition Success Rate (%)**
The percentage of TPU creation attempts that succeed. Use this to detect stockout periods. Based on log events, highly reliable.
```promql
(
  sum(sum_over_time({__name__="logging.googleapis.com/user/tpu_creation_requests", monitored_resource="audited_resource", status="NOTICE"}[${__interval}]))
  /
  sum(sum_over_time({__name__="logging.googleapis.com/user/tpu_creation_requests", monitored_resource="audited_resource"}[${__interval}]))
) * 100
```

**2. TPU Creation Requests by Status**
A custom log-based metric counting the number of creation attempts, segmented by success (NOTICE) or failure (ERROR/Stockouts). This is the most reliable “what’s failing?” view.
```promql
sort_desc(sum by (zone, status)(sum_over_time({__name__="logging.googleapis.com/user/tpu_creation_requests", monitored_resource="audited_resource"}[${__interval}])) > 0)
```

**2b. TPU Creation Requests by Accelerator Type**
Requires the `accelerator_type` label on the `tpu_creation_requests` metric (see Custom Metrics below).
```promql
sort_desc(sum by (zone, accelerator_type, status)(sum_over_time({__name__="logging.googleapis.com/user/tpu_creation_requests", monitored_resource="audited_resource"}[${__interval}])) > 0)
```

**3. TPU Node Terminations by Zone (Macro)**
A custom log-based metric counting TPU node termination events. Use this to detect fleet-level termination storms.
```promql
sort_desc(sum by (zone)(sum_over_time({__name__="logging.googleapis.com/user/tpu_node_terminations", monitored_resource="audited_resource"}[${__interval}])) > 0)
```

**4. TPU Worker Preemptions by Zone (Low-Level)**
This is the raw `compute.instances.preempted` signal. It can be many events per TPU VM node depending on topology.
```promql
sort_desc(sum by (zone)(sum_over_time({__name__="logging.googleapis.com/user/tpu_preemptions", monitored_resource="gce_instance"}[${__interval}])) > 0)
```

### Utilization Widgets (Medium Reliability)

These widgets are based on `duty_cycle`. Use for trends where it exists, not for absolute fleet counts. If coverage (widget 7) is near zero in a location, treat the utilization widgets as untrustworthy there.

**5. Fleet Utilization (avg duty cycle %)**
Average duty cycle across the reporting subset. Use for trend analysis.
```promql
avg(last_over_time({__name__="tpu.googleapis.com/accelerator/duty_cycle"}[${__interval}]))
```

**6. Idle duty_cycle (%)**
Percentage of duty_cycle series with duty cycle < 1%. Identifies underutilized resources within the reporting subset.
```promql
(
  count(last_over_time({__name__="tpu.googleapis.com/accelerator/duty_cycle"}[${__interval}]) < 0.01)
  /
  count(last_over_time({__name__="tpu.googleapis.com/accelerator/duty_cycle"}[${__interval}]))
) * 100
```

**7. Duty Cycle Coverage (by location)**
Shows where `duty_cycle` exists. If a location drops to ~0, utilization widgets become untrustworthy there.
```promql
sort_desc(count by (location)(last_over_time({__name__="tpu.googleapis.com/accelerator/duty_cycle"}[${__interval}])))
```

**8. Active duty_cycle Series Count**
Count of duty_cycle time series currently reporting (coverage proxy, not “cores”).
```promql
count(last_over_time({__name__="tpu.googleapis.com/accelerator/duty_cycle"}[${__interval}]))
```

**9. Active duty_cycle worker_ids Count**
Count of unique `worker_id` values currently reporting (not equal to TPU VM node count).
```promql
count(count by (worker_id)(last_over_time({__name__="tpu.googleapis.com/accelerator/duty_cycle"}[${__interval}])))
```

### Informational Widgets (Low Reliability)

These widgets show quota allocations, NOT actual running TPUs. Use with caution.

**10. TPU Quota Usage by Region**
Shows quota reservations per location. ⚠️ Does NOT reflect actual running TPUs.
```promql
sort_desc(sum by ("location")(last_over_time({"__name__"="serviceruntime.googleapis.com/quota/allocation/usage","monitored_resource"="consumer_quota","service"="tpu.googleapis.com"}[${__interval}])) > 0)
```

**11. TPU Quota Usage by Type**
Shows quota reservations by accelerator type. ⚠️ Does NOT reflect actual running TPUs.
```promql
sort_desc(sum by ("quota_metric")(last_over_time({"__name__"="serviceruntime.googleapis.com/quota/allocation/usage","monitored_resource"="consumer_quota","service"="tpu.googleapis.com"}[${__interval}])) > 0)
```

## Custom Metrics

GCP doesn't export metrics for "TPU Creation Requests" and "TPU Node Terminations" by default, which are highly relevant for us given our setup. Below are the commands used to create these metrics.

**Notes:**

- If you need to change an existing log-based metric’s filter or labels, you must delete and recreate it (this will temporarily break dashboards that depend on it):
  - `gcloud logging metrics delete <metric_name> --project=hai-gcp-models`
- In PromQL, include `monitored_resource=...` for log-based metrics to avoid “multiple possible monitored resource types” query errors.

### Metric: TPU Creation Requests
This tracks every time we ask for a TPU. We extract severity to know if it succeeded (NOTICE) or failed (ERROR) and include `accelerator_type` for macro dashboards.

```shell
echo 'name: tpu_creation_requests
description: Counter for TPU creation attempts (Success vs Rejected)
filter: "protoPayload.methodName=\"google.cloud.tpu.v2alpha1.Tpu.CreateNode\""
metricDescriptor:
  metricKind: DELTA
  valueType: INT64
  labels:
    - key: zone
      valueType: STRING
    - key: status
      valueType: STRING
    - key: accelerator_type
      valueType: STRING
labelExtractors:
  zone: "REGEXP_EXTRACT(protoPayload.resourceName, \"locations/([^/]+)/\")"
  status: "EXTRACT(severity)"
  accelerator_type: "EXTRACT(protoPayload.request.node.accelerator_type)"' | gcloud logging metrics create tpu_creation_requests --config-from-file=- --project=hai-gcp-models
```

### Metric: TPU Node Terminations (Recommended)
This tracks TPU node termination events (`tpu.nodes.terminate`). This is the macro “node got taken away” signal to use for storm detection.

```shell
echo 'name: tpu_node_terminations
description: Counter for TPU node termination events
filter: "protoPayload.methodName=\"tpu.nodes.terminate\" AND protoPayload.resourceName:\"projects/hai-gcp-models/\""
metricDescriptor:
  metricKind: DELTA
  valueType: INT64
  labels:
    - key: zone
      valueType: STRING
labelExtractors:
  zone: "REGEXP_EXTRACT(protoPayload.resourceName, \"locations/([^/]+)/\")"' | gcloud logging metrics create tpu_node_terminations --config-from-file=- --project=hai-gcp-models
```

### Metric: TPU Worker Preemptions (Low-Level)
This tracks the low-level “GCE instance got preempted” signal. Keep this if it’s useful for debugging spikes, but don’t interpret it as “TPU VM nodes lost”.

```shell
echo 'name: tpu_preemptions
description: Counter for TPU worker instance preemption events (low-level)
filter: "protoPayload.methodName=\"compute.instances.preempted\" AND resource.type=\"gce_instance\""
metricDescriptor:
  metricKind: DELTA
  valueType: INT64
  labels:
    - key: zone
      valueType: STRING
labelExtractors:
  zone: "EXTRACT(resource.labels.zone)"' | gcloud logging metrics create tpu_preemptions --config-from-file=- --project=hai-gcp-models
```

## Agents: Checking PromQL queries

PromQL queries can be tested as follows:
```shell
PROJECT="$(gcloud config get-value project 2>/dev/null)"
TOKEN="$(gcloud auth print-access-token 2>/dev/null)"
QUERY='sort_desc(sum by (location)(last_over_time({__name__="serviceruntime.googleapis.com/quota/allocation/usage",monitored_resource="consumer_quota",service="tpu.googleapis.com"}[5m])) > 0)'

curl -sS \
  -H "Authorization: Bearer ${TOKEN}" \
  --get \
  --data-urlencode "query=${QUERY}" \
  "https://monitoring.googleapis.com/v1/projects/${PROJECT}/location/global/prometheus/api/v1/query" \
| jq .
```

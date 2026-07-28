# Training stall alert contract

The stalled-training alert is warning-only. It must not kick, restart, or profile a job. Its operator action is to capture `iris process profile distributed` for each affected task before deciding whether to intervene.

The proposed Grafana rule evaluates once a minute and is pending for five minutes. It emits one row per running root job only when all conditions hold:

- Telltale is fresh within 90 seconds.
- `levanter_step` has not advanced for 15 minutes after a positive step, or a job in `initializing` has stayed at step zero for 45 minutes.
- `iris.task_state` reports the root job running.
- At least 75 percent of its GPU nodes have mean `iris.worker.gpu_util_pct >= 90` over five minutes.

The row label `classification=collective_like` when the median `gpu_power_w / gpu_power_limit_w` is below 0.35. Power is evidence only, not an alert gate.

Required metric producers are `levanter_step`, `levanter_progress_time_seconds`, and numeric `levanter_phase` (`initializing=0`, `training=1`, `finished=2`) in Telltale, plus `iris.worker.gpu_util_pct`, `gpu_power_w`, and `gpu_power_limit_w`. The Iris DCGM producer emits the hardware metrics. `TelltaleTracker` initializes phase/progress, records the wall time after its completed-step `train/loss` callback, and marks a finished run.

The Grafana projection groups by the canonical run label `COALESCE(run, job_id)`. Its query must emit `run`, `classification`, and one numeric `value` column. The metric contract is:

```sql
WITH recent AS (
  SELECT COALESCE(run, job_id) AS run, name, MAX(ts) AS latest_ts, MAX(value) AS latest_value
  FROM "telltale"
  WHERE name IN ('levanter_step', 'levanter_progress_time_seconds', 'levanter_phase')
    AND ts >= now() - INTERVAL '45 minutes'
  GROUP BY 1, 2
)
SELECT run,
  CASE WHEN median_power_ratio < 0.35 THEN 'collective_like' ELSE 'stalled' END AS classification,
  1 AS value
FROM training_stall_candidates
WHERE telltale_age_seconds <= 90
  AND task_state = 'running'
  AND high_util_node_fraction >= 0.75
  AND ((phase = 0 AND step = 0 AND progress_age_seconds >= 2700)
       OR (phase = 1 AND step > 0 AND progress_age_seconds >= 900));
```

`training_stall_candidates` is the required join of the three Telltale values, `iris.task_state`, and five-minute `iris.worker` GPU aggregates by running job. The dashboard and alert provisioning still need this projection wired through the Grafana bridge.

// Dashboard SQL helpers for the iris.task_status finelog namespace.
//
// Task details show the latest retained report. Live job summaries require
// a recent report.

export const TASK_STATUS_NAMESPACE = 'iris.task_status'

// Workers must re-emit faster than this window to remain in live job summaries.
export const TASK_STATUS_RETENTION_INTERVAL = "INTERVAL '10 minutes'"

function sqlString(value: string): string {
  return `'${value.replace(/'/g, "''")}'`
}

const FRESH_CLAUSE = `ts > now() - ${TASK_STATUS_RETENTION_INTERVAL}`

/** Latest retained row for the current submission, including completed tasks. */
export function detailSql(taskId: string, submittedAtMs: number): string {
  return `
SELECT status_text_detail_md, status_text_summary_md
FROM "${TASK_STATUS_NAMESPACE}"
WHERE task_id = ${sqlString(taskId)}
  AND ${submittedAtMs > 0 ? `ts >= to_timestamp_millis(${submittedAtMs})` : 'FALSE'}
ORDER BY ts DESC, attempt_id DESC
LIMIT 1
`.trim()
}

/** One latest summary row per task id, batched. Returns empty SQL when no ids. */
export function batchSummarySql(taskIds: readonly string[]): string {
  if (taskIds.length === 0) return ''
  const list = taskIds.map(sqlString).join(',')
  return `
SELECT task_id, status_text_summary_md
FROM "${TASK_STATUS_NAMESPACE}"
WHERE task_id IN (${list})
  AND ${FRESH_CLAUSE}
QUALIFY row_number() OVER (PARTITION BY task_id ORDER BY ts DESC, attempt_id DESC) = 1
`.trim()
}

// Dashboard SQL helpers for the iris.task_status finelog namespace.
//
// The namespace is append-only; rows from a worker that stopped pushing stay
// on disk until finelog's level-compactor reclaims them. The retention filter
// here is what actually hides a stale task from the UI, so workers must keep
// re-emitting faster than this window.

export const TASK_STATUS_NAMESPACE = 'iris.task_status'

// Workers must re-emit faster than this window or their tasks will fall off the UI.
export const TASK_STATUS_RETENTION_INTERVAL = "INTERVAL '10 minutes'"

function sqlString(value: string): string {
  return `'${value.replace(/'/g, "''")}'`
}

const FRESH_CLAUSE = `ts > now() - ${TASK_STATUS_RETENTION_INTERVAL}`

/** Latest row for a single task within the retention window. */
export function detailSql(taskId: string): string {
  return `
SELECT status_text_detail_md, status_text_summary_md
FROM "${TASK_STATUS_NAMESPACE}"
WHERE task_id = ${sqlString(taskId)}
  AND ${FRESH_CLAUSE}
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

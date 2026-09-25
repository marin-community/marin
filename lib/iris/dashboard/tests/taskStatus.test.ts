import assert from 'node:assert/strict'
import { DatabaseSync } from 'node:sqlite'
import test from 'node:test'

import { detailSql } from '../src/utils/taskStatus.ts'

test('task details keep the latest report after a completed task stops reporting', () => {
  const database = new DatabaseSync(':memory:')
  try {
    database.function('to_timestamp_millis', (value) => new Date(Number(value)).toISOString())
    database.exec(`CREATE TABLE "iris.task_status" (
      task_id TEXT, attempt_id INTEGER, ts TEXT,
      status_text_detail_md TEXT, status_text_summary_md TEXT
    )`)
    const insert = database.prepare('INSERT INTO "iris.task_status" VALUES (?, ?, ?, ?, ?)')
    const taskId = "/owner/job's-coordinator/0"
    const submittedAtMs = Date.parse('2020-01-01T12:00:00.000Z')
    insert.run(taskId, 0, '2020-01-01T12:00:00.000Z', 'Starting', 'Starting')
    insert.run(taskId, 0, '2020-01-01T12:01:00.000Z', '[Zephyr](https://example.com/run)', 'Finished')
    insert.run('/owner/another-job/0', 0, '2020-01-01T12:02:00.000Z', 'Other task', 'Other task')

    const rows = database.prepare(detailSql(taskId, submittedAtMs)).all()
    assert.equal(rows.length, 1)
    assert.equal(rows[0].status_text_detail_md, '[Zephyr](https://example.com/run)')
    assert.equal(rows[0].status_text_summary_md, 'Finished')
    assert.deepEqual(database.prepare(detailSql('/owner/no-report/0', submittedAtMs)).all(), [])
    assert.deepEqual(database.prepare(detailSql(taskId, 0)).all(), [])

    const resubmittedAtMs = Date.parse('2020-01-01T12:10:00.000Z')
    assert.deepEqual(database.prepare(detailSql(taskId, resubmittedAtMs)).all(), [])
    insert.run(taskId, 0, '2020-01-01T12:11:00.000Z', 'New run', 'Starting')
    assert.equal(database.prepare(detailSql(taskId, resubmittedAtMs)).get()?.status_text_detail_md, 'New run')
  } finally {
    database.close()
  }
})

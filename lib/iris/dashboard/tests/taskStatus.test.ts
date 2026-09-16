import assert from 'node:assert/strict'
import { DatabaseSync } from 'node:sqlite'
import test from 'node:test'

import { detailSql } from '../src/utils/taskStatus.ts'

test('task details keep the latest report after a completed task stops reporting', () => {
  const database = new DatabaseSync(':memory:')
  try {
    database.exec(`CREATE TABLE "iris.task_status" (
      task_id TEXT, attempt_id INTEGER, ts TEXT,
      status_text_detail_md TEXT, status_text_summary_md TEXT
    )`)
    const insert = database.prepare('INSERT INTO "iris.task_status" VALUES (?, ?, ?, ?, ?)')
    const taskId = "/owner/job's-coordinator/0"
    insert.run(taskId, 0, '2020-01-01 12:00:00', 'Starting', 'Starting')
    insert.run(taskId, 0, '2020-01-01 12:01:00', '[Zephyr](https://example.com/run)', 'Finished')
    insert.run('/owner/another-job/0', 0, '2020-01-01 12:02:00', 'Other task', 'Other task')

    const rows = database.prepare(detailSql(taskId)).all()
    assert.equal(rows.length, 1)
    assert.equal(rows[0].status_text_detail_md, '[Zephyr](https://example.com/run)')
    assert.equal(rows[0].status_text_summary_md, 'Finished')
    assert.deepEqual(database.prepare(detailSql('/owner/no-report/0')).all(), [])
  } finally {
    database.close()
  }
})

import assert from 'node:assert/strict'
import test from 'node:test'

import { expandDashboardSql } from './dashboardSql.ts'

const base = {
  fromMs: 1000,
  toMs: 2000,
  intervalMs: 100,
  variables: { job_id: 'trainer' },
}

test('expands typed time and string literals', () => {
  assert.equal(
    expandDashboardSql(
      'SELECT {{from_ms}}, {{to_ms}}, {{interval_ms}}, {{job_id}}',
      base,
    ),
    "SELECT 1000, 2000, 100, 'trainer'",
  )
})

test('keeps adversarial viewer input inside one SQL string literal', () => {
  const values = {
    ...base,
    variables: { job_id: "job'; DROP TABLE telemetry_v1; --" },
  }
  assert.equal(
    expandDashboardSql('SELECT * FROM telemetry_v1 WHERE job_id = {{job_id}}', values),
    "SELECT * FROM telemetry_v1 WHERE job_id = 'job''; DROP TABLE telemetry_v1; --'",
  )
})

test('rejects unknown, malformed, and NUL-containing variables', () => {
  assert.throws(() => expandDashboardSql('SELECT {{missing}}', base), /unknown dashboard variable/)
  assert.throws(() => expandDashboardSql('SELECT {{job-id}}', base), /invalid dashboard macro/)
  assert.throws(
    () => expandDashboardSql('SELECT {{job_id}}', { ...base, variables: { job_id: 'bad\0value' } }),
    /NUL bytes/,
  )
})

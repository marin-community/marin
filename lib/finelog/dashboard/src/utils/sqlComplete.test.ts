import assert from 'node:assert/strict'
import test from 'node:test'

import { completionsFor, quoteIdentifier, tokenAt } from './sqlComplete.ts'

const schema = [
  {
    namespace: 'iris.task',
    columns: [{ name: 'status', type: 'string' }],
  },
  {
    namespace: 'telemetry_v1',
    columns: [
      { name: 'timestamp_ms', type: 'timestamp_ms' },
      { name: 'name', type: 'string' },
      { name: 'cluster', type: 'string' },
      { name: 'value', type: 'float64' },
    ],
  },
  {
    namespace: 'log',
    columns: [
      { name: 'epoch_ms', type: 'int64' },
      { name: 'key', type: 'string' },
      { name: 'data', type: 'string' },
      { name: 'cluster', type: 'string' },
    ],
  },
]

test('reads the identifier under the caret, not the whole statement', () => {
  const sql = 'SELECT clus FROM x'
  assert.deepEqual(tokenAt(sql, 11), { text: 'clus', start: 7 })
  // Caret in leading whitespace has no token to complete.
  assert.deepEqual(tokenAt(sql, 7), { text: '', start: 7 })
})

test('offers only namespaces after FROM', () => {
  const found = completionsFor('SELECT * FROM tele', 18, schema)
  assert.deepEqual(
    found.map((c) => [c.value, c.kind]),
    [['telemetry_v1', 'namespace']],
  )
})

test('offers columns elsewhere, prefix matches first', () => {
  const found = completionsFor('SELECT na', 9, schema)
  assert.equal(found[0].value, 'name')
  assert.equal(found[0].kind, 'column')
})

test('credits a shared column to every namespace holding it', () => {
  const [cluster] = completionsFor('SELECT cluster', 14, schema)
  assert.equal(cluster.value, 'cluster')
  assert.equal(cluster.detail, '2 namespaces · string')
})

test('ranks columns of the namespaces the statement already names', () => {
  const found = completionsFor('SELECT k FROM log', 8, schema)
  // `key` belongs to the namespace in the FROM clause; DISTINCT is a keyword
  // that also contains a k, and must not outrank it.
  assert.equal(found[0].value, 'key')
})

test('falls back to substring matches when nothing starts with the token', () => {
  const found = completionsFor('SELECT stamp', 12, schema)
  assert.equal(found[0].value, 'timestamp_ms')
})

test('inserts identifiers the dialect would swallow in quoted form', () => {
  // `CLUSTER BY` is dialect syntax, so a bare `cluster` ends the select list.
  assert.equal(quoteIdentifier('cluster'), '"cluster"')
  // A dotted namespace reads as schema-qualified unless quoted.
  assert.equal(quoteIdentifier('iris.task'), '"iris.task"')
  assert.equal(quoteIdentifier('name'), 'name')
})

test('completion inserts the quoted form while listing the plain name', () => {
  const [cluster] = completionsFor('SELECT name, clus', 17, schema)
  assert.equal(cluster.value, 'cluster')
  assert.equal(cluster.insert, '"cluster"')

  const [ns] = completionsFor('SELECT * FROM iris', 18, schema)
  assert.equal(ns.value, 'iris.task')
  assert.equal(ns.insert, '"iris.task"')
})

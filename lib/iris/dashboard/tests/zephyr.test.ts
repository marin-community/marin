import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { DatabaseSync } from 'node:sqlite'
import { test } from 'node:test'
import { relativePayload, shuffleSnapshotsSql, stageView, joinStep, stageLayout } from '../src/utils/zephyr.ts'

const linear = [
  { stage_name: 'stage0-Scatter', label: 'Scatter', stage_type: 'map_worker', has_reduce: false, dependencies: [] },
  { stage_name: 'stage1-Reduce', label: 'Reduce', stage_type: 'reduce_worker', has_reduce: true, dependencies: ['stage0-Scatter'] },
]
const joined = [
  linear[0],
  { ...linear[0], stage_name: 'join-right-1-0-stage0' },
  { ...linear[1], stage_name: 'join-right-1-0-stage1', dependencies: ['join-right-1-0-stage0'] },
  { ...linear[1], stage_name: 'stage1-Reduce → Join', dependencies: ['stage0-Scatter', 'join-right-1-0-stage1'] },
]

test('join plans default to Graph and saved choices override either plan default', () => {
  assert.equal(stageView(linear, null), 'List')
  assert.equal(stageView(joined, null), 'Graph')
  assert.equal(stageView(linear, 'Graph'), 'Graph')
  assert.equal(stageView(joined, 'List'), 'List')
  assert.equal(stageView(joined, 'obsolete'), 'Graph')
})

test('right-input labels identify steps within their own join operation', () => {
  const anotherBranch = { ...linear[0], stage_name: 'join-right-1-1-stage0' }
  const stages = [...joined, anotherBranch]
  assert.deepEqual(joinStep(joined[1], stages), { parentStage: 1, step: 1, total: 2 })
  assert.deepEqual(joinStep(joined[2], stages), { parentStage: 1, step: 2, total: 2 })
  assert.deepEqual(joinStep(anotherBranch, stages), { parentStage: 1, step: 1, total: 1 })
  assert.equal(joinStep(linear[0], stages), null)
})

test('graph shows a right-input chain feeding the join separately from the left input', () => {
  const graph = stageLayout(joined)
  const [left, rightStart, rightEnd, joinNode] = graph.nodes
  assert.equal(left.y, joinNode.y)
  assert.equal(rightStart.y, rightEnd.y)
  assert.notEqual(left.y, rightStart.y)
  assert.ok(rightStart.x < rightEnd.x && rightEnd.x < joinNode.x)
  assert.deepEqual(graph.edges.map(edge => edge.key).sort(), [
    'join-right-1-0-stage0:join-right-1-0-stage1',
    'join-right-1-0-stage1:stage1-Reduce → Join',
    'stage0-Scatter:stage1-Reduce → Join',
  ])
})

test('payload ratios distinguish unreported, empty, and skewed reducers', () => {
  assert.equal(relativePayload(null, 10), '—')
  assert.equal(relativePayload(0, 10), '0×')
  assert.equal(relativePayload(5500, 10), '550×')
  assert.equal(relativePayload(5635, 10), '564×')
  assert.equal(relativePayload(19, 10), '1.9×')
  assert.equal(relativePayload(33, 16.5), '2×')
  assert.equal(relativePayload(10, 0), 'median 0')
  assert.equal(relativePayload(10, null), '—')
})

test('Iris and Grafana select measured retries over placeholders and preserve observed zeros', context => {
  const dashboard = JSON.parse(readFileSync(new URL('../../../../infra/grafana/dashboards/zephyr.json', import.meta.url), 'utf8'))
  const precedence = (sql: string) => sql.match(/PARTITION BY target_shard ORDER BY (.*?)\) AS sample_rank/s)![1].replace(/\s+/g, ' ').trim()
  const iris = precedence(shuffleSnapshotsSql('execution', 'stage', 1))
  const queries: string[] = []
  function collect(value: unknown) {
    if (typeof value === 'string' && value.includes('PARTITION BY target_shard')) queries.push(value)
    else if (value && typeof value === 'object') Object.values(value).forEach(collect)
  }
  collect(dashboard)
  assert.ok(queries.length > 0)
  const database = new DatabaseSync(':memory:')
  context.after(() => database.close())
  database.exec(`CREATE TABLE samples (target_shard INTEGER, input_rows INTEGER, attempt INTEGER, ts INTEGER, seq INTEGER);
INSERT INTO samples VALUES
(0,7,0,1,1), (0,NULL,0,9,2),
(1,3,0,9,3), (1,8,1,2,4),
(2,0,0,1,5), (2,NULL,0,9,6),
(3,NULL,0,1,7),
(4,9,0,1,8), (4,11,0,1,9),
(5,1,0,1,10), (5,2,0,2,11)`)
  for (const order of [iris, ...queries.map(precedence)]) {
    const result = database.prepare(`WITH ranked AS (
SELECT *, ROW_NUMBER() OVER (PARTITION BY target_shard ORDER BY ${order}) AS sample_rank FROM samples
) SELECT target_shard, input_rows, attempt FROM ranked WHERE sample_rank = 1 ORDER BY target_shard`).all()
    assert.deepEqual(result.map(row => [row.target_shard, row.input_rows, row.attempt]), [
      [0, 7, 0], [1, 8, 1], [2, 0, 0], [3, null, 0], [4, 11, 0], [5, 2, 0],
    ])
  }
})

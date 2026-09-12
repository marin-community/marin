import { sqlString } from './taskStatus.ts'

export const REDUCER_PAGE_SIZE = 20
export const EXECUTION_LIMIT = 100
export const STAGE_NODE_WIDTH = 224
export const STAGE_NODE_HEIGHT = 96
const STAGE_COLUMN_GAP = 32
const STAGE_ROW_GAP = 28
const STAGE_PADDING = 8

export const STAGE_VIEW_STORAGE_KEY = 'iris-zephyr-stage-view'
export const STAGE_VIEWS = ['List', 'Graph'] as const
export type StageView = typeof STAGE_VIEWS[number]

export interface ExecutionStage {
  stage_name: string
  label: string
  stage_type: string
  has_reduce: boolean
  dependencies: string[]
}

export function stageLayout(stages: ExecutionStage[]) {
  const nodes: Array<ExecutionStage & { x: number; y: number }> = []
  const columns = new Map<string, number>()
  const lanes = new Map<string, number>()
  let nextLane = 0
  for (const stage of stages) {
    const column = stage.dependencies.length
      ? Math.max(...stage.dependencies.map(dependency => {
        const parent = columns.get(dependency)
        if (parent === undefined) throw new Error(`Missing or unordered stage dependency: ${dependency}`)
        return parent + 1
      })) : 0
    const row = stage.dependencies.length ? lanes.get(stage.dependencies[0])! : nextLane++
    columns.set(stage.stage_name, column)
    lanes.set(stage.stage_name, row)
    nodes.push({
      ...stage,
      x: column * (STAGE_NODE_WIDTH + STAGE_COLUMN_GAP) + STAGE_PADDING,
      y: row * (STAGE_NODE_HEIGHT + STAGE_ROW_GAP) + STAGE_PADDING,
    })
  }
  const byName = new Map(nodes.map(node => [node.stage_name, node]))
  const edges = nodes.flatMap(node => node.dependencies.map(dependency => {
    const parent = byName.get(dependency)!
    return {
      key: `${dependency}:${node.stage_name}`,
      x1: parent.x + STAGE_NODE_WIDTH, y1: parent.y + STAGE_NODE_HEIGHT / 2,
      x2: node.x, y2: node.y + STAGE_NODE_HEIGHT / 2,
    }
  }))
  return {
    nodes, edges,
    width: Math.max(STAGE_NODE_WIDTH + STAGE_PADDING * 2, ...nodes.map(node => node.x + STAGE_NODE_WIDTH + STAGE_PADDING)),
    height: Math.max(STAGE_NODE_HEIGHT + STAGE_PADDING * 2, ...nodes.map(node => node.y + STAGE_NODE_HEIGHT + STAGE_PADDING)),
  }
}

export function stageView(stages: ExecutionStage[], saved: string | null): StageView {
  if (saved === 'List' || saved === 'Graph') return saved
  return stages.some(stage => stage.stage_name.startsWith('join-right-')) ? 'Graph' : 'List'
}

export function joinStep(stage: ExecutionStage, stages: ExecutionStage[]): { parentStage: number; step: number; total: number } | null {
  const branch = stage.stage_name.match(/^join-right-(\d+)-(\d+)-stage\d+$/)
  if (!branch) return null
  const prefix = `join-right-${branch[1]}-${branch[2]}-stage`
  const chain = stages.filter(item => item.stage_name.startsWith(prefix))
  const step = chain.findIndex(item => item.stage_name === stage.stage_name) + 1
  return { parentStage: Number(branch[1]), step, total: chain.length }
}

export interface ExecutionPlan {
  execution_id: string
  coordinator_job_id: string
  ts: number
  input_shards: number
  stages_json: string
}

export interface StageStat {
  stage_name: string
  status: string
  elapsed: number
  items: number
  total_shards: number
  mem_peak_bytes_max: number
}

export interface ReducerStat {
  target_shard: number
  input_rows: number | null
  payload_bytes: number | null
  num_sources: number | null
  attempt: number
  task_status?: string | null
}

export interface ShuffleSummary {
  persisted_targets: number
  observed_targets: number
  expected_targets: number | null
  median_rows: number | null
  max_rows: number | null
  median_bytes: number | null
  max_bytes: number | null
}

export function timePredicate(startMs: number): string {
  return `ts >= TIMESTAMP ${sqlString(new Date(startMs).toISOString())} AND ts <= now()`
}

export function executionPlansSql(jobId: string, startMs: number): string {
  const root = '/' + jobId.split('/').filter(Boolean).slice(0, 2).join('/')
  return `SELECT execution_id, coordinator_job_id, ts, input_shards, stages_json
FROM "zephyr.execution"
WHERE root_job_id = ${sqlString(root)} AND ${timePredicate(startMs)}
  AND (coordinator_job_id = ${sqlString(jobId)} OR starts_with(coordinator_job_id, ${sqlString(jobId + '/')}))
QUALIFY ROW_NUMBER() OVER (PARTITION BY execution_id ORDER BY ts DESC, seq DESC) = 1
ORDER BY ts DESC LIMIT ${EXECUTION_LIMIT}`
}

export function stageStatsSql(execution: string, startMs: number): string {
  return `SELECT stage_name, status, elapsed, items, total_shards, mem_peak_bytes_max
FROM "zephyr.stage" WHERE execution_id = ${sqlString(execution)} AND ${timePredicate(startMs)}
QUALIFY ROW_NUMBER() OVER (PARTITION BY stage_name ORDER BY ts DESC, seq DESC) = 1`
}

export function shuffleSnapshotsSql(execution: string, stage: string, startMs: number): string {
  return `WITH snapshots AS (
SELECT *, ROW_NUMBER() OVER (
PARTITION BY target_shard ORDER BY attempt DESC, (input_rows IS NOT NULL) DESC, ts DESC, seq DESC
) AS sample_rank
FROM "zephyr.shuffle" WHERE execution_id = ${sqlString(execution)}
AND stage_name = ${sqlString(stage)} AND ${timePredicate(startMs)}
)`
}

export function reducerStatsSql(execution: string, stage: string, startMs: number, page: number): string {
  return `${shuffleSnapshotsSql(execution, stage, startMs)}
SELECT target_shard, input_rows, payload_bytes, num_sources, attempt FROM snapshots
WHERE sample_rank = 1 ORDER BY payload_bytes DESC NULLS LAST, target_shard
LIMIT ${REDUCER_PAGE_SIZE} OFFSET ${page * REDUCER_PAGE_SIZE}`
}

export function reducerTaskStatsSql(execution: string, stage: string, startMs: number, page: number): string {
  return `WITH targets AS (${reducerStatsSql(execution, stage, startMs, page)}),
task_states AS (
SELECT shard_idx, status FROM "zephyr.worker"
WHERE execution_id = ${sqlString(execution)} AND stage_name = ${sqlString(stage)} AND ${timePredicate(startMs)}
AND shard_idx IN (SELECT target_shard FROM targets)
QUALIFY ROW_NUMBER() OVER (PARTITION BY shard_idx ORDER BY ts DESC, seq DESC) = 1
)
SELECT targets.*, task_states.status AS task_status FROM targets
LEFT JOIN task_states ON targets.target_shard = task_states.shard_idx
ORDER BY payload_bytes DESC NULLS LAST, target_shard`
}

export function relativePayload(value: number | null, median: number | null): string {
  if (value === null || median === null) return '—'
  if (median === 0) return 'median 0'
  const ratio = value / median
  return `${ratio.toLocaleString(undefined, { maximumFractionDigits: ratio >= 10 ? 0 : 1 })}×`
}

export function shuffleSummarySql(execution: string, stage: string, startMs: number): string {
  return `${shuffleSnapshotsSql(execution, stage, startMs)}
SELECT COUNT(*) AS persisted_targets, COUNT(input_rows) AS observed_targets,
MAX(num_targets) AS expected_targets, MEDIAN(CAST(input_rows AS DOUBLE)) AS median_rows,
MAX(input_rows) AS max_rows, MEDIAN(CAST(payload_bytes AS DOUBLE)) AS median_bytes,
MAX(payload_bytes) AS max_bytes FROM snapshots WHERE sample_rank = 1`
}

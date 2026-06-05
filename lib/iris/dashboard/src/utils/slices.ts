/**
 * Slice view-model helpers for the Autoscaler tab.
 *
 * A "slice" is the autoscaler's unit of scaling: one CPU VM, or one TPU pod
 * made of several worker hosts. The dashboard renders one row per slice, so it
 * needs a single resolved state plus the per-slice task occupancy aggregated
 * across the slice's hosts.
 *
 * Two facts drive the logic here:
 *  - A booting slice has *no VMs yet* — its workers haven't registered. State
 *    therefore comes from the authoritative `SliceInfo.state` field, not from
 *    the VM list (which would render a booting slice as "unknown").
 *  - A co-scheduled job spans every host of a slice, so the raw scheduler
 *    buckets list the same job once per host. We dedupe to one entry per job.
 */
import type { SliceInfo } from '@/types/rpc'
import { timestampMs } from '@/utils/formatting'

export type SliceLifecycle = 'requesting' | 'booting' | 'initializing' | 'ready' | 'failed'

/** Display status: lifecycle plus the orthogonal in-use / idle overlay on ready slices. */
export type SliceStatus = SliceLifecycle | 'in_use' | 'idle'

/** One job's occupancy on a slice, aggregated across the slice's hosts. */
export interface SliceJob {
  jobId: string
  userId: string
  /** Total running tasks of this job on the slice (gang members count once each). */
  taskCount: number
  /** Number of the slice's hosts running this job. */
  hostCount: number
}

export interface SliceView {
  sliceId: string
  lifecycle: SliceLifecycle
  status: SliceStatus
  /** Wall-clock age since creation, or null if unknown. */
  ageMs: number | null
  /** Number of worker hosts in the slice (0 while booting). */
  hostCount: number
  /** Total running tasks across the slice's hosts. */
  taskCount: number
  /** Distinct jobs occupying the slice, deduped across hosts. */
  jobs: SliceJob[]
  errorMessage: string
  /** How long the slice has been idle (ready but no tasks), or null. */
  idleForMs: number | null
}

const LIFECYCLE_VALUES: ReadonlySet<string> = new Set([
  'requesting',
  'booting',
  'initializing',
  'ready',
  'failed',
])

/**
 * Resolve a slice's lifecycle state from the authoritative `SliceInfo.state`.
 *
 * The controller serves both this dashboard and the status RPC, so `state` is
 * always one of the known lifecycle values. The default only guards against an
 * unexpected wire value, and resolves to "booting" rather than "unknown".
 */
export function sliceLifecycle(slice: SliceInfo): SliceLifecycle {
  const state = (slice.state ?? '').toLowerCase()
  return LIFECYCLE_VALUES.has(state) ? (state as SliceLifecycle) : 'booting'
}

/**
 * Build a per-slice view-model.
 *
 * @param slice       the slice from the autoscaler status
 * @param workerJobs  map of worker_id → jobs running on it (from the scheduler).
 *                    Falls back to the VM's vm_id when worker_id is unset.
 * @param now         current epoch ms (injected for testability)
 */
export function buildSliceView(
  slice: SliceInfo,
  workerJobs: Map<string, SliceJob[]>,
  now: number,
): SliceView {
  const vms = slice.vms ?? []
  const lifecycle = sliceLifecycle(slice)

  // Task occupancy. running_task_count (from the autoscaler RPC) drives the
  // in-use overlay so it matches the group-level badge counts; the per-job
  // breakdown comes from the scheduler buckets.
  let taskCount = 0
  const jobMap = new Map<string, SliceJob>()
  for (const vm of vms) {
    taskCount += vm.runningTaskCount ?? 0
    const workerId = vm.workerId || vm.vmId
    for (const chip of workerJobs.get(workerId) ?? []) {
      const existing = jobMap.get(chip.jobId)
      if (existing) {
        existing.taskCount += chip.taskCount
        existing.hostCount += chip.hostCount
      } else {
        jobMap.set(chip.jobId, { ...chip })
      }
    }
  }
  const jobs = Array.from(jobMap.values()).sort((a, b) => b.taskCount - a.taskCount)

  // If the scheduler reported jobs but running_task_count lagged, trust the
  // scheduler so a slice never shows jobs with a zero count.
  if (taskCount === 0 && jobs.length > 0) {
    taskCount = jobs.reduce((n, j) => n + j.taskCount, 0)
  }

  let status: SliceStatus = lifecycle
  let idleForMs: number | null = null
  if (lifecycle === 'ready') {
    if (slice.idle) {
      status = 'idle'
      const since = timestampMs(slice.lastActive)
      idleForMs = since ? now - since : null
    } else if (taskCount > 0) {
      status = 'in_use'
    }
  }

  const createdMs = timestampMs(slice.createdAt)
  return {
    sliceId: slice.sliceId,
    lifecycle,
    status,
    ageMs: createdMs ? now - createdMs : null,
    hostCount: vms.length,
    taskCount,
    jobs,
    errorMessage: slice.errorMessage ?? '',
    idleForMs,
  }
}

/** Sort key so the operator-actionable slices (failed, then longest-booting) come first. */
const STATUS_RANK: Record<SliceStatus, number> = {
  failed: 0,
  requesting: 1,
  booting: 1,
  initializing: 1,
  idle: 2,
  in_use: 3,
  ready: 4,
}

/** Order slices for display: problems first, then booting (oldest first), then healthy. */
export function sortSliceViews(views: SliceView[]): SliceView[] {
  return views.slice().sort((a, b) => {
    const ra = STATUS_RANK[a.status]
    const rb = STATUS_RANK[b.status]
    if (ra !== rb) return ra - rb
    // Within booting, surface the oldest (most likely stuck) first.
    if (ra === 1) return (b.ageMs ?? 0) - (a.ageMs ?? 0)
    return a.sliceId.localeCompare(b.sliceId)
  })
}

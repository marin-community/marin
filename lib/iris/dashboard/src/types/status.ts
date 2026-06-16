/**
 * Status enums, display names, and Tailwind color class mappings for job and task states.
 *
 * Proto enums serialize as strings like "JOB_STATE_RUNNING" or "TASK_STATE_PENDING".
 * The dashboard normalizes these to lowercase names ("running", "pending") via stateToName().
 */
import type { SliceStatus } from '@/utils/slices'

// -- Normalized state values (lowercase, prefix-stripped) --

export type JobState =
  | 'unspecified'
  | 'pending'
  | 'building'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'killed'
  | 'worker_failed'
  | 'unschedulable'

export type TaskState =
  | 'unspecified'
  | 'pending'
  | 'building'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'killed'
  | 'worker_failed'
  | 'unschedulable'
  | 'assigned'
  | 'preempted'
  | 'cosched_failed'

// The DB stores state as an integer column. This maps integer values to
// normalized state names so the dashboard works with both proto enum strings
// (e.g. "JOB_STATE_RUNNING") and raw integer values from the query API.
const STATE_INT_MAP: Record<number, string> = {
  0: 'unspecified',
  1: 'pending',
  2: 'building',
  3: 'running',
  4: 'succeeded',
  5: 'failed',
  6: 'killed',
  7: 'worker_failed',
  8: 'unschedulable',
  9: 'assigned',
  10: 'preempted',
  11: 'cosched_failed',
}

/**
 * Strip the proto enum prefix (JOB_STATE_ or TASK_STATE_) and lowercase.
 * Also handles integer state values from the database.
 */
export function stateToName(protoState: string | number | null | undefined): string {
  if (protoState === null || protoState === undefined) return 'unknown'
  if (protoState === '') return 'unknown'
  if (typeof protoState === 'number') return STATE_INT_MAP[protoState] ?? 'unknown'

  return protoState.replace(/^(JOB_STATE_|TASK_STATE_)/, '').toLowerCase()
}

/** Human-readable display name for a normalized state. */
export function stateDisplayName(state: string): string {
  return STATE_DISPLAY_NAMES[state] ?? state
}

const STATE_DISPLAY_NAMES: Record<string, string> = {
  unspecified: 'Unspecified',
  pending: 'Pending',
  building: 'Building',
  running: 'Running',
  succeeded: 'Succeeded',
  failed: 'Failed',
  killed: 'Killed',
  worker_failed: 'Worker Failed',
  unschedulable: 'Unschedulable',
  assigned: 'Assigned',
  preempted: 'Preempted',
  cosched_failed: 'Cosched Failed',
  unknown: 'Unknown',
}

// -- Tailwind color class mappings --

export interface StatusColorClasses {
  text: string
  bg: string
  border: string
  dot: string
}

const STATUS_COLORS: Record<string, StatusColorClasses> = {
  running: {
    text: 'text-accent',
    bg: 'bg-accent-subtle',
    border: 'border-accent-border',
    dot: 'bg-accent',
  },
  succeeded: {
    text: 'text-status-success',
    bg: 'bg-status-success-bg',
    border: 'border-status-success-border',
    dot: 'bg-status-success',
  },
  failed: {
    text: 'text-status-danger',
    bg: 'bg-status-danger-bg',
    border: 'border-status-danger-border',
    dot: 'bg-status-danger',
  },
  worker_failed: {
    text: 'text-status-danger',
    bg: 'bg-status-danger-bg',
    border: 'border-status-danger-border',
    dot: 'bg-status-danger',
  },
  preempted: {
    text: 'text-status-warning',
    bg: 'bg-status-warning-bg',
    border: 'border-status-warning-border',
    dot: 'bg-status-warning',
  },
  pending: {
    text: 'text-status-warning',
    bg: 'bg-status-warning-bg',
    border: 'border-status-warning-border',
    dot: 'bg-status-warning',
  },
  unschedulable: {
    text: 'text-status-warning',
    bg: 'bg-status-warning-bg',
    border: 'border-status-warning-border',
    dot: 'bg-status-warning',
  },
  building: {
    text: 'text-status-purple',
    bg: 'bg-status-purple-bg',
    border: 'border-status-purple-border',
    dot: 'bg-status-purple',
  },
  assigned: {
    text: 'text-status-orange',
    bg: 'bg-status-orange-bg',
    border: 'border-status-orange-border',
    dot: 'bg-status-orange',
  },
  killed: {
    text: 'text-text-muted',
    bg: 'bg-surface-sunken',
    border: 'border-surface-border',
    dot: 'bg-text-muted',
  },
  cosched_failed: {
    text: 'text-status-danger',
    bg: 'bg-status-danger-bg',
    border: 'border-status-danger-border',
    dot: 'bg-status-danger',
  },
}

const DEFAULT_COLORS: StatusColorClasses = {
  text: 'text-text-muted',
  bg: 'bg-surface-sunken',
  border: 'border-surface-border',
  dot: 'bg-text-muted',
}

/** Get the Tailwind color classes for a given normalized state name. */
export function statusColors(state: string): StatusColorClasses {
  return STATUS_COLORS[state] ?? DEFAULT_COLORS
}

/** Ordered list of job/task states that have explicit color mappings. */
export const STATUS_COLOR_ORDER: readonly string[] = [
  'running',
  'succeeded',
  'failed',
  'worker_failed',
  'cosched_failed',
  'preempted',
  'pending',
  'unschedulable',
  'building',
  'assigned',
  'killed',
] as const

// -- Slice status styling (Autoscaler tab) --

/**
 * Visual style for a slice-status badge (dot + pill colors + label), keyed by the
 * resolved SliceStatus. Shared by the per-group summary, the slice list, and the
 * legend so they stay in sync. Counts over these are always slice-granular.
 */
export interface SliceStatusStyle {
  /** Short label rendered in the badge, e.g. "in use", "free". */
  label: string
  /** One-line explanation for tooltips and the legend. */
  description: string
  /** Tailwind background class for the solid status dot. */
  dot: string
  bg: string
  text: string
  border: string
}

export const SLICE_STATUS_STYLES: Record<SliceStatus, SliceStatusStyle> = {
  available: {
    label: 'free',
    description: 'Ready and healthy with no tasks — free to place work on now',
    dot: 'bg-status-success',
    bg: 'bg-status-success-bg',
    text: 'text-status-success',
    border: 'border-status-success-border',
  },
  in_use: {
    label: 'in use',
    description: 'Ready and healthy with tasks actively running',
    dot: 'bg-accent',
    bg: 'bg-accent-subtle',
    text: 'text-accent',
    border: 'border-accent-border',
  },
  idle: {
    label: 'idle',
    description: 'Ready and healthy, no tasks, idle past the threshold — a scale-down candidate',
    dot: 'bg-status-warning',
    bg: 'bg-status-warning-bg',
    text: 'text-status-warning',
    border: 'border-status-warning-border',
  },
  degraded: {
    label: 'degraded',
    description: 'Missing or unhealthy hosts — cannot be scheduled until workers recover',
    dot: 'bg-status-orange',
    bg: 'bg-status-orange-bg',
    text: 'text-status-orange',
    border: 'border-status-orange-border',
  },
  requesting: {
    label: 'requesting',
    description: 'Scale-up request in flight to the provider',
    dot: 'bg-status-purple',
    bg: 'bg-status-purple-bg',
    text: 'text-status-purple',
    border: 'border-status-purple-border',
  },
  booting: {
    label: 'booting',
    description: 'VMs starting up',
    dot: 'bg-status-purple',
    bg: 'bg-status-purple-bg',
    text: 'text-status-purple',
    border: 'border-status-purple-border',
  },
  initializing: {
    label: 'initializing',
    description: 'Runtime setup in progress',
    dot: 'bg-status-purple',
    bg: 'bg-status-purple-bg',
    text: 'text-status-purple',
    border: 'border-status-purple-border',
  },
  failed: {
    label: 'failed',
    description: 'Slice failed to provision or was lost',
    dot: 'bg-status-danger',
    bg: 'bg-status-danger-bg',
    text: 'text-status-danger',
    border: 'border-status-danger-border',
  },
  // 'ready' is always resolved to a capacity status before display; kept for
  // type completeness with a neutral healthy style.
  ready: {
    label: 'ready',
    description: 'Ready',
    dot: 'bg-status-success',
    bg: 'bg-status-success-bg',
    text: 'text-status-success',
    border: 'border-status-success-border',
  },
}

/**
 * Order for the per-group slice-status summary and the legend: working and free
 * capacity first, then problems, then provisioning.
 */
export const SLICE_STATUS_SUMMARY_ORDER: readonly SliceStatus[] = [
  'in_use',
  'available',
  'idle',
  'degraded',
  'failed',
  'requesting',
  'booting',
  'initializing',
] as const

// -- Progress bar segment colors (Jobs tab) --

/**
 * Tailwind background color for a task-state segment in the Jobs tab progress bar.
 * Kept as a single Tailwind class because only the fill color matters for a stacked
 * bar; text/border colors come from the state-level STATUS_COLORS when rendered
 * as a badge elsewhere.
 */
export const SEGMENT_COLORS: Record<string, string> = {
  succeeded: 'bg-status-success',
  running: 'bg-accent',
  building: 'bg-status-purple',
  assigned: 'bg-status-orange',
  failed: 'bg-status-danger',
  worker_failed: 'bg-status-danger',
  cosched_failed: 'bg-status-danger',
  preempted: 'bg-status-warning',
  killed: 'bg-text-muted',
  pending: 'bg-surface-border',
}

/** Order to render segments in the Jobs tab progress bar. */
export const SEGMENT_ORDER: readonly string[] = [
  'succeeded',
  'running',
  'building',
  'assigned',
  'failed',
  'worker_failed',
  'cosched_failed',
  'preempted',
  'killed',
  'pending',
] as const

// -- Categorical color palette --

/** Shared categorical palette for charts, bars, and legends. */
export const CATEGORICAL_COLORS = [
  '#1877F2', '#F0701A', '#5A24C7', '#E42C97', '#00487C', '#0EAC96',
  '#AB76FF', '#B50550', '#0099E6', '#22085F', '#783301',
] as const

/** Diverging positive-to-negative palette (index 0 = most negative, 10 = most positive).
 *  Use for heatmaps, utilization gauges, or any scale from bad → neutral → good. */
export const DIVERGING_COLORS = [
  '#B50550', '#BF0F76', '#FC7BC6', '#FFADE4', '#FFD9F2',
  '#F0F2F5',
  '#CBF9D7', '#A3E6B5', '#45BD62', '#2A9142', '#1D632E',
] as const

/** Diverging hot-to-cold palette (index 0 = hottest, 10 = coldest).
 *  Use for temperature-style scales from hot (red/orange) → neutral → cold (blue). */
export const DIVERGING_HOT_TO_COLD = [
  '#AA2312', '#D4311C', '#EB630E', '#FFB973', '#FFF2A6',
  '#F0F2F5',
  '#CDE5FF', '#76B6FF', '#1877F2', '#083E89', '#07316D',
] as const

// -- VM state helpers --

/** Strip the VM_STATE_ prefix and lowercase. */
export function vmStateToName(protoState: string | null | undefined): string {
  if (!protoState) return 'unknown'
  return protoState.replace(/^VM_STATE_/, '').toLowerCase()
}

// -- Accelerator type helpers --

/** Convert ACCELERATOR_TYPE_* enum or numeric value to lowercase name. */
export function acceleratorTypeFriendly(accelType: string | number | null | undefined): string {
  if (typeof accelType === 'string') {
    if (accelType.startsWith('ACCELERATOR_TYPE_')) {
      return accelType.replace('ACCELERATOR_TYPE_', '').toLowerCase()
    }
    return accelType.toLowerCase()
  }
  const typeMap: Record<number, string> = { 0: 'unspecified', 1: 'cpu', 2: 'gpu', 3: 'tpu' }
  return typeMap[accelType as number] ?? `unknown(${accelType})`
}

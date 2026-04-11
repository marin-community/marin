/**
 * Status enums, display names, and Tailwind color class mappings for job and task states.
 *
 * Proto enums serialize as strings like "JOB_STATE_RUNNING" or "TASK_STATE_PENDING".
 * The dashboard normalizes these to lowercase names ("running", "pending") via stateToName().
 */

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
  'preempted',
  'pending',
  'unschedulable',
  'building',
  'assigned',
  'killed',
] as const

// -- Slice lifecycle badge styling (Autoscaler tab) --

/**
 * Visual style for a single-letter slice badge rendered in the Autoscaler tab.
 *
 * Letters:
 *   - R = ready and available (fully provisioned, no tasks assigned)
 *   - U = in-use (ready with at least one VM running a task)
 *   - Q = requesting (scale-up in flight)
 *   - B = booting (VMs coming up)
 *   - I = initializing (runtime setup)
 *   - F = failed
 */
export interface SliceBadgeStyle {
  letter: string
  label: string
  bg: string
  text: string
  border: string
}

export const SLICE_STATE_STYLES: Record<string, SliceBadgeStyle> = {
  ready: {
    letter: 'R',
    label: 'Ready and available (no tasks assigned)',
    bg: 'bg-status-success-bg',
    text: 'text-status-success',
    border: 'border-status-success-border',
  },
  in_use: {
    letter: 'U',
    label: 'In use (ready with tasks actively running)',
    bg: 'bg-status-orange-bg',
    text: 'text-status-orange',
    border: 'border-status-orange-border',
  },
  requesting: {
    letter: 'Q',
    label: 'Requesting (scale-up in flight)',
    bg: 'bg-accent-subtle',
    text: 'text-accent',
    border: 'border-accent-border',
  },
  booting: {
    letter: 'B',
    label: 'Booting (VMs starting)',
    bg: 'bg-status-purple-bg',
    text: 'text-status-purple',
    border: 'border-status-purple-border',
  },
  initializing: {
    letter: 'I',
    label: 'Initializing (runtime setup)',
    bg: 'bg-status-warning-bg',
    text: 'text-status-warning',
    border: 'border-status-warning-border',
  },
  failed: {
    letter: 'F',
    label: 'Failed',
    bg: 'bg-status-danger-bg',
    text: 'text-status-danger',
    border: 'border-status-danger-border',
  },
}

/** Order in which slice badges should be rendered in the Autoscaler tab. */
export const SLICE_BADGE_ORDER: readonly string[] = [
  'ready',
  'in_use',
  'requesting',
  'booting',
  'initializing',
  'failed',
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
  'preempted',
  'killed',
  'pending',
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

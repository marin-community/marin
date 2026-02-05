/**
 * Shared utility functions for Iris dashboard components.
 * Extracted from the inline JS in controller/dashboard.py and worker/dashboard.py.
 */

export function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

export function formatDuration(startMs, endMs) {
  if (!startMs) return '-';
  const end = endMs || Date.now();
  const seconds = Math.floor((end - parseInt(startMs)) / 1000);
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

export function formatRelativeTime(timestampMs) {
  if (!timestampMs) return '-';
  const seconds = Math.floor((Date.now() - parseInt(timestampMs)) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export function formatTimestamp(ms) {
  if (!ms) return '-';
  return new Date(ms).toLocaleString();
}

/** Convert proto enum like JOB_STATE_PENDING or TASK_STATE_RUNNING to lowercase name. */
export function stateToName(protoState) {
  // Handle null/undefined (truly missing values)
  if (protoState === null || protoState === undefined) return 'unknown';

  // Handle empty string (edge case)
  if (protoState === '') return 'unknown';

  // Handle numeric enum value 0 (should not happen in JSON, but be defensive)
  if (protoState === 0) return 'unspecified';

  // Handle string enum values - strip prefix and lowercase
  // Example: "JOB_STATE_PENDING" -> "pending", "TASK_STATE_UNSPECIFIED" -> "unspecified"
  const name = protoState.replace(/^(JOB_STATE_|TASK_STATE_)/, '').toLowerCase();
  return name;
}

export function formatAttributeValue(v) {
  if (!v) return '';
  if (v.stringValue !== undefined) return v.stringValue;
  if (v.intValue !== undefined) return String(v.intValue);
  if (v.floatValue !== undefined) return String(v.floatValue);
  if (typeof v === 'string') return v;
  return JSON.stringify(v);
}

export function formatAttributes(attrs) {
  if (!attrs) return '-';
  const entries = Object.entries(attrs)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => k + '=' + formatAttributeValue(v))
    .join(', ');
  return entries || '-';
}

/** Convert accelerator type enum to friendly name (e.g. ACCELERATOR_TYPE_TPU -> tpu). */
export function acceleratorTypeFriendly(accelType) {
  if (typeof accelType === 'string') {
    if (accelType.startsWith('ACCELERATOR_TYPE_')) {
      return accelType.replace('ACCELERATOR_TYPE_', '').toLowerCase();
    }
    return accelType.toLowerCase();
  }
  const typeMap = { 0: 'unspecified', 1: 'cpu', 2: 'gpu', 3: 'tpu' };
  return typeMap[accelType] || `unknown(${accelType})`;
}

export function formatAcceleratorDisplay(accelType, variant) {
  const friendly = acceleratorTypeFriendly(accelType);
  if (variant) return `${friendly} (${variant})`;
  return friendly;
}

export function formatVmState(state) {
  if (!state) return 'unknown';
  return state.replace('VM_STATE_', '').toLowerCase();
}

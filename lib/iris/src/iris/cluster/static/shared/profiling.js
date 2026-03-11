/**
 * Shared profiling utilities for decoding and downloading profile data
 * from the ProfileTask RPC response.
 */

/**
 * Decode base64 profile data, trigger a file download, and clean up the blob URL.
 *
 * @param {string} base64Data - Base64-encoded profile bytes from the RPC response
 * @param {string} target - Target identifier used to generate the filename
 * @param {string} profilerType - Profiler type ('cpu', 'memory', or 'threads')
 * @param {string} format - Profile format (e.g. 'speedscope', 'flamegraph', 'raw', 'table', 'stats')
 */
export function downloadProfile(base64Data, target, profilerType, format) {
  const raw = atob(base64Data);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

  // Determine MIME type and extension based on profiler and format
  let mimeType, extension;
  if (profilerType === 'threads') {
    mimeType = 'text/plain';
    extension = '.txt';
  } else if (profilerType === 'cpu') {
    if (format === 'FLAMEGRAPH') {
      mimeType = 'image/svg+xml';
      extension = '.svg';
    } else if (format === 'SPEEDSCOPE') {
      mimeType = 'application/json';
      extension = '.speedscope.json';
    } else { // RAW
      mimeType = 'text/plain';
      extension = '.txt';
    }
  } else { // memory
    if (format === 'FLAMEGRAPH') {
      mimeType = 'text/html';
      extension = '.html';
    } else if (format === 'TABLE') {
      mimeType = 'text/plain';
      extension = '.txt';
    } else { // STATS
      mimeType = 'application/json';
      extension = '.json';
    }
  }

  const blob = new Blob([bytes], { type: mimeType });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = `profile-${profilerType}-${target.replace(/\//g, '_')}${extension}`;
  a.click();

  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Run a profile RPC call, handle errors, and download the result.
 *
 * @param {Function} rpcFn - The RPC function to call (controllerRpc or workerRpc)
 * @param {string} target - Target to profile: task attempt ID (/job/.../task/0:0) or /system/process
 * @param {Object} options - Profiling options
 * @param {number} [options.durationSeconds=10] - Profiling duration
 * @param {string} [options.profilerType='cpu'] - Profiler type ('cpu', 'memory', or 'threads')
 * @param {string} [options.format='SPEEDSCOPE'] - Output format enum (CPU: FLAMEGRAPH|SPEEDSCOPE|RAW, Memory: FLAMEGRAPH|TABLE|STATS)
 * @param {number} [options.rateHz=100] - Sample rate for CPU profiling
 * @param {boolean} [options.leaks=false] - Enable leak detection for memory profiling
 * @returns {Promise<void>}
 * @throws {Error} on RPC failure or if the response contains an error field
 */
export async function profileAndDownload(
  rpcFn,
  target,
  {
    durationSeconds = 10,
    profilerType = 'cpu',
    format = 'SPEEDSCOPE',
    rateHz = 100,
    leaks = false
  } = {}
) {
  // Build ProfileType based on profiler selection
  let profileType;
  if (profilerType === 'threads') {
    profileType = { threads: {} };
  } else if (profilerType === 'cpu') {
    profileType = { cpu: { format, rateHz } };
  } else {
    profileType = { memory: { format, leaks } };
  }

  const resp = await rpcFn('ProfileTask', { target, durationSeconds, profileType });
  if (resp.error) {
    throw new Error(resp.error);
  }
  downloadProfile(resp.profileData, target, profilerType, format);
}

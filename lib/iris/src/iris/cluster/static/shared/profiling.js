/**
 * Shared profiling utilities for decoding and downloading profile data
 * from the ProfileTask RPC response.
 */

/**
 * Decode base64 profile data, trigger a file download, and clean up the blob URL.
 *
 * @param {string} base64Data - Base64-encoded profile bytes from the RPC response
 * @param {string} taskId - Task ID used to generate the filename
 * @param {string} format - Profile format (e.g. 'speedscope', 'flamegraph', 'raw')
 */
export function downloadProfile(base64Data, taskId, format = 'speedscope') {
  const raw = atob(base64Data);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

  const mimeType = format === 'flamegraph' ? 'image/svg+xml' : 'application/json';
  const extension = format === 'flamegraph' ? '.svg' : '.speedscope.json';

  const blob = new Blob([bytes], { type: mimeType });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = 'profile-' + taskId.replace(/\//g, '_') + extension;
  a.click();

  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Run a profile RPC call, handle errors, and download the result.
 *
 * @param {Function} rpcFn - The RPC function to call (controllerRpc or workerRpc)
 * @param {string} taskId - Task ID to profile
 * @param {Object} options - Profiling options
 * @param {number} [options.durationSeconds=10] - Profiling duration
 * @param {string} [options.format='speedscope'] - Output format
 * @returns {Promise<void>}
 * @throws {Error} on RPC failure or if the response contains an error field
 */
export async function profileAndDownload(rpcFn, taskId, { durationSeconds = 10, format = 'speedscope' } = {}) {
  const resp = await rpcFn('ProfileTask', { taskId, durationSeconds, format });
  if (resp.error) {
    throw new Error(resp.error);
  }
  downloadProfile(resp.profileData, taskId, format);
}

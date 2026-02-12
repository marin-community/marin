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

/**
 * Download memory profile data from a MemoryProfile RPC response.
 *
 * @param {string} base64Data - Base64-encoded profile bytes from the RPC response
 * @param {string} taskId - Task ID used to generate the filename
 * @param {string} format - Profile format (e.g. 'flamegraph', 'table', 'stats')
 */
export function downloadMemoryProfile(base64Data, taskId, format = 'flamegraph') {
  const raw = atob(base64Data);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

  const extMap = { flamegraph: '.html', table: '.txt', stats: '.json' };
  const mimeMap = { flamegraph: 'text/html', table: 'text/plain', stats: 'application/json' };

  const extension = extMap[format] || '.html';
  const mimeType = mimeMap[format] || 'text/html';

  const blob = new Blob([bytes], { type: mimeType });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = 'memory-profile-' + taskId.replace(/\//g, '_') + extension;
  a.click();

  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * View memory profile HTML directly in a modal window.
 *
 * @param {string} base64Data - Base64-encoded HTML from the RPC response
 * @param {string} taskId - Task ID for the modal title
 */
export function viewMemoryProfileHTML(base64Data, taskId) {
  const raw = atob(base64Data);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

  const blob = new Blob([bytes], { type: 'text/html' });
  const url = URL.createObjectURL(blob);

  // Create modal overlay
  const modal = document.createElement('div');
  modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8);z-index:10000;display:flex;align-items:center;justify-content:center;';

  const container = document.createElement('div');
  container.style.cssText = 'background:white;width:95%;height:95%;display:flex;flex-direction:column;border-radius:8px;overflow:hidden;';

  // Header with title and close button
  const header = document.createElement('div');
  header.style.cssText = 'padding:12px 16px;background:#f5f5f5;border-bottom:1px solid #ddd;display:flex;justify-content:space-between;align-items:center;';
  header.innerHTML = `<strong>Memory Profile: ${taskId}</strong>`;

  const closeBtn = document.createElement('button');
  closeBtn.textContent = '✕ Close';
  closeBtn.style.cssText = 'padding:6px 12px;cursor:pointer;background:#fff;border:1px solid #ccc;border-radius:4px;';
  closeBtn.onclick = () => {
    URL.revokeObjectURL(url);
    document.body.removeChild(modal);
  };
  header.appendChild(closeBtn);

  // Iframe for the HTML report
  const iframe = document.createElement('iframe');
  iframe.src = url;
  iframe.style.cssText = 'flex:1;border:none;';

  container.appendChild(header);
  container.appendChild(iframe);
  modal.appendChild(container);
  document.body.appendChild(modal);
}

/**
 * Run a memory profile RPC call, handle errors, and show/download the result.
 *
 * @param {Function} rpcFn - The RPC function to call (controllerRpc or workerRpc)
 * @param {string} taskId - Task ID to profile
 * @param {Object} options - Profiling options
 * @param {number} [options.durationSeconds=10] - Profiling duration
 * @param {boolean} [options.leaks=false] - Only track memory leaks
 * @param {string} [options.format='flamegraph'] - Output format
 * @param {boolean} [options.view=true] - View HTML inline (if format is flamegraph)
 * @returns {Promise<void>}
 * @throws {Error} on RPC failure or if the response contains an error field
 */
export async function memoryProfileAndView(rpcFn, taskId, { durationSeconds = 10, leaks = false, format = 'flamegraph', view = true } = {}) {
  const resp = await rpcFn('MemoryProfile', { taskId, durationSeconds, leaks, format });
  if (resp.error) {
    throw new Error(resp.error);
  }

  // For HTML flamegraph, offer to view inline
  if (format === 'flamegraph' && view) {
    viewMemoryProfileHTML(resp.profileData, taskId);
  } else {
    downloadMemoryProfile(resp.profileData, taskId, format);
  }
}

/**
 * Generic Connect RPC caller for a given service.
 */
async function connectRpc(service, method, body = {}) {
  const response = await fetch(`/${service}/${method}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    throw new Error(`RPC ${method} failed: ${response.status}`);
  }
  return response.json();
}

/**
 * RPC helper for calling Connect RPC endpoints on the controller.
 */
export function controllerRpc(method, body = {}) {
  return connectRpc('iris.cluster.ControllerService', method, body);
}

/**
 * RPC helper for calling Connect RPC endpoints on the worker.
 */
export function workerRpc(method, body = {}) {
  return connectRpc('iris.cluster.WorkerService', method, body);
}

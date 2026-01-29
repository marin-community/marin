/**
 * RPC helper for calling Connect RPC endpoints on the controller.
 */
export async function controllerRpc(method, body = {}) {
  const response = await fetch(`/iris.cluster.ControllerService/${method}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    throw new Error(`RPC ${method} failed: ${response.status}`);
  }
  return response.json();
}

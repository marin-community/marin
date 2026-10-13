async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: options.body ? { "Content-Type": "application/json", ...options.headers } : options.headers,
  });
  if (response.status === 204) return null;
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`;
    const error = new Error(detail);
    Object.assign(error, { status: response.status });
    throw error;
  }
  return body;
}

export const chartApi = {
  list: () => request("/plantt/api/charts"),
  get: (id) => request(`/plantt/api/charts/${id}`),
  create: (document) => request("/plantt/api/charts", { method: "POST", body: JSON.stringify({ document }) }),
  update: (id, document, revision) =>
    request(`/plantt/api/charts/${id}`, { method: "PUT", body: JSON.stringify({ document, revision }) }),
  remove: (id, revision) =>
    request(`/plantt/api/charts/${id}`, { method: "DELETE", body: JSON.stringify({ revision }) }),
};

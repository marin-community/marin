// Relative backend calls. A leading slash would escape the applet prefix.
import { ref } from "vue";

export async function getJson(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  const text = await response.text();
  if (!response.ok) {
    let message = text.slice(0, 500);
    try {
      const body = JSON.parse(text);
      if (body && body.detail) message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // Keep the raw text.
    }
    throw new Error(`${response.status}: ${message}`);
  }
  return text ? JSON.parse(text) : null;
}

/** Reactive loader: {data, loading, error, refresh}. `pathFn` returns null to skip a fetch. */
export function useJson(pathFn) {
  const data = ref(null);
  const loading = ref(false);
  const error = ref(null);
  let generation = 0;
  async function refresh() {
    const path = pathFn();
    if (!path) return;
    const mine = ++generation;
    loading.value = true;
    try {
      const result = await getJson(path);
      if (mine === generation) {
        data.value = result;
        error.value = null;
      }
    } catch (e) {
      if (mine === generation) error.value = e.message || String(e);
    } finally {
      if (mine === generation) loading.value = false;
    }
  }
  function reset() {
    generation++;
    data.value = null;
    error.value = null;
    loading.value = false;
  }
  return { data, loading, error, refresh, reset };
}

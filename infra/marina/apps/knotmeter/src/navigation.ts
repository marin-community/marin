import { useSyncExternalStore } from "react";

const subscribe = (notify: () => void) => {
  window.addEventListener("hashchange", notify);
  return () => window.removeEventListener("hashchange", notify);
};
const snapshot = () => window.location.hash.slice(1) || "/chart";

export function useRoute() {
  const route = useSyncExternalStore(subscribe, snapshot);
  const [path, query = ""] = route.split("?");
  return { path, params: new URLSearchParams(query) };
}

export function setParameters(values: Record<string, string>) {
  const [path, query = ""] = snapshot().split("?");
  const params = new URLSearchParams(query);
  for (const [key, value] of Object.entries(values)) {
    if (value) params.set(key, value);
    else params.delete(key);
  }
  window.history.replaceState(null, "", `#${path}?${params}`);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}

export function navigate(path: string) {
  window.location.hash = path;
}

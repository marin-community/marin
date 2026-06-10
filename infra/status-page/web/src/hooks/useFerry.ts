import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import { fetchFerry } from "../api";
import { autoRefreshAtom } from "../state";

// Frontend polls slightly less often than the backend TTL (60s) — the
// backend cache is the authoritative shield for GitHub's rate limit.
const REFETCH_INTERVAL_MS = 60_000;

export function useFerry() {
  const autoRefresh = useAtomValue(autoRefreshAtom);
  return useQuery({
    queryKey: ["ferry"],
    queryFn: fetchFerry,
    refetchInterval: autoRefresh ? REFETCH_INTERVAL_MS : false,
    staleTime: 30_000,
  });
}

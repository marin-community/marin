import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import { fetchBuilds } from "../api";
import { autoRefreshAtom } from "../state";

// Same cadence as ferry — the backend TTL (60s) is the real shield
// against the GitHub rate limit; the frontend just needs to keep pace.
const REFETCH_INTERVAL_MS = 60_000;

export function useBuilds() {
  const autoRefresh = useAtomValue(autoRefreshAtom);
  return useQuery({
    queryKey: ["builds"],
    queryFn: fetchBuilds,
    refetchInterval: autoRefresh ? REFETCH_INTERVAL_MS : false,
    staleTime: 30_000,
  });
}

import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import { fetchWorkersHistory } from "../api";
import { autoRefreshAtom } from "../state";

// The canary writes a new worker_healthy point every 60s and the server
// shields finelog behind a 60s TTL, so anything faster is wasted traffic.
const REFETCH_INTERVAL_MS = 60_000;

export function useWorkersHistory() {
  const autoRefresh = useAtomValue(autoRefreshAtom);
  return useQuery({
    queryKey: ["workers", "history"],
    queryFn: fetchWorkersHistory,
    refetchInterval: autoRefresh ? REFETCH_INTERVAL_MS : false,
    staleTime: 15_000,
  });
}

import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import { fetchWorkersHistory } from "../api";
import { autoRefreshAtom } from "../state";

// Refetches on the same cadence as the server sampler — any faster is
// wasted traffic since the ring buffer only gains a new point every 30s.
const REFETCH_INTERVAL_MS = 30_000;

export function useWorkersHistory() {
  const autoRefresh = useAtomValue(autoRefreshAtom);
  return useQuery({
    queryKey: ["workers", "history"],
    queryFn: fetchWorkersHistory,
    refetchInterval: autoRefresh ? REFETCH_INTERVAL_MS : false,
    staleTime: 15_000,
  });
}

import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import { fetchWorkers } from "../api";
import { autoRefreshAtom } from "../state";

// Matches the server sampler cadence (30s) so the UI ticks roughly in
// lockstep with the ring buffer.
const REFETCH_INTERVAL_MS = 30_000;

export function useWorkers() {
  const autoRefresh = useAtomValue(autoRefreshAtom);
  return useQuery({
    queryKey: ["workers"],
    queryFn: fetchWorkers,
    refetchInterval: autoRefresh ? REFETCH_INTERVAL_MS : false,
    staleTime: 10_000,
  });
}

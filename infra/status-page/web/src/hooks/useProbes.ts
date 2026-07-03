import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import { fetchProbes } from "../api";
import { autoRefreshAtom } from "../state";

// Probe metrics move slowly (health checks ≤5min, provisioning 15min) and the
// backend shields finelog behind a 60s TTL; poll on the same cadence.
const REFETCH_INTERVAL_MS = 60_000;

export function useProbes() {
  const autoRefresh = useAtomValue(autoRefreshAtom);
  return useQuery({
    queryKey: ["probes"],
    queryFn: fetchProbes,
    refetchInterval: autoRefresh ? REFETCH_INTERVAL_MS : false,
    staleTime: 30_000,
  });
}

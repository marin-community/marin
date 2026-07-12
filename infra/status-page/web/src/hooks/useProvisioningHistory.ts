import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import { fetchProvisioningHistory } from "../api";
import { autoRefreshAtom } from "../state";

// Provisioning gauges turn over every 15min and the server shields finelog
// behind a 60s TTL, so a 60s refetch is plenty.
const REFETCH_INTERVAL_MS = 60_000;

export function useProvisioningHistory() {
  const autoRefresh = useAtomValue(autoRefreshAtom);
  return useQuery({
    queryKey: ["provisioning", "history"],
    queryFn: fetchProvisioningHistory,
    refetchInterval: autoRefresh ? REFETCH_INTERVAL_MS : false,
    staleTime: 30_000,
  });
}

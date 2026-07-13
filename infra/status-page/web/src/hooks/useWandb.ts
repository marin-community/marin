import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import { fetchWandb } from "../api";
import { autoRefreshAtom } from "../state";

// Training metrics move at eval cadence (hours) and the server shields
// W&B behind a 5min TTL, so anything faster is wasted traffic.
const REFETCH_INTERVAL_MS = 300_000;

export function useWandb() {
  const autoRefresh = useAtomValue(autoRefreshAtom);
  return useQuery({
    queryKey: ["wandb"],
    queryFn: fetchWandb,
    refetchInterval: autoRefresh ? REFETCH_INTERVAL_MS : false,
    staleTime: 60_000,
  });
}

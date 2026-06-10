import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import { fetchControlPlaneHealth } from "../api";
import { autoRefreshAtom } from "../state";

// Matches the backend sampler cadence.
const REFETCH_INTERVAL_MS = 30_000;

export function useControlPlaneHealth() {
  const autoRefresh = useAtomValue(autoRefreshAtom);
  return useQuery({
    queryKey: ["control-plane", "health"],
    queryFn: fetchControlPlaneHealth,
    refetchInterval: autoRefresh ? REFETCH_INTERVAL_MS : false,
    staleTime: 15_000,
  });
}

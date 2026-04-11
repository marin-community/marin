import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import { fetchOrch } from "../api";
import { autoRefreshAtom } from "../state";

const REFETCH_INTERVAL_MS = 15_000;

export function useOrch() {
  const autoRefresh = useAtomValue(autoRefreshAtom);
  return useQuery({
    queryKey: ["orch"],
    queryFn: fetchOrch,
    refetchInterval: autoRefresh ? REFETCH_INTERVAL_MS : false,
    staleTime: 5_000,
  });
}

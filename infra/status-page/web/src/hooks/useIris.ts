import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import { fetchIris } from "../api";
import { autoRefreshAtom } from "../state";

const REFETCH_INTERVAL_MS = 15_000;

export function useIris() {
  const autoRefresh = useAtomValue(autoRefreshAtom);
  return useQuery({
    queryKey: ["iris"],
    queryFn: fetchIris,
    refetchInterval: autoRefresh ? REFETCH_INTERVAL_MS : false,
    staleTime: 5_000,
  });
}

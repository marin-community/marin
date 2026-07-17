import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import { fetchNightlies } from "../api";
import { autoRefreshAtom } from "../state";

const REFETCH_INTERVAL_MS = 60_000;

export function useNightlies() {
  const autoRefresh = useAtomValue(autoRefreshAtom);
  return useQuery({
    queryKey: ["nightlies"],
    queryFn: fetchNightlies,
    refetchInterval: autoRefresh ? REFETCH_INTERVAL_MS : false,
    staleTime: 30_000,
  });
}

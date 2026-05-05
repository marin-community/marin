import { useQuery } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import { fetchJobs } from "../api";
import { autoRefreshAtom } from "../state";

// Jobs counts don't need to be as fresh as workers — the 24h window
// moves slowly. Backend TTL is 60s; poll on the same cadence.
const REFETCH_INTERVAL_MS = 60_000;

export function useJobs() {
  const autoRefresh = useAtomValue(autoRefreshAtom);
  return useQuery({
    queryKey: ["jobs"],
    queryFn: fetchJobs,
    refetchInterval: autoRefresh ? REFETCH_INTERVAL_MS : false,
    staleTime: 30_000,
  });
}

// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
import { useCallback, useSyncExternalStore } from "react";

const NOOP = () => {};

export const MOBILE_VIEWPORT_QUERY = "(max-width: 767px)";

export const COARSE_POINTER_QUERY = "(hover: none) and (pointer: coarse)";

function canMatch(): boolean {
  return (
    typeof window !== "undefined" && typeof window.matchMedia === "function"
  );
}

export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (onStoreChange: () => void) => {
      if (!canMatch()) return NOOP;
      const mediaQuery = window.matchMedia(query);
      mediaQuery.addEventListener("change", onStoreChange);
      return () => mediaQuery.removeEventListener("change", onStoreChange);
    },
    [query],
  );
  const getSnapshot = useCallback(
    () => canMatch() && window.matchMedia(query).matches,
    [query],
  );
  return useSyncExternalStore(subscribe, getSnapshot, () => false);
}

export function useIsMobileViewport(): boolean {
  return useMediaQuery(MOBILE_VIEWPORT_QUERY);
}

export function useIsCoarsePointer(): boolean {
  return useMediaQuery(COARSE_POINTER_QUERY);
}

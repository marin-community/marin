// Jotai atoms for UI state.
//
// Network state (ferry/orch data) lives in react-query. Jotai handles the
// small UI knobs: auto-refresh toggle, whether orch raw JSON is expanded,
// etc. This split keeps each library doing what it's best at.

import { atom } from "jotai";

export const autoRefreshAtom = atom<boolean>(true);
export const orchRawExpandedAtom = atom<boolean>(false);

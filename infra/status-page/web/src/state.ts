// Jotai atoms for UI state.
//
// Network state (ferry/iris/workers/jobs/builds data) lives in
// react-query. Jotai handles the small UI knobs like the auto-refresh
// toggle. This split keeps each library doing what it's best at.

import { atom } from "jotai";

export const autoRefreshAtom = atom<boolean>(true);

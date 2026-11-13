// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
const TOUCH_COMPAT_WINDOW_MS = 1000;

export const TOUCH_ONLY_QUERY = "(hover: none) and (pointer: coarse)";

let lastTouchAt = Number.NEGATIVE_INFINITY;
let installed = false;

const now = () =>
  typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now();

const markTouch = () => {
  lastTouchAt = now();
};

const onPointerDown = (event: PointerEvent) => {
  if (event.pointerType === "touch" || event.pointerType === "pen") markTouch();
};

const onPointerMove = (event: PointerEvent) => {
  if (event.pointerType === "mouse") lastTouchAt = Number.NEGATIVE_INFINITY;
};

export function installTouchInputTracking(): void {
  if (installed || typeof document === "undefined") return;
  installed = true;
  const opts: AddEventListenerOptions = { capture: true, passive: true };
  document.addEventListener("touchstart", markTouch, opts);
  document.addEventListener("pointerdown", onPointerDown, opts);
  document.addEventListener("pointermove", onPointerMove, opts);
}

interface SourceCapabilitiesLike {
  firesTouchEvents?: boolean;
}

export function isTouchCompatMouseEvent(event?: Event | null): boolean {
  installTouchInputTracking();
  const capabilities = (
    event as { sourceCapabilities?: SourceCapabilitiesLike } | null
  )?.sourceCapabilities;
  if (capabilities?.firesTouchEvents) return true;
  if (now() - lastTouchAt < TOUCH_COMPAT_WINDOW_MS) return true;
  if (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function"
  ) {
    return window.matchMedia(TOUCH_ONLY_QUERY).matches;
  }
  return false;
}

export function resetTouchInputTrackingForTests(): void {
  lastTouchAt = Number.NEGATIVE_INFINITY;
}

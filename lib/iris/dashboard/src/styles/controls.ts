/**
 * The class vocabulary for the log panel's controls: two sizes, and one on/off
 * treatment shared by every toggle in it.
 *
 * A toggle binds the state string alongside the base string rather than folding
 * the resting colours into the base. Tailwind resolves competing utilities
 * (`border-*`, `bg-*`, `text-*`) by their order in the generated stylesheet, not
 * by their order in a class list, so a base colour and an active colour listed
 * together would win unpredictably.
 */

/** Toolbar-height control: the query row's inputs, selects, and buttons. */
export const CONTROL = 'inline-flex h-7 items-center gap-1.5 rounded border px-2 text-sm disabled:opacity-40'

/** Inline control: the view bar's toggles and nav arrows, one step smaller. */
export const CONTROL_SM = 'inline-flex h-6 items-center gap-1 rounded border px-1.5 text-xs disabled:opacity-40'

/** Engaged toggle, or the open end of a menu. */
export const CONTROL_ON = 'border-accent-border bg-accent-subtle text-accent'

/** Resting toggle or plain button. */
export const CONTROL_OFF = 'border-surface-border bg-surface text-text-secondary hover:enabled:bg-surface-sunken'

/**
 * Native selects, which cannot carry the flex layout above. The explicit
 * background and text colour keep the closed control on the dashboard palette
 * in dark mode; `color-scheme` in main.css covers the dropdown the browser
 * renders on top of it.
 */
export const SELECT = 'h-7 rounded border border-surface-border bg-surface px-1.5 text-sm text-text'

/** Free-text query fields. Callers set the height, so it matches the row they sit in. */
export const FIELD = 'rounded border border-surface-border bg-surface px-2 font-mono '
  + 'placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent'

// Time display and entry for the app: US Eastern on screen, UTC on hover.

const EASTERN_ZONE = "America/New_York";
const UTC_ZONE = "UTC";

const easternPartsFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: EASTERN_ZONE,
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
  timeZoneName: "short",
});

const easternClockFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: EASTERN_ZONE,
  hour: "numeric",
  minute: "2-digit",
  timeZoneName: "short",
});

const easternFieldFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: EASTERN_ZONE,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

const utcFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: UTC_ZONE,
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
  timeZoneName: "short",
});

export function fmtEastern(ms) {
  return easternPartsFormatter.format(new Date(ms));
}

export function fmtEasternClock(ms) {
  return easternClockFormatter.format(new Date(ms));
}

export function fmtUtc(ms) {
  return utcFormatter.format(new Date(ms));
}

/** Eastern wall-clock parts of an instant, as named strings. */
function easternFields(ms) {
  const parts = {};
  for (const part of easternFieldFormatter.formatToParts(new Date(ms))) {
    if (part.type !== "literal") parts[part.type] = part.value;
  }
  return parts;
}

/** The `datetime-local` value a person in Eastern would type for this instant. */
export function pickerValue(ms) {
  const fields = easternFields(ms);
  return `${fields.year}-${fields.month}-${fields.day}T${fields.hour}:${fields.minute}`;
}

/** Offset of Eastern from UTC, in milliseconds, at the given instant. */
function easternOffsetMs(instantMs) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: EASTERN_ZONE,
    timeZoneName: "longOffset",
  }).formatToParts(new Date(instantMs));
  const gmt = parts.find((part) => part.type === "timeZoneName")?.value ?? "GMT+00:00";
  const match = gmt.match(/GMT([+-])(\d{2}):(\d{2})/);
  if (!match) return 0;
  const sign = match[1] === "-" ? -1 : 1;
  return sign * (Number(match[2]) * 60 + Number(match[3])) * 60_000;
}

/** Epoch ms for a `datetime-local` value the person means as Eastern wall time. */
export function pickerToMs(value) {
  if (!value) return null;
  // Parse the wall time as if it were UTC, then shift by the Eastern offset —
  // twice, so a value on a DST boundary lands with the offset of the instant itself.
  const asUtc = Date.parse(`${value}:00Z`) ?? Date.parse(`${value}Z`);
  if (Number.isNaN(asUtc)) return null;
  const firstPass = asUtc - easternOffsetMs(asUtc);
  return asUtc - easternOffsetMs(firstPass);
}

export function floorToStep(ms, stepMs) {
  return Math.floor(ms / stepMs) * stepMs;
}

export function clamp(ms, lo, hi) {
  return Math.min(Math.max(ms, lo), hi);
}

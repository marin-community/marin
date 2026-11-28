// Stable per-user colors. The palette lives in styles.css as `.u0 … .u11` with
// light and dark values; assignment hashes the user id so a user keeps its
// color as others come and go.

export const PALETTE_SIZE = 12;

function hash(text) {
  let value = 2166136261;
  for (const character of text) {
    value ^= character.codePointAt(0);
    value = Math.imul(value, 16777619);
  }
  return value >>> 0;
}

/** user_id → palette index, collision-free within the active set. */
export function assignColors(userIds) {
  const used = new Set();
  const colors = {};
  for (const id of [...userIds].sort()) {
    let index = id ? hash(id) % PALETTE_SIZE : 0;
    while (used.has(index)) index = (index + 1) % PALETTE_SIZE;
    used.add(index);
    colors[id] = index;
  }
  return colors;
}

export function colorClass(userId, colors) {
  const index = colors[userId ?? ""];
  return Number.isInteger(index) ? `u${index}` : "u0";
}

const KIBIBYTE = 1024;
const BYTE_UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];

export function formatBytes(bytes: number): string {
  let value = bytes;
  let unit = 0;
  while (value >= KIBIBYTE && unit < BYTE_UNITS.length - 1) {
    value /= KIBIBYTE;
    unit += 1;
  }
  return `${value >= 100 ? Math.round(value) : value.toFixed(1)} ${BYTE_UNITS[unit]}`;
}

const BYTE_UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']

export function humanBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B'
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), BYTE_UNITS.length - 1)
  const val = bytes / Math.pow(1024, i)
  return (val >= 100 ? Math.round(val) : val.toFixed(1)) + ' ' + BYTE_UNITS[i]
}

export function humanCost(usd) {
  return '$' + usd.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

export function humanCount(n) {
  return n.toLocaleString('en-US')
}

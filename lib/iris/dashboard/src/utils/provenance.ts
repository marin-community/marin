import type { Provenance } from '@/types/rpc'

/**
 * Render build provenance as a single line, mirroring the Python
 * ``rigging.provenance.Provenance.__str__``:
 *   clean: "<baseCommit> (<branch>) (<builtBy>)"
 *   dirty: "<treeHash> (off of <baseCommit>) (<branch>) (<builtBy>)"
 */
export function formatProvenance(p?: Provenance): string {
  if (!p) return '-'
  let suffix = ''
  if (p.branch) suffix += ` (${p.branch})`
  if (p.builtBy) suffix += ` (${p.builtBy})`
  if (p.dirty) return `${p.treeHash ?? ''} (off of ${p.baseCommit ?? ''})${suffix}`
  return `${p.baseCommit || p.treeHash || '-'}${suffix}`
}

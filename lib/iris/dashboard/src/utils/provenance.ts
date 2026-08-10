import type { Provenance } from '@/types/rpc'

/**
 * Render build provenance as a single line:
 *   clean: "<baseCommit> (<branch>) (<builtBy>)"
 *   dirty: "<treeHash> (off of <baseCommit>) (<branch>) (<builtBy>)"
 * A clean build shows the recognizable commit; a dirty build shows the
 * content tree hash plus the commit it was built off of.
 */
export function formatProvenance(p?: Provenance): string {
  if (!p) return '-'
  let suffix = ''
  if (p.branch) suffix += ` (${p.branch})`
  if (p.builtBy) suffix += ` (${p.builtBy})`
  if (p.dirty) return `${p.treeHash ?? ''} (off of ${p.baseCommit ?? ''})${suffix}`
  return `${p.baseCommit || p.treeHash || '-'}${suffix}`
}

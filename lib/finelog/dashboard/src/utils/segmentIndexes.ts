import type { SegmentInfo } from '@/types/introspection'
import type { ProtoSchema } from '@/types/stats'

export interface IndexCoverage {
  id: string
  indexed: number
  eligible: number
}

export interface SegmentIndexSummary {
  stableIndexed: number
  stableEligible: number
  l0Unindexed: number
  methods: IndexCoverage[]
  adaptiveMethods: IndexCoverage[]
  countColumns: IndexCoverage[]
  bytes: number
}

/** Planner-facing section IDs required by the registered namespace policy. */
export function configuredIndexMethods(schema: ProtoSchema | null): string[] {
  if (!schema) return []
  const ids: string[] = []
  let hasExactPostings = false
  let hasValueCounts = false
  for (const column of schema.columns ?? []) {
    if (column.index?.trigram) ids.push(`trigram:${column.name}`)
    hasExactPostings ||= Boolean(column.index?.exactValues?.length)
    hasValueCounts ||= Boolean(column.index?.valueCounts)
  }
  if (hasExactPostings || schema.projections?.length) ids.push('exact-postings')
  if (hasValueCounts) ids.push('value-counts')
  for (const projection of schema.projections ?? []) ids.push(`projection:${projection.name}`)
  return ids.sort()
}

function configuredAdaptiveMethods(schema: ProtoSchema): string[] {
  return (
    schema.groupedExtrema?.map(
      (config) =>
        `group-extrema:${config.filterColumn}:${config.groupJsonColumn}:${config.groupJsonKey}:${config.extremaColumn}`,
    ) ?? []
  ).sort()
}

/** Coverage among local segments whose physical artifacts were inspected. */
export function segmentIndexSummary(
  segments: SegmentInfo[],
  schema: ProtoSchema | null,
): SegmentIndexSummary | null {
  if (!schema) return null
  const methods = configuredIndexMethods(schema)
  if (!methods.length && !schema.columns?.some((column) => column.type === 'COLUMN_TYPE_STRING')) {
    return null
  }
  const inspected = segments.filter((segment) => segment.physical)
  const stable = inspected.filter((segment) => segment.level >= 1)
  const l0 = inspected.filter((segment) => segment.level === 0)
  const sectionIds = (segment: SegmentInfo) =>
    new Set(
      segment.physical?.indexBundle?.sections
        .filter((section) => section.available)
        .map((section) => section.id) ?? [],
    )
  const sections = (segment: SegmentInfo) =>
    segment.physical?.indexBundle?.sections.filter((section) => section.available) ?? []
  const adaptiveMethods = [
    ...new Set(
      [
        ...configuredAdaptiveMethods(schema),
        ...stable.flatMap((segment) =>
          sections(segment)
            .filter((section) => section.kind === 'group_extrema')
            .map((section) => section.id),
        ),
      ],
    ),
  ].sort()
  const requiredCountColumns =
    schema?.columns?.filter((column) => column.index?.valueCounts).map((column) => column.name) ?? []
  const requiredPostingColumns = new Set(
    schema?.columns
      ?.filter((column) => column.index?.exactValues?.length)
      .map((column) => column.name) ?? [],
  )
  for (const projection of schema?.projections ?? []) {
    if (projection.predicateColumn) requiredPostingColumns.add(projection.predicateColumn)
  }
  const covers = (segment: SegmentInfo, method: string): boolean => {
    const available = sections(segment)
    if (method === 'value-counts') {
      const columns = new Set(
        available.find((section) => section.id === method)?.columns ?? [],
      )
      return requiredCountColumns.every((column) => columns.has(column))
    }
    if (method === 'exact-postings') {
      const columns = new Set(
        available.find((section) => section.id === method)?.columns ?? [],
      )
      return [...requiredPostingColumns].every((column) => columns.has(column))
    }
    return sectionIds(segment).has(method)
  }
  const stableIndexed = stable.filter((segment) =>
    methods.every((method) => covers(segment, method)),
  ).length
  const countColumnNames = [
    ...new Set(
      stable.flatMap((segment) =>
        sections(segment)
          .filter((section) => section.id === 'value-counts')
          .flatMap((section) => section.columns),
      ),
    ),
  ].sort()
  return {
    stableIndexed,
    stableEligible: stable.length,
    l0Unindexed: l0.length,
    methods: methods.map((id) => {
      return {
        id,
        indexed: stable.filter((segment) => covers(segment, id)).length,
        eligible: stable.length,
      }
    }),
    adaptiveMethods: adaptiveMethods.map((id) => ({
      id,
      indexed: stable.filter((segment) => covers(segment, id)).length,
      eligible: stable.length,
    })),
    countColumns: countColumnNames.map((column) => ({
      id: column,
      indexed: stable.filter((segment) =>
        sections(segment).some(
          (section) => section.id === 'value-counts' && section.columns.includes(column),
        ),
      ).length,
      eligible: stable.length,
    })),
    bytes: inspected.reduce(
      (total, segment) =>
        total +
        (segment.physical?.indexBundle?.bytes ?? 0) +
        (segment.physical?.indexBundle?.externalBytes ?? 0),
      0,
    ),
  }
}

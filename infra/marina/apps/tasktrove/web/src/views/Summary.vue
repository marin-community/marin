<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import {
  converterPolicies,
  count,
  manifest,
  shortRef,
  sourceDetails,
  type Counts,
  type Manifest,
} from '../corpus'

type SourceRow = {
  name: string
  family: string
  verdict: 'keep' | 'drop'
  input: number
  kept: number
  statuses: [string, number][]
  converters: string[]
  graders: string[]
  environments: string[]
  transformation: string
  reason: string
}

const loaded = ref<Manifest>()
const problem = ref('')
const sourceQuery = ref('')
const sourceVerdict = ref<'all' | 'keep' | 'drop'>('all')

onMounted(async () => {
  try {
    loaded.value = await manifest()
  } catch (error) {
    problem.value = String(error)
  }
})

const retained = computed(() => {
  if (!loaded.value) return 0
  return Math.round((loaded.value.clean_tasks / loaded.value.input_tasks) * 100)
})
const modes = computed(() => Object.entries(loaded.value?.by_mode ?? {}).sort((a, b) => b[1] - a[1]))
const environments = computed(() => Object.values(loaded.value?.dockerfiles ?? {}))
const keptSources = computed(
  () => Object.values(loaded.value?.source_verdicts ?? {}).filter((source) => source.verdict === 'keep').length,
)
const rejectedSources = computed(
  () => Object.values(loaded.value?.source_verdicts ?? {}).filter((source) => source.verdict === 'drop').length,
)
const droppedAtSource = computed(() => loaded.value?.by_status.dropped_source ?? 0)
const rowRejected = computed(() =>
  loaded.value ? loaded.value.input_tasks - loaded.value.clean_tasks - droppedAtSource.value : 0,
)
const rejected = computed(() => (loaded.value ? loaded.value.input_tasks - loaded.value.clean_tasks : 0))

function nonzero(values: Counts | undefined): string[] {
  return Object.entries(values ?? {}).filter(([, value]) => value > 0).map(([name]) => name)
}

const sourceRows = computed<SourceRow[]>(() => {
  const dataset = loaded.value
  if (!dataset) return []
  const details = sourceDetails(dataset)
  return Object.entries(dataset.source_verdicts).map(([name, verdict]) => {
    const statuses = Object.entries(dataset.by_source[name] ?? {}).sort((a, b) => b[1] - a[1])
    const converters = nonzero(details[name]?.converters)
    const graders = nonzero(details[name]?.modes)
    const sourceEnvironments = [...new Set(
      Object.values(dataset.dockerfiles)
        .filter((environment) => (environment.sources[name] ?? 0) > 0)
        .map((environment) => environment.base_image),
    )].sort()
    return {
      name,
      family: verdict.family,
      verdict: verdict.verdict,
      input: statuses.reduce((sum, [, tasks]) => sum + tasks, 0),
      kept: dataset.by_source[name]?.converted ?? 0,
      statuses: statuses.filter(([status]) => status !== 'converted'),
      converters,
      graders,
      environments: sourceEnvironments,
      transformation: converters.map((converter) => converterPolicies[converter]?.transformation).filter(Boolean).join(' '),
      reason: verdict.reason,
    }
  }).sort((a, b) => b.kept - a.kept || b.input - a.input || a.name.localeCompare(b.name))
})

const filteredSources = computed(() => {
  const query = sourceQuery.value.trim().toLocaleLowerCase()
  return sourceRows.value.filter((source) => {
    if (sourceVerdict.value !== 'all' && source.verdict !== sourceVerdict.value) return false
    if (!query) return true
    return [source.name, source.family, source.reason, ...source.converters, ...source.graders]
      .some((value) => value.toLocaleLowerCase().includes(query))
  })
})

function statusSummary(statuses: [string, number][]): string {
  return statuses.map(([status, tasks]) => `${status} ${count(tasks)}`).join(' · ')
}
</script>

<template>
  <p class="problem" v-if="problem">{{ problem }}</p>
  <template v-if="loaded">
    <section class="hero">
      <p class="eyebrow">TaskTrove Clean</p>
      <h1>A task collection with explicit, testable rewards</h1>
      <p class="lede">
        We started with {{ count(loaded.input_tasks) }} heterogeneous agent tasks and kept
        {{ count(loaded.clean_tasks) }} whose instructions, environments, and graders could be made sound
        without guessing the intended answer.
      </p>
      <div class="hero-actions">
        <RouterLink class="button primary" to="/browse">Browse the clean Parquet</RouterLink>
        <a class="button" href="https://github.com/marin-community/marin/pull/9061">Open the implementation PR</a>
      </div>
    </section>

    <section class="metrics" aria-label="Dataset summary">
      <div><strong>{{ count(loaded.clean_tasks) }}</strong><span>clean tasks</span></div>
      <div><strong>{{ retained }}%</strong><span>of input retained</span></div>
      <div><strong>{{ keptSources }}</strong><span>kept sources</span></div>
      <div><strong>{{ modes.length }}</strong><span>verifier modes</span></div>
      <div><strong>{{ environments.length }}</strong><span>Docker environments</span></div>
    </section>

    <section class="paper">
      <div class="section-heading">
        <p class="eyebrow">Method</p>
        <h2>Cleanup pipeline</h2>
        <p>The branches are the actual release decisions: every input row reaches either the clean Parquet or the rejection ledger.</p>
      </div>
      <div class="dag-wrap">
        <svg class="dag" viewBox="0 0 1120 430" role="img" aria-labelledby="dag-title dag-description">
          <title id="dag-title">TaskTrove cleanup directed acyclic graph</title>
          <desc id="dag-description">Raw TaskTrove tasks are inventoried, reviewed by source, normalized, checked per row, and written to one clean Parquet file or a rejection ledger.</desc>
          <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" /></marker></defs>
          <g class="dag-edges">
            <path d="M190 105 H245" /><path d="M415 105 H470" /><path d="M640 105 H695" /><path d="M865 105 H920" />
            <path d="M330 145 V270" /><path d="M395 310 H500" /><path d="M780 145 V310 H725" /><path d="M1005 145 V270" />
          </g>
          <g class="dag-node">
            <rect x="20" y="65" width="170" height="80" rx="8" /><text x="40" y="94"><tspan>Raw TaskTrove</tspan><tspan x="40" dy="24" class="dag-count">{{ count(loaded.input_tasks) }} rows</tspan></text>
            <rect x="245" y="65" width="170" height="80" rx="8" /><text x="265" y="94"><tspan>Inventory</tspan><tspan x="265" dy="24" class="dag-note">templates + graders</tspan></text>
            <rect x="470" y="65" width="170" height="80" rx="8" /><text x="490" y="94"><tspan>Normalize</tspan><tspan x="490" dy="24" class="dag-note">{{ Object.keys(loaded.by_converter).length }} converters</tspan></text>
            <rect x="695" y="65" width="170" height="80" rx="8" /><text x="715" y="94"><tspan>Row checks</tspan><tspan x="715" dy="24" class="dag-note">shape · gold · empty</tspan></text>
            <rect x="920" y="65" width="170" height="80" rx="8" /><text x="940" y="94"><tspan>Final reshard</tspan><tspan x="940" dy="24" class="dag-note">one Parquet file</tspan></text>
          </g>
          <g class="dag-node accepted">
            <rect x="920" y="270" width="170" height="80" rx="8" /><text x="940" y="299"><tspan>Clean tasks</tspan><tspan x="940" dy="24" class="dag-count">{{ count(loaded.clean_tasks) }}</tspan></text>
          </g>
          <g class="dag-node rejected-node">
            <rect x="225" y="270" width="170" height="80" rx="8" /><text x="245" y="299"><tspan>Dropped sources</tspan><tspan x="245" dy="24" class="dag-count">{{ count(droppedAtSource) }} rows</tspan></text>
            <rect x="500" y="270" width="225" height="80" rx="8" /><text x="520" y="299"><tspan>Rejection ledger</tspan><tspan x="520" dy="24" class="dag-count">{{ count(rejected) }} rows total</tspan></text>
          </g>
          <text x="785" y="247" class="dag-annotation">{{ count(rowRejected) }} row-level rejects</text>
          <text x="20" y="407" class="dag-footnote">Source policy runs before conversion. Exact instruction dedup and verifier challenges run per row. Every decision is counted in manifest.json.</text>
        </svg>
      </div>
    </section>

    <section class="paper two-column">
      <div>
        <div class="section-heading">
          <p class="eyebrow">Decision rule</p>
          <h2>Best effort, with a hard quality floor</h2>
        </div>
        <p>
          Small deterministic repairs are allowed: trim stray whitespace, preserve a grader’s explicit polarity,
          install its declared runtime, or route it through the general script verifier. We do not infer answers with
          content-specific heuristics, rebuild missing repositories, generate replacement tests, or keep a grader that
          accepts an invalid solution.
        </p>
      </div>
      <div>
        <div class="section-heading">
          <p class="eyebrow">Published release</p>
          <h2>One file, plus its audit trail</h2>
        </div>
        <p>
          Survivors are in <code>tasks/part-00000.parquet</code>. <code>ledger.parquet</code> records every rejected row;
          <code>manifest.json</code> records source policy and aggregate counts. The Marina server reads the release from
          the existing <code>marin-us-east-02a</code> S3 bucket with its configured credentials.
        </p>
      </div>
    </section>

    <section id="sources" class="paper source-audit">
      <div class="section-heading">
        <p class="eyebrow">Source audit</p>
        <h2>What happened to every source</h2>
        <p>Counts and decisions come from the release manifest. Converter and grader names describe the normalized tasks that survived.</p>
      </div>
      <div class="source-controls">
        <label><span>Find a source</span><input v-model="sourceQuery" type="search" placeholder="name, family, converter, grader, or reason" /></label>
        <label><span>Disposition</span><select v-model="sourceVerdict"><option value="all">All sources</option><option value="keep">Kept</option><option value="drop">Dropped</option></select></label>
        <span>{{ count(filteredSources.length) }} sources</span>
      </div>
      <div class="data-table-wrap">
        <table class="data-table source-table">
          <thead><tr><th>Source</th><th>Disposition</th><th>Input</th><th>Kept</th><th>Transformation</th><th>Grader</th><th>Decision and row rejects</th></tr></thead>
          <tbody>
            <tr v-for="source in filteredSources" :key="source.name">
              <td><b>{{ source.name }}</b><small>{{ source.family }}</small></td>
              <td><span class="verdict" :data-verdict="source.verdict">{{ source.verdict === 'keep' ? 'Kept' : 'Dropped' }}</span></td>
              <td class="number">{{ count(source.input) }}</td>
              <td class="number">
                <RouterLink v-if="source.kept" :to="{ path: '/browse', query: { source: source.name } }">{{ count(source.kept) }}</RouterLink>
                <span v-else>0</span>
              </td>
              <td>
                <template v-if="source.verdict === 'keep'">
                  <span>{{ source.transformation || 'Normalize into the common task archive.' }}</span>
                  <small>{{ source.converters.join(', ') }}<template v-if="source.environments.length"> · {{ source.environments.join(', ') }}</template></small>
                </template>
                <span v-else>Not converted.</span>
              </td>
              <td><div class="grader-list" v-if="source.graders.length"><span v-for="grader in source.graders" :key="grader" class="mode">{{ grader }}</span></div><span v-else>—</span></td>
              <td><span>{{ source.reason }}</span><small v-if="source.statuses.length">Row outcomes: {{ statusSummary(source.statuses) }}</small></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="paper">
      <div class="section-heading">
        <p class="eyebrow">What changed</p>
        <h2>Cleanup milestones</h2>
      </div>
      <div class="milestones">
        <article><span>01</span><h3>Structured outputs</h3><p>Recovered sound TOML, XML, and CSV graders; rejected malformed or unconstrained rows.</p></article>
        <article><span>02</span><h3>Malformed answers</h3><p>Left ambiguous MCQA and broken ARC rows out instead of adding answer-extraction heuristics.</p></article>
        <article><span>03</span><h3>Test tasks</h3><p>Reinstated self-contained Python katas and removed sources with trivial, missing, or environment-bound tests.</p></article>
        <article><span>04</span><h3>Repository SWE</h3><p>Kept sound non-Python repositories through the script fallback; dropped JavaScript and TypeScript after weak golden results.</p></article>
        <article><span>05</span><h3>Judge rubrics</h3><p>Reviewed kept and deferred judge sources and restored the polarity of 151 negated multichallenge checks.</p></article>
        <article><span>06</span><h3>Reproducibility</h3><p>Every output records the source revision, verifier commit, converter, mode, environment, tags, and rejection reason.</p></article>
      </div>
    </section>

    <section class="paper distribution">
      <div class="section-heading"><p class="eyebrow">Final composition</p><h2>Tasks by verifier mode</h2></div>
      <div class="bars">
        <div v-for="[mode, tasks] in modes" :key="mode" class="bar-row">
          <span class="mono">{{ mode }}</span><div><i :style="{ width: `${Math.max(1, (tasks / modes[0][1]) * 100)}%` }" /></div><b>{{ count(tasks) }}</b>
        </div>
      </div>
      <p class="provenance">
        Source <code>{{ loaded.tasktrove.hf_id }}</code> @ <code>{{ shortRef(loaded.tasktrove.revision) }}</code> · verifier
        <code>{{ shortRef(loaded.verify_tool_ref) }}</code>
      </p>
    </section>
  </template>
  <p class="working" v-else-if="!problem">Reading the clean dataset manifest…</p>
</template>

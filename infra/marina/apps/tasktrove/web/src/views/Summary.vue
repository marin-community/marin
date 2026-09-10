<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { count, manifest, shortRef, type Manifest } from '../corpus'

const loaded = ref<Manifest>()
const problem = ref('')

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
        <h2>From archive to training-ready task</h2>
      </div>
      <ol class="pipeline">
        <li><b>Inventory</b><span>Fingerprint every source and grader template.</span></li>
        <li><b>Decide</b><span>Keep answerable sources; record a reason for every dropped source.</span></li>
        <li><b>Normalize</b><span>Map each surviving task onto one declared verifier mode or the script fallback.</span></li>
        <li><b>Challenge</b><span>Reject leaked golds, empty-pass graders, malformed specs, and invalid perturbations.</span></li>
        <li><b>Package</b><span>Emit clean task shards, a rejection ledger, and this reproducible manifest.</span></li>
      </ol>
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
          <p class="eyebrow">Source review</p>
          <h2>{{ keptSources }} kept, {{ rejectedSources }} dropped</h2>
        </div>
        <p>
          Source-level decisions are backed by template inspection and deterministic samples. Per-row verification then
          catches null graders, answer leakage, empty-output passes, malformed task shapes, and broken expected answers.
          Easy but correctly graded katas remain available with a selection tag.
        </p>
      </div>
    </section>

    <section class="paper">
      <div class="section-heading">
        <p class="eyebrow">What changed</p>
        <h2>Cleanup milestones</h2>
      </div>
      <div class="milestones">
        <article>
          <span>01</span><h3>Structured outputs</h3>
          <p>Recovered sound TOML, XML, and CSV graders; rejected schemas that accept unconstrained documents.</p>
        </article>
        <article>
          <span>02</span><h3>Malformed answers</h3>
          <p>Left ambiguous MCQA and broken ARC rows out instead of adding answer-extraction heuristics.</p>
        </article>
        <article>
          <span>03</span><h3>Test tasks</h3>
          <p>Reinstated self-contained Python katas and removed sources with trivial, missing, or environment-bound tests.</p>
        </article>
        <article>
          <span>04</span><h3>Repository SWE</h3>
          <p>Kept sound non-Python repositories through the script fallback; dropped JavaScript and TypeScript after weak golden results.</p>
        </article>
        <article>
          <span>05</span><h3>Judge rubrics</h3>
          <p>Reviewed kept and deferred judge sources and restored the polarity of 151 negated multichallenge checks.</p>
        </article>
        <article>
          <span>06</span><h3>Reproducibility</h3>
          <p>Every output records the source revision, verifier commit, converter, mode, environment, tags, and rejection reason.</p>
        </article>
      </div>
    </section>

    <section class="paper distribution">
      <div class="section-heading">
        <p class="eyebrow">Final composition</p>
        <h2>Tasks by verifier mode</h2>
      </div>
      <div class="bars">
        <div v-for="[mode, tasks] in modes" :key="mode" class="bar-row">
          <span class="mono">{{ mode }}</span>
          <div><i :style="{ width: `${Math.max(1, (tasks / modes[0][1]) * 100)}%` }" /></div>
          <b>{{ count(tasks) }}</b>
        </div>
      </div>
      <p class="provenance">
        Source <code>{{ loaded.tasktrove.hf_id }}</code> @
        <code>{{ shortRef(loaded.tasktrove.revision) }}</code> · verifier
        <code>{{ shortRef(loaded.verify_tool_ref) }}</code>
      </p>
    </section>
  </template>
  <p class="working" v-else-if="!problem">Reading the clean dataset manifest…</p>
</template>

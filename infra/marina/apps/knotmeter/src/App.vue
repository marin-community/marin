<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import AppBar from "@marina/AppBar.vue";
import CorpusView from "./CorpusView.vue";
import MetricPlot from "./MetricPlot.vue";
import Timeline from "./Timeline.vue";
import {
  api,
  modelName,
  sourceUrl,
  type Corpus,
  type RequestRecord,
  type Run,
} from "./types";

const runs = ref<Run[]>([]);
const model = ref("");
const hardware = ref("");
const date = ref("");
const percentile = ref("p90");
const catalogError = ref("");
const error = ref("");
const loading = ref(true);
const run = ref<Run | null>(null);
const requests = ref<RequestRecord[]>([]);
const corpus = ref<Corpus | null>(null);
const hash = ref(location.hash || "#/overview");
function syncHash() {
  hash.value = location.hash || "#/overview";
}
window.addEventListener("hashchange", syncHash);
onBeforeUnmount(() => window.removeEventListener("hashchange", syncHash));
const route = computed(() => {
  const [path, query] = hash.value.replace(/^#\/?/, "").split("?");
  const [view, key] = path!.split("/");
  return {
    view: view || "overview",
    key: decodeURIComponent(key || ""),
    query: new URLSearchParams(query),
  };
});
const models = computed(() =>
  [...new Set(runs.value.map((r) => r.model))].sort(),
);
const modelRuns = computed(() =>
  runs.value.filter((r) => r.model === model.value),
);
const hardwareOptions = computed(() =>
  [...new Set(modelRuns.value.map((r) => r.hardware))].sort(),
);
const dates = computed(() =>
  [...new Set(modelRuns.value.map((r) => r.date))].sort().reverse(),
);
const filtered = computed(() =>
  modelRuns.value.filter(
    (r) =>
      (!hardware.value || r.hardware === hardware.value) &&
      (!date.value || r.date === date.value),
  ),
);
const onlyOneDate = computed(
  () => new Set(filtered.value.map((r) => r.date)).size < 2,
);
watch(model, () => {
  hardware.value = "";
  date.value = "";
});
let requestVersion = 0;
watch(
  route,
  async (current) => {
    const version = ++requestVersion;
    loading.value = true;
    error.value = "";
    run.value = null;
    corpus.value = null;
    requests.value = [];
    try {
      if (current.view === "run") {
        const [detail, timeline] = await Promise.all([
          api<Run>(`runs/${encodeURIComponent(current.key)}`),
          api<{ requests: RequestRecord[] }>(
            `runs/${encodeURIComponent(current.key)}/requests`,
          ),
        ]);
        if (version === requestVersion) {
          run.value = detail;
          requests.value = timeline.requests;
        }
      } else if (current.view === "corpus") {
        const detail = await api<Run>(
          `runs/${encodeURIComponent(current.query.get("run") || "")}`,
        );
        const params = new URLSearchParams({
          dataset: detail.dataset,
          revision: detail.dataset_revision,
        });
        const data = await api<Corpus>(
          `corpus/${encodeURIComponent(current.key)}?${params}`,
        );
        if (version === requestVersion) {
          run.value = detail;
          corpus.value = data;
        }
      }
    } catch (e) {
      if (version === requestVersion) error.value = String(e);
    } finally {
      if (version === requestVersion) loading.value = false;
    }
  },
  { immediate: true },
);
function corpusUrl(request: RequestRecord) {
  return `#/corpus/${encodeURIComponent(request.corpus_id)}?${new URLSearchParams({ run: run.value!.key, outer: String(request.source_outer_idx) })}`;
}
function openCorpus(request: RequestRecord) {
  location.hash = corpusUrl(request);
}
const output = (r: Run) =>
  r.metrics.throughput.per_gpu.output_tput_tps.toFixed(5);
const latency = (r: Run, metric: string, scale = 1) =>
  (r.metrics.latency[metric]![percentile.value]! * scale).toFixed(3);
function provisional(r: Run): boolean {
  return r.source.run_id === 40705471793;
}
onMounted(async () => {
  try {
    runs.value = await api<Run[]>("runs");
    model.value =
      models.value.find((m) => m.includes("Snowball")) || models.value[0] || "";
  } catch (e) {
    catalogError.value = String(e);
  }
});
</script>

<template>
  <AppBar title="Knotmeter" home="https://marina.oa.dev/">
    <template #nav
      ><a
        href="#/overview"
        :aria-current="route.view === 'overview' ? 'page' : undefined"
        >Measurements</a
      ><a
        href="#/history"
        :aria-current="route.view === 'history' ? 'page' : undefined"
        >Dated history</a
      ><a
        v-if="run"
        :href="`#/run/${run.key}`"
        :aria-current="route.view === 'run' ? 'page' : undefined"
        >Run details</a
      ></template
    >
    <template #actions><span class="chip">Saved results</span></template>
  </AppBar>
  <main>
    <div class="page-heading">
      <div>
        <p class="eyebrow">Inference performance</p>
        <h1>
          {{
            route.view === "history"
              ? "Dated history"
              : route.view === "corpus"
                ? "Corpus conversation"
                : route.view === "run"
                  ? "Run details"
                  : "Measurements"
          }}
        </h1>
        <p class="muted">AgentX · BF16 · Output tokens/s per physical GPU</p>
      </div>
      <span class="chip">{{ runs.length }} imported runs</span>
    </div>
    <p v-if="catalogError" class="notice" role="alert">{{ catalogError }}</p>
    <p v-if="error" class="notice" role="alert">{{ error }}</p>
    <p v-if="loading" role="status">Loading…</p>
    <template v-if="['overview', 'history'].includes(route.view)">
      <div class="filters catalog-filters">
        <label
          >Model<select v-model="model" aria-label="Model">
            <option v-for="m in models" :key="m" :value="m">
              {{ m.split("/").at(-1) }}
            </option>
          </select></label
        >
        <label
          >Hardware<select v-model="hardware" aria-label="Hardware">
            <option value="">All measured hardware</option>
            <option v-for="h in hardwareOptions" :key="h">{{ h }}</option>
          </select></label
        >
        <label
          >Source date (UTC)<select
            v-model="date"
            aria-label="Source date (UTC)"
          >
            <option value="">All dates</option>
            <option v-for="d in dates" :key="d">{{ d }}</option>
          </select></label
        >
        <label
          >Latency percentile<select
            v-model="percentile"
            aria-label="Latency percentile"
          >
            <option>p50</option>
            <option>p75</option>
            <option>p90</option>
            <option>p95</option>
          </select></label
        >
      </div>
      <p v-if="!runs.length && !catalogError" class="empty">
        No runs imported. Load a saved AgentX collector bundle into this Marina
        preview.
      </p>
      <template v-else>
        <p class="notice subtle">
          All measured loads are shown.
          {{
            hardwareOptions.length < 2
              ? "This model has one measured hardware setup; a chip comparison needs the same model on another chip."
              : "Compare matching workload and software settings before attributing differences to hardware."
          }}
        </p>
        <p v-if="filtered.some(provisional)" class="notice">
          Hero is the saved October 28 provisional run: prefix caching disabled,
          max batched tokens 512, max sequences 64. Its optimized setup has not
          been measured here.
        </p>
        <div v-if="route.view === 'overview'" class="chart-grid">
          <section class="panel">
            <h2>Throughput and load</h2>
            <MetricPlot
              :runs="filtered"
              axis="concurrency"
              :percentile="percentile"
            />
            <p class="muted">
              C is the configured concurrency. Each dot opens its run; no
              measured loads are hidden.
            </p>
          </section>
          <section class="panel">
            <h2>Throughput and latency</h2>
            <MetricPlot :runs="filtered" axis="ttft" :percentile="percentile" />
            <p class="muted">
              {{ percentile }} first-token latency from the upstream profiling
              aggregate.
            </p>
          </section>
        </div>
        <section v-else class="panel">
          <h2>Measured output throughput by date</h2>
          <MetricPlot :runs="filtered" axis="date" :percentile="percentile" />
          <p class="muted">
            {{
              onlyOneDate
                ? "One date is available. There is no measured trend yet."
                : "Points retain their measured loads and recipes. A change in settings may explain a change in throughput."
            }}
          </p>
        </section>
        <section class="panel">
          <div class="section-heading">
            <h2>Run inventory</h2>
            <span>{{ filtered.length }} matching runs · AgentX / BF16</span>
          </div>
          <div class="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Model / source</th>
                  <th>Date (UTC)</th>
                  <th>Hardware</th>
                  <th>Load</th>
                  <th>Output tokens/s/GPU</th>
                  <th>{{ percentile }} TTFT (s)</th>
                  <th>{{ percentile }} ITL (ms)</th>
                  <th>Profiling requests</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="r in filtered" :key="r.key">
                  <td>
                    <a :href="`#/run/${r.key}`">{{ modelName(r) }}</a
                    ><small
                      >{{ r.source.run_id }} / attempt
                      {{ r.source.run_attempt }}</small
                    ><small v-if="provisional(r)" class="warn"
                      >Provisional saved run</small
                    >
                  </td>
                  <td>{{ r.date }}</td>
                  <td>
                    {{ r.physical_gpus }} × {{ r.hardware
                    }}<small
                      >{{ r.hosts }}
                      {{ r.hosts === 1 ? "host" : "hosts" }}</small
                    >
                  </td>
                  <td>C{{ r.concurrency }}</td>
                  <td class="num" :data-output="r.key">{{ output(r) }}</td>
                  <td class="num">{{ latency(r, "ttft") }}</td>
                  <td class="num">{{ latency(r, "itl", 1000) }}</td>
                  <td class="num">{{ r.profiling_requests }}</td>
                </tr>
              </tbody>
            </table>
            <p v-if="!filtered.length" class="empty">
              No runs match these filters.
            </p>
          </div>
        </section>
      </template>
    </template>
    <template v-else-if="route.view === 'run' && run">
      <div class="run-heading">
        <h2>{{ modelName(run) }}</h2>
        <span
          >{{ run.physical_gpus }} × {{ run.hardware }} · {{ run.hosts }} hosts
          · C{{ run.concurrency }} · {{ run.date }} UTC</span
        >
      </div>
      <p v-if="provisional(run)" class="notice">
        Provisional October 28 Hero run. Prefix caching was disabled; max
        batched tokens 512 and max sequences 64. Its optimized setup has not
        been measured here.
      </p>
      <div class="stat-grid">
        <div>
          <span>Output tokens/s/GPU</span
          ><strong data-testid="run-output">{{ output(run) }}</strong>
        </div>
        <div>
          <span>p90 first token</span
          ><strong>{{ run.metrics.latency.ttft!.p90!.toFixed(3) }} s</strong>
        </div>
        <div>
          <span>p90 inter-token latency</span
          ><strong
            >{{ (run.metrics.latency.itl!.p90! * 1000).toFixed(3) }} ms</strong
          >
        </div>
        <div>
          <span>Successful profiling / all records</span
          ><strong
            >{{ run.profiling_requests }} / {{ run.request_count }}</strong
          >
        </div>
      </div>
      <Timeline :key="run.key" :requests="requests" @corpus="openCorpus" />
      <section class="panel">
        <h2>Runtime and provenance</h2>
        <dl class="provenance">
          <dt>Original attempt</dt>
          <dd>
            <a :href="sourceUrl(run)" target="_blank" rel="noreferrer"
              >{{ run.source.run_id }} / {{ run.source.run_attempt }} ↗</a
            >
            · {{ run.source_created_at }}
          </dd>
          <dt>Marin source</dt>
          <dd>
            <a
              :href="`https://github.com/${run.source.repository}/commit/${run.source.head_sha}`"
              target="_blank"
              rel="noreferrer"
              ><code>{{ run.source.head_sha }}</code></a
            >
          </dd>
          <dt>Model identity</dt>
          <dd>
            <code>{{ run.model }} @ {{ run.model_revision }}</code>
          </dd>
          <dt>Runtime</dt>
          <dd>
            <code>{{ run.runtime }}</code>
          </dd>
          <dt>Recipe fingerprint</dt>
          <dd>
            <code>{{ run.recipe }}</code>
          </dd>
          <dt>Corpus</dt>
          <dd>
            {{ run.dataset }} @ <code>{{ run.dataset_revision }}</code> ·
            {{ run.dataset_sessions }} conversations
          </dd>
          <dt>Window</dt>
          <dd>
            {{ run.metrics.throughput.duration_seconds }} seconds, upstream
            profiling window
          </dd>
        </dl>
        <details>
          <summary>Original settings, validation and artifact hashes</summary>
          <pre>{{
            JSON.stringify(
              {
                aggregate: run.aggregate,
                configuration: run.configuration,
                manifest: run.manifest,
                validation: run.validation,
                files: run.source_files,
                import: run.content_sha256,
              },
              null,
              2,
            )
          }}</pre>
        </details>
      </section>
    </template>
    <template v-else-if="route.view === 'corpus' && corpus && run"
      ><a :href="`#/run/${run.key}`">← Back to measured run</a
      ><CorpusView :corpus="corpus" :outer="route.query.get('outer') || ''"
    /></template>
    <p v-else-if="!loading && !error" class="empty">
      This view is unavailable. <a href="#/overview">Open measurements</a>.
    </p>
  </main>
</template>

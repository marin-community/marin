<script setup lang="ts">
import { computed, ref } from "vue";
import type { RequestRecord } from "./types";

const props = defineProps<{ requests: RequestRecord[] }>();
const emit = defineEmits<{ corpus: [request: RequestRecord] }>();
const phase = ref("profiling");
const conversation = ref("");
const selected = ref<RequestRecord | null>(null);
const conversations = computed(() =>
  [...new Set(props.requests.map((r) => r.conversation_id))].sort(),
);
const visible = computed(() =>
  props.requests.filter(
    (r) =>
      (!phase.value || r.phase === phase.value) &&
      (!conversation.value || r.conversation_id === conversation.value),
  ),
);
const lanes = computed(() =>
  [...new Set(visible.value.map((r) => r.conversation_id))].sort(),
);
const bounds = computed(() => {
  if (!visible.value.length) return { start: 0, span: 1 };
  const start = Math.min(...visible.value.map((r) => r.start_ns));
  return {
    start,
    span: Math.max(1, Math.max(...visible.value.map((r) => r.end_ns)) - start),
  };
});
const x = (ns: number) =>
  170 + (800 * (ns - bounds.value.start)) / bounds.value.span;
const y = (r: RequestRecord) =>
  28 + lanes.value.indexOf(r.conversation_id) * 22;
const first = (r: RequestRecord) =>
  Math.min(r.end_ns, r.start_ns + (r.ttft_ms ?? 0) * 1e6);
const seconds = (ns: number) => (ns / 1e9).toFixed(3);
</script>

<template>
  <section class="panel">
    <div class="section-heading">
      <h2>Request timeline</h2>
      <span>{{ visible.length }} / {{ requests.length }} requests</span>
    </div>
    <div class="filters">
      <label
        >Phase<select v-model="phase" aria-label="Phase">
          <option value="profiling">Profiling</option>
          <option value="warmup">Warmup</option>
          <option value="">All phases</option>
        </select></label
      >
      <label
        >Conversation<select v-model="conversation" aria-label="Conversation">
          <option value="">All conversations</option>
          <option v-for="id in conversations" :key="id">{{ id }}</option>
        </select></label
      >
    </div>
    <p class="muted">
      Each row is a replay conversation. Time is relative to the first saved
      request; warmup and errors remain available. Select a request to inspect
      its source.
    </p>
    <p v-if="!visible.length" class="empty">No requests in this selection.</p>
    <div v-else class="timeline-scroll">
      <svg
        class="timeline"
        role="img"
        aria-label="Request timeline"
        :style="{ height: `${Math.max(200, lanes.length * 22 + 60)}px` }"
        :viewBox="`0 0 1000 ${Math.max(200, lanes.length * 22 + 60)}`"
      >
        <g v-for="tick in [0, 0.25, 0.5, 0.75, 1]" :key="tick">
          <line
            :x1="170 + 800 * tick"
            :x2="170 + 800 * tick"
            y1="20"
            :y2="lanes.length * 22 + 40"
            class="grid"
          />
          <text :x="170 + 800 * tick" y="13" text-anchor="middle">
            {{ seconds(bounds.start + bounds.span * tick) }}s
          </text>
        </g>
        <text
          v-for="(id, index) in lanes"
          :key="id"
          x="160"
          :y="40 + index * 22"
          text-anchor="end"
        >
          {{ id.slice(0, 12)
          }}{{ id.includes("::") ? id.slice(id.indexOf("::")) : "" }}
        </text>
        <g
          v-for="r in visible"
          :key="r.id"
          :data-start-ns="r.start_ns"
          :data-end-ns="r.end_ns"
          tabindex="0"
          role="button"
          :aria-label="`Turn ${r.turn}, ${r.conversation_id}, ${r.phase}${r.error ? ', error' : ''}`"
          @click="selected = r"
          @keydown.enter="selected = r"
        >
          <title>
            {{ r.conversation_id }} · turn {{ r.turn }} ·
            {{ seconds(r.end_ns - r.start_ns) }}s · {{ r.error || r.phase }}
          </title>
          <rect
            :x="x(r.start_ns)"
            :y="y(r)"
            :width="Math.max(2, x(r.end_ns) - x(r.start_ns))"
            height="15"
            :fill="r.error ? '#b1473f' : '#258568'"
          />
          <rect
            v-if="!r.error && r.ttft_ms !== null"
            :x="x(r.start_ns)"
            :y="y(r)"
            :width="Math.max(1, x(first(r)) - x(r.start_ns))"
            height="15"
            fill="#d99b38"
          />
        </g>
      </svg>
    </div>
    <p class="legend">
      <span class="waiting">●</span> Waiting for first token
      <span class="decode">●</span> Remaining response
      <span class="failed">●</span> Error
    </p>
    <div v-if="selected" class="request-detail" aria-live="polite">
      <h3>Turn {{ selected.turn }} · {{ selected.phase }}</h3>
      <p>
        <code>{{ selected.conversation_id }}</code>
      </p>
      <dl class="facts">
        <div>
          <dt>Start / end</dt>
          <dd>
            {{ seconds(selected.start_ns) }} / {{ seconds(selected.end_ns) }}s
          </dd>
        </div>
        <div>
          <dt>First token</dt>
          <dd>
            {{
              selected.ttft_ms === null
                ? "Unavailable"
                : `${selected.ttft_ms.toFixed(3)} ms`
            }}
          </dd>
        </div>
        <div>
          <dt>Input / output</dt>
          <dd>
            {{ selected.input_tokens ?? "—" }} /
            {{ selected.output_tokens ?? "—" }} tokens
          </dd>
        </div>
        <div>
          <dt>Status</dt>
          <dd>{{ selected.error || "Completed" }}</dd>
        </div>
      </dl>
      <button @click="emit('corpus', selected)">
        Open corpus conversation →
      </button>
    </div>
  </section>
</template>

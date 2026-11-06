<script setup lang="ts">
import { computed, onMounted, nextTick } from "vue";
import type { Corpus, CorpusTurn } from "./types";

const props = defineProps<{ corpus: Corpus; outer: string }>();
const rows = computed(() =>
  props.corpus.structure.nodes.flatMap<CorpusTurn & { parent: string }>(
    (node) =>
      node.kind === "subagent"
        ? (node.children ?? []).map((child) => ({
            ...child,
            parent: node.label ?? "Subagent",
            rawIndex: node.rawIndex,
          }))
        : [{ ...node, parent: "" }],
  ),
);
onMounted(async () => {
  await nextTick();
  document
    .querySelector('[data-selected="true"]')
    ?.scrollIntoView({ block: "center" });
});
</script>
<template>
  <section class="panel">
    <div class="section-heading">
      <h2>Original corpus conversation</h2>
      <span>{{ rows.length }} requests</span>
    </div>
    <p>
      <code>{{ corpus.conversation_id }}</code>
    </p>
    <p class="muted">
      {{ corpus.dataset }}<br />Revision <code>{{ corpus.revision }}</code> ·
      <a
        :href="`https://huggingface.co/datasets/${corpus.dataset}/tree/${corpus.revision}`"
        target="_blank"
        rel="noreferrer"
        >Dataset source ↗</a
      >
    </p>
    <p class="muted">
      Original token and timing structure, including subagent calls. These are
      corpus timings, separate from the measured replay above.
    </p>
    <div class="table-scroll corpus-table">
      <table>
        <thead>
          <tr>
            <th>Source index</th>
            <th>Turn / group</th>
            <th>Original model</th>
            <th>Start (s)</th>
            <th>End (s)</th>
            <th>Input tokens</th>
            <th>Output tokens</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(row, i) in rows"
            :key="i"
            :data-selected="String(row.rawIndex) === outer"
          >
            <td>{{ row.rawIndex }}</td>
            <td>
              {{ row.parent ? `${row.parent} / ` : "" }}{{ row.turnIndex }}
            </td>
            <td>{{ row.model || "—" }}</td>
            <td>{{ row.startS?.toFixed(3) ?? "—" }}</td>
            <td>{{ row.endS?.toFixed(3) ?? "—" }}</td>
            <td>{{ row.in }}</td>
            <td>{{ row.out }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({ total: Number, page: Number, pageSize: Number });
const emit = defineEmits(["page"]);
const pages = computed(() => Math.max(1, Math.ceil(props.total / props.pageSize)));
const first = computed(() => (props.total ? (props.page - 1) * props.pageSize + 1 : 0));
const last = computed(() => Math.min(props.total, props.page * props.pageSize));
</script>

<template>
  <div class="pager">
    <span>{{ first.toLocaleString() }}–{{ last.toLocaleString() }} of {{ total.toLocaleString() }}</span>
    <select :value="pageSize" @change="emit('page', 1, Number($event.target.value))">
      <option v-for="n in [25, 50, 100, 200]" :key="n" :value="n">{{ n }} per page</option>
    </select>
    <button :disabled="page <= 1" @click="emit('page', 1, pageSize)">First</button>
    <button :disabled="page <= 1" @click="emit('page', page - 1, pageSize)">Previous</button>
    <span>Page {{ page.toLocaleString() }} of {{ pages.toLocaleString() }}</span>
    <button :disabled="page >= pages" @click="emit('page', page + 1, pageSize)">Next</button>
    <button :disabled="page >= pages" @click="emit('page', pages, pageSize)">Last</button>
  </div>
</template>

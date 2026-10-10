<script setup>
import { ref, onMounted, onBeforeUnmount, watch } from "vue";
import { getDocument, GlobalWorkerOptions } from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
GlobalWorkerOptions.workerSrc = workerUrl;
const props = defineProps({ url: String });
const canvas = ref(null),
  page = ref(1),
  pages = ref(0),
  error = ref(""),
  loading = ref(true);
let document, task, renderTask;
async function render() {
  if (!document || !canvas.value) return;
  renderTask?.cancel();
  const current = await document.getPage(page.value);
  const viewport = current.getViewport({ scale: 1.3 });
  canvas.value.width = viewport.width;
  canvas.value.height = viewport.height;
  renderTask = current.render({ canvas: canvas.value, viewport });
  try {
    await renderTask.promise;
  } catch (e) {
    if (e.name !== "RenderingCancelledException") error.value = e.message;
  }
}
onMounted(async () => {
  try {
    const response = await fetch(
      `api/media?${new URLSearchParams({ url: props.url })}`,
    );
    if (!response.ok) throw new Error((await response.json()).detail);
    task = getDocument({
      data: new Uint8Array(await response.arrayBuffer()),
      isEvalSupported: false,
    });
    document = await task.promise;
    pages.value = document.numPages;
    await render();
  } catch (e) {
    error.value = e.message;
  } finally {
    loading.value = false;
  }
});
watch(page, render);
onBeforeUnmount(() => {
  renderTask?.cancel();
  task?.destroy();
});
</script>
<template>
  <div class="pdf-view">
    <div class="pdf-controls">
      <span v-if="loading">Loading PDF…</span
      ><template v-else
        ><button :disabled="page <= 1" @click="page--">← Previous</button
        ><span>Page {{ page }} of {{ pages }}</span
        ><button :disabled="page >= pages" @click="page++">
          Next →
        </button></template
      >
    </div>
    <p v-if="error" role="alert">{{ error }}</p>
    <canvas ref="canvas" aria-label="PDF page"></canvas>
  </div>
</template>

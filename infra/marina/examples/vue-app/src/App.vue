<script setup>
import { ref } from "vue";

const count = ref(0);
</script>

<template>
  <main>
    <h1>Vue counter</h1>
    <p>The button has been pressed {{ count }} times.</p>
    <button type="button" @click="count += 1">Increment</button>
  </main>
</template>

<style>
body {
  font: 16px/1.5 system-ui, sans-serif;
}

main {
  max-width: 40rem;
  margin: 3rem auto;
  padding: 0 1rem;
}
</style>

<script setup lang="ts">
withDefaults(defineProps<{ label?: string }>(), { label: 'Loading' })
</script>

<template>
  <svg
    class="oa-grow inline-block shrink-0"
    role="img"
    :aria-label="label"
    viewBox="5.5 -0.4 92 92"
    fill="none"
    stroke="currentColor"
    stroke-width="4.1"
  >
    <path d="M35 84.8A11.3 11.3 0 0 0 46.3 73.5V39.8A13.3 13.3 0 0 0 33 26.5H17a8 8 0 1 0 0 16a8 8 0 1 0 0-16" />
    <path d="M66.3 84.8A11 11 0 0 1 55.3 73.8V21.5A13 13 0 0 1 68.3 8.5H78.3a8 8 0 1 1 0 16a8 8 0 1 1 0-16" />
    <path d="M55.3 55.2A12.7 12.7 0 0 1 68 42.5H86a8 8 0 1 1 0 16a8 8 0 1 1 0-16" />
  </svg>
</template>

<style scoped>
.oa-grow {
  width: 1.5em;
  height: 1.5em;
}
.oa-grow path {
  stroke-dasharray: 151;
  stroke-dashoffset: 0;
  animation: oa-grow 2s ease-in-out infinite;
}
.oa-grow path:nth-of-type(3) {
  animation-name: oa-grow-b;
}
@keyframes oa-grow {
  0% {
    stroke-dashoffset: 151;
    opacity: 1;
  }
  60%,
  85% {
    stroke-dashoffset: 0;
    opacity: 1;
  }
  100% {
    stroke-dashoffset: 0;
    opacity: 0;
  }
}
@keyframes oa-grow-b {
  0%,
  20% {
    stroke-dashoffset: 151;
    opacity: 1;
  }
  62%,
  85% {
    stroke-dashoffset: 60;
    opacity: 1;
  }
  100% {
    stroke-dashoffset: 60;
    opacity: 0;
  }
}
@keyframes oa-fade {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.35;
  }
}
@media (prefers-reduced-motion: reduce) {
  .oa-grow path,
  .oa-grow path:nth-of-type(3) {
    animation: oa-fade 3s ease-in-out infinite;
  }
}
</style>
